import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import pickle
import sys
import logging
import time
import gc  # Garbage collector
from algo_units.preprocess import PeepsPreProcessor
from google.cloud import storage
from common.consts_and_utils import BUCKET_NAME, CHUNK_SIZE, MAX_WORKERS, RAW_DATA_FOLDER
from preprocessing.gcs_utils import get_image_paths_from_bucket
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables from .env file (useful for local development)
from dotenv import load_dotenv
load_dotenv()

PREPROCESS_TMP = "preprocess/tmp"

def log_process_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"{label} Process ID {os.getpid()} memory usage: "
                f"RSS = {mem_info.rss} bytes ({mem_info.rss/1024/1024:.2f} MB), "
                f"VMS = {mem_info.vms} bytes ({mem_info.vms/1024/1024:.2f} MB)")


def store_chunk_result(chunk_result, chunk_idx, bucket, session_key):
    """
    Serializes and uploads the given chunk_result (tuple of (results_list, paths_list))
    to the Cloud Storage bucket provided, under a session-specific tmp folder.
    """
    data = pickle.dumps(chunk_result)
    session_prefix = f"{session_key}/{PREPROCESS_TMP}"
    blob_name = f"{session_prefix}/chunk_{chunk_idx}.pkl"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="application/octet-stream")
    file_size = len(data)
    logger.info(f"Uploaded chunk {chunk_idx} to gs://{bucket.name}/{blob_name}, size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")

def load_chunk_result(chunk_idx, bucket, session_key):
    """
    Downloads and deserializes the chunk_result (tuple of (results_list, paths_list))
    from the Cloud Storage bucket provided, under the session-specific tmp folder.
    """
    session_prefix = f"{session_key}/{PREPROCESS_TMP}"
    blob_name = f"{session_prefix}/chunk_{chunk_idx}.pkl"
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    return pickle.loads(data)

def log_total_chunk_files_size(bucket, session_key):
    """
    Logs the total size of all chunk files in the bucket for the given session.
    """
    session_prefix = f"{session_key}/{PREPROCESS_TMP}"
    total_size = 0
    blobs = list(bucket.list_blobs(prefix=f"{session_prefix}/chunk_"))
    for blob in blobs:
        total_size += blob.size
    logger.info(f"Total size of chunk files in bucket {bucket.name} for session {session_key}: {total_size} bytes ({total_size/1024/1024:.2f} MB)")
    return total_size

def delete_chunk_files(bucket, session_key):
    """
    Deletes all intermediate chunk files from the Cloud Storage bucket for the given session.
    """
    session_prefix = f"{session_key}/{PREPROCESS_TMP}"
    total_deleted = 0
    blobs = list(bucket.list_blobs(prefix=f"{session_prefix}/chunk_"))
    for blob in blobs:
        size = blob.size
        blob.delete()
        total_deleted += size
        logger.info(f"Deleted intermediate blob: {blob.name} (size: {size} bytes)")
    logger.info(f"Total deleted chunk files size for session {session_key}: {total_deleted} bytes ({total_deleted/1024/1024:.2f} MB)")

def preprocess_chunk(session_key_image_paths_tuple):
    try:
        log_process_memory("Before execution:")
        session_key, image_paths = session_key_image_paths_tuple
        preprocessor = PeepsPreProcessor(session_key=session_key)
        result = preprocessor.execute(image_paths)  # returns a tuple: (results list, paths_list)
        log_process_memory("After execution:")
        return result
    except Exception as e:
        logger.error("Error in process:", exc_info=e)


def divide_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(session_key):
    try:
        start_pre = time.time()

        # Initialize the Cloud Storage client once for both image retrieval and intermediate storage
        client = storage.Client()

        # Get image paths from the source bucket using BUCKET_NAME
        image_paths = get_image_paths_from_bucket(session_key, client, BUCKET_NAME, RAW_DATA_FOLDER)
        num_cpus = max(1, MAX_WORKERS)
        logger.info(f"Number of CPUs used for multiprocessing: {num_cpus}")

        # Divide image paths into chunks (if fewer than CHUNK_SIZE, use one chunk)
        chunks = list(divide_chunks(image_paths, CHUNK_SIZE)) if len(image_paths) > CHUNK_SIZE else [image_paths]

        # Use BUCKET_NAME for intermediate storage by getting the bucket instance
        intermediate_bucket = client.bucket(BUCKET_NAME)

        # Process each chunk in parallel and upload intermediate results under the session-specific tmp folder
        chunk_count = 0
        with ProcessPoolExecutor(max_workers=num_cpus, max_tasks_per_child=1) as executor:
            for chunk_result in executor.map(preprocess_chunk, [(session_key, chunk) for chunk in chunks]):
                logger.info(f"Chunk {chunk_count} processed with {len(chunk_result[0])} results and {len(chunk_result[1])} paths.")
                logger.info("Uploading intermediate results to Cloud Storage")
                store_chunk_result(chunk_result, chunk_count, intermediate_bucket, session_key)
                chunk_count += 1
                gc.collect()

        logger.info(f"Total chunks saved: {chunk_count}")
        log_total_chunk_files_size(intermediate_bucket, session_key)

        # Load all chunks back from the bucket for final aggregation
        extended_results = []
        extended_paths_times = []
        for idx in range(chunk_count):
            results_list, paths_list = load_chunk_result(idx, intermediate_bucket, session_key)
            extended_results.extend(results_list)
            extended_paths_times.extend(paths_list)

        # Create an instance of PeepsPreProcessor for uploading aggregated artifacts
        preprocessor = PeepsPreProcessor(session_key=session_key)
        preprocessor.store_aggregated_artifacts_to_gcs(extended_results, extended_paths_times)

        logger.info(f"all_results len: {len(preprocessor.results)}")
        if preprocessor.results:
            logger.info(f"all_results[0] len: {len(preprocessor.results[0])}")

        end_pre = time.time()
        preprocess_time = end_pre - start_pre

        logger.info("Deleting intermediate chunk files from Cloud Storage...")
        delete_chunk_files(intermediate_bucket, session_key)

        return {"status": "success", "message": f"Preprocessing completed successfully. Time elapsed: {preprocess_time:.2f} seconds"}

    except Exception as e:
        logger.exception("Error during preprocessing", exc_info=e)
        sys.exit(1)

if __name__ == "__main__":
    # Retrieve the session key from an environment variable
    session_key = os.getenv("SESSION_KEY")
    if not session_key:
        logger.error("No session key provided. Set SESSION_KEY environment variable.")
        sys.exit(1)

    logger.info(f"Session key: {session_key}")
    result = main(session_key)
    logger.info(result)
