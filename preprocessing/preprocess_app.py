import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import sys
import logging
import time
import gc  # Garbage collector
from algo_units.preprocess import PeepsPreProcessor
from google.cloud import storage
from common.consts_and_utils import BUCKET_NAME, MAX_WORKERS, RAW_DATA_FOLDER

from preprocessing.gcs_utils import get_image_paths_from_bucket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables from .env file (useful for local development)
from dotenv import load_dotenv
load_dotenv()

def preprocess_chunk(session_key_image_paths_tuple):
    try:
        session_key, image_paths = session_key_image_paths_tuple
        preprocessor = PeepsPreProcessor(session_key=session_key)
        
        return preprocessor.execute(image_paths)  # returns a tuple: (results list for each face, (path, datetime_taken) for each image) 
    except Exception as e:
        logger.error("Error in process:", exc_info=e)

def divide_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(session_key):
    try:
        # Measure time
        start_pre = time.time()
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket_name = BUCKET_NAME
        raw_folder = RAW_DATA_FOLDER
        # Retrieve all image paths 
        image_paths = get_image_paths_from_bucket(session_key, storage_client, bucket_name, raw_folder)
        num_cpus = max(1, MAX_WORKERS)
        logger.info(f"Number of CPUs used for multiprocessing: {num_cpus}")

        # Divide image paths into chunks
        chunks = list(divide_chunks(image_paths, len(image_paths) // num_cpus))

        extended_results = []
        extended_paths_times = []
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            for chunk_result in executor.map(preprocess_chunk, [(session_key, chunk) for chunk in chunks]):
                extended_results.extend(chunk_result[0])
                extended_paths_times.extend(chunk_result[1])


        # log len of results before uploading and len of paths_times
        logger.info(f"extended_results len: {len(extended_results)}")
        logger.info(f"extended_paths_times len: {len(extended_paths_times)}")
        # Create an instance of PeepsPreProcessor for uploading
        preprocessor = PeepsPreProcessor(session_key=session_key)
        preprocessor.store_aggregated_artifacts_to_gcs(extended_results, extended_paths_times)
        
        # Logging results
        logger.info(f"all_results len: {len(preprocessor.results)}")
        if preprocessor.results:
            logger.info(f"all_results[0] len: {len(preprocessor.results[0])}")
        
        # Measure time
        end_pre = time.time()
        preprocess_time = end_pre - start_pre
        
        # Return success message with time taken
        return {"status": "success", "message": f"Preprocessing completed successfully. Time elapsed: {preprocess_time:.2f} seconds"}
    
    except Exception as e:
        logging.exception("Error during preprocessing", exc_info=e)
        # Return error message and exit with status code 1
        sys.exit(1)

if __name__ == "__main__":
    # Retrieve the session key from an environment variable
    session_key = os.getenv("SESSION_KEY")
    if not session_key:
        logger.error("No session key provided. Set SESSION_KEY environment variable.")
        sys.exit(1)

    logger.info(f"Session key: {session_key}")

    # Run the main function and log the result
    result = main(session_key)
    logger.info(result)
