import os
import sys
import logging
import time
import gc  # Garbage collector
from algo_units.preprocess import PeepsPreProcessor
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables from .env file (useful for local development)
from dotenv import load_dotenv
load_dotenv()

def main(session_key):
    try:
        # Measure time
        start_pre = time.time()
        
        # Initialize the PeepsPreProcessor with the provided session key
        preprocessor = PeepsPreProcessor(session_key=session_key)
        
        # Execute the preprocessing, embedding, and uploading intermediate data and images to GCS
        preprocessor.execute()
        
        # Logging results
        logger.info(f"all_results len: {len(preprocessor.results)}")
        if preprocessor.results:
            logger.info(f"all_results[0] len: {len(preprocessor.results[0])}")
        
        # Measure time
        end_pre = time.time()
        preprocess_time = end_pre - start_pre
        
        # Delete the preprocessor instance and its attributes to free memory
        del preprocessor
        gc.collect()  # Explicitly invoke garbage collection
        
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
