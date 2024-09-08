import os
import sys
import logging
import time
import gc  # Garbage collector
from vector_indexing.algo_units.indexing import IndexingService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables from .env file (useful for local development)
from dotenv import load_dotenv
load_dotenv()

def main(session_key):
    try:
        # Measure time
        start_indexing = time.time()
        
        # Initialize the face indexing service with the provided session key
        indexer = IndexingService(session_key=session_key)
        
        # Execute the indexing process: loading data, vector indexing, and waiting for index readiness
        indexer.execute()
        
        # Measure time
        end_indexing = time.time()
        indexing_time = end_indexing - start_indexing
        
        # Delete the indexer instance and its attributes to free memory
        del indexer
        gc.collect()  # Explicitly invoke garbage collection
        
        # Return success message with time taken
        return {"status": "success", "message": f"Indexing completed successfully. Time elapsed: {indexing_time:.2f} seconds"}
    
    except Exception as e:
        logger.exception("Error during indexing", exc_info=e)
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
