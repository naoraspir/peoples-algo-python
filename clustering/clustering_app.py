import os
import sys
import logging
import time
import gc  # Garbage collector
from algo_units.clustering import FaceClustering

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables from .env file (useful for local development)
from dotenv import load_dotenv
load_dotenv()

def main(session_key):
    try:
        # Measure time
        start_clustering = time.time()
        
        # Initialize the face clusterer with the provided session key
        clusterer = FaceClustering(session_key=session_key)
        
        # Execute the clustering, postprocessing, and uploading intermediate data and images to GCS
        clusterer.execute()
        
        # Measure time
        end_clustering = time.time()
        clustering_time = end_clustering - start_clustering
        
        # Delete the clusterer instance and its attributes to free memory
        del clusterer
        gc.collect()  # Explicitly invoke garbage collection
        
        # Return success message with time taken
        return {"status": "success", "message": f"clustering completed successfully. Time elapsed: {clustering_time:.2f} seconds"}
    
    except Exception as e:
        logging.exception("Error during clustering", exc_info=e)
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
