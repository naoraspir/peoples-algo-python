import gc
import logging
import os
import sys
import time
import re
from pipeline_executor import PipelineExecutor
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter
from google.api_core.exceptions import PermissionDenied

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables from .env file (useful for local development)
load_dotenv()

def is_valid_email(email):
    if email is None:
        return False
    # Basic regex for email validation
    email_regex = re.compile(r"[^@]+@[^@]+\.[^@]+")
    return re.match(email_regex, email) is not None

def get_sessions_from_firestore():
    try:
        client = firestore.Client(database="peoples-prod")
        coupons_ref = client.collection("coupons")
        query = coupons_ref.where(filter=FieldFilter("workflowStatus", "==", "uploaded"))
        sessions = []
        default_email = os.getenv("EMAIL_ADDRESS")
        
        if not default_email:
            logger.error("No default email address provided. Set EMAIL_ADDRESS environment variable.")
            sys.exit(1)
        
        logger.info("Executing Firestore query to find pending sessions.")
        for doc in query.stream():
            data = doc.to_dict()
            session_key = doc.id
            email = data.get('email', default_email)
            photographer_name = data.get('photographerName', 'N/A')
            
            if not is_valid_email(email):
                logger.warning(f"Invalid email address {email} for session {session_key}. Using default email {default_email}")
                email = default_email
                # Update the Firestore document to set the email to the default email
                try:
                    doc_ref = client.collection("coupons").document(session_key)
                    doc_ref.update({"email": default_email})
                    logger.info(f"Updated session {session_key} email to {default_email}")
                except Exception as e:
                    logger.error(f"Failed to update session {session_key} email to {default_email}: {e}")
            
            logger.info(f"Found session: {session_key}, Email: {email}, Photographer: {photographer_name}")
            sessions.append((session_key, email))
        
        if not sessions:
            logger.info("No pending sessions found.")
        
        return sessions
    except PermissionDenied as e:
        logger.error(f"Permission denied when accessing Firestore: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while querying Firestore: {e}")
        sys.exit(1)

def update_session_status(session_key, status):
    try:
        client = firestore.Client(database="peoples-prod")
        doc_ref = client.collection("coupons").document(session_key)
        logger.info(f"Attempting to update session {session_key} status to {status}")
        doc_ref.update({"workflowStatus": status})
        logger.info(f"Updated session {session_key} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update session {session_key} status to {status}: {e}")
        raise

def run_pipeline(session_key, email_address):
    try:
        # Set status to "runningAlgo"
        update_session_status(session_key, "runningAlgo")
        
        # Measure time
        start_pipeline_execution = time.time()
        
        # Init the pipeline executor with the provided session key
        executor = PipelineExecutor(session_key, email_address)
        status = executor.run()
        
        # Measure time
        end_pipeline_execution = time.time()
        execution_time = end_pipeline_execution - start_pipeline_execution
        
        # Delete the executor instance and its attributes to free memory
        del executor
        gc.collect()  # Explicitly invoke garbage collection

        if status == "failure":
            update_session_status(session_key, "uploaded")
            raise Exception("Pipeline execution failed")
        # Set status to "ready" upon successful completion
        update_session_status(session_key, "ready")

        # Return success message with time taken for the pipeline to run
        return {"status": "success", "message": f"Pipeline completed successfully. Time elapsed: {execution_time:.2f} seconds"}
    
    except Exception as e:
        logger.exception("Error during pipeline execution", exc_info=e)
        
        # Set status back to "uploaded" upon failure
        update_session_status(session_key, "uploaded")
        
        # Return error message and exit with status code 1
        sys.exit(1)

if __name__ == "__main__":
    # Retrieve the session key and email address from environment variables
    session_key = os.getenv("SESSION_KEY")
    default_email_address = os.getenv("EMAIL_ADDRESS")
    if not session_key:
        logger.error("No session key provided. Set SESSION_KEY environment variable.")
        sys.exit(1)
    if not default_email_address:
        logger.error("No email address provided. Set EMAIL_ADDRESS environment variable.")
        sys.exit(1)

    if session_key.lower() == "none":
        logger.info("SESSION_KEY is 'none', querying Firestore for pending sessions.")
        sessions = get_sessions_from_firestore()
        if not sessions:
            logger.info("No pending sessions found.")
            sys.exit(0)

        for session_key, email_address in sessions:
            logger.info(f"Running pipeline for session key: {session_key} and email: {email_address}")
            result = run_pipeline(session_key, email_address)
            logger.info(result)
    else:
        logger.info(f"Running pipeline for session key: {session_key} and email: {default_email_address}")
        result = run_pipeline(session_key, default_email_address)
        logger.info(result)
