import logging
import os
import sys
import re
from sales_lead_notifier import SalesLeadNotifier
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter
from google.api_core.exceptions import PermissionDenied
from config import Config

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

def get_contacts_from_firestore():
    try:
        client = firestore.Client(project=Config.FIRESTORE_DATABASE)
        contacts_ref = client.collection(Config.FIRESTORE_COLLECTION)
        query = contacts_ref.where(filter=FieldFilter(Config.FIRESTORE_STATUS_FIELD, "==", Config.FIRESTORE_STATUS_NOT_NOTIFIED))
        contacts = []
        
        logger.info("Executing Firestore query to find new contacts.")
        for doc in query.stream():
            data = doc.to_dict()
            contact_key = doc.id
            
            # Log contact details
            logger.info(f"Found contact: {contact_key}")
            contacts.append((contact_key, data))
        
        return contacts
    except PermissionDenied as e:
        logger.error(f"Permission denied when accessing Firestore: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while querying Firestore: {e}")
        sys.exit(1)

def update_contact_status(contact_key, status):
    try:
        client = firestore.Client(project=Config.FIRESTORE_DATABASE)
        doc_ref = client.collection(Config.FIRESTORE_COLLECTION).document(contact_key)
        logger.info(f"Attempting to update contact {contact_key} status to {status}")
        doc_ref.update({Config.FIRESTORE_STATUS_FIELD: status})
        logger.info(f"Updated contact {contact_key} status to {status}")
    except Exception as e:
        logger.error(f"Failed to update contact {contact_key} status to {status}: {e}")
        raise

if __name__ == "__main__":
    # Retrieve the email address from environment variables
    email_address = os.getenv("EMAIL_ADDRESS")
    if not email_address:
        logger.error("No email address provided. Set EMAIL_ADDRESS environment variable.")
        sys.exit(1)

    logger.info("Querying Firestore for new contacts.")
    contacts = get_contacts_from_firestore()
    if not contacts:
        logger.info("No new contacts to notify.")
        sys.exit(0)

    logger.info(f"Sending notification for {len(contacts)} new contacts.")
    notifier = SalesLeadNotifier(email_address, [contact[1] for contact in contacts])
    status = notifier.run()
    
    if status == "success":
        for contact_key, _ in contacts:
            update_contact_status(contact_key, Config.FIRESTORE_STATUS_NOTIFIED)

    logger.info(f"Notification process completed with status: {status}")
