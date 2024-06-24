import logging
import requests
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)

def log(message):
    logging.info(message)

class SalesLeadNotifier:
    def __init__(self, email_address, contacts_details):
        self.email_address = email_address
        self.contacts_details = contacts_details
        self.NOTIFICATION_URL = Config.NOTIFICATION_HTTP_API
        self.status = "success"

    def run(self):
        try:
            self.notify_new_leads()
        except Exception as e:
            log(f"Error during sales lead notification: {e}")
            self.status = "failure"
        finally:
            return self.status

    def notify_new_leads(self):
        log(f"Starting notification for new leads to {self.email_address}")
        try:
            log(f"contact details: {self.contacts_details}")#DEBUG
            response = requests.post(
                self.NOTIFICATION_URL, 
                json={"email_address": self.email_address, "contacts_details": self.contacts_details}
            )
        except requests.exceptions.RequestException as e:
            log(f"Failed to send notification: {e}")
            raise

        if response.status_code == 200:
            log(f"Notification sent successfully to {self.email_address}")
        else:
            log(f"Failed to send notification: {response.text}")
            raise Exception(f"Failed to send notification: {response.text}")
