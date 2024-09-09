import logging
import os
from google.cloud import run_v2
from google.cloud.run_v2.types import RunJobRequest
from google.cloud import storage
import requests
from config import Config
from common.consts_and_utils import BUCKET_NAME, RAW_DATA_FOLDER
from google.cloud.exceptions import NotFound

# Set up logging
logging.basicConfig(level=logging.INFO)

def log(message):
    logging.info(message)

class PipelineExecutor:
    def __init__(self, session_key, email_address):
        self.session_key = session_key
        self.email_address = email_address
        self.environment = os.getenv("RUN_ENV", "prod")  # Default to 'prod' if not set
        # Set the notification URL according to the email address given
        self.NOTIFICATION_URL = Config.NOTIFICATION_HTTP_API
        self.steps = [
            self.clean_bucket,
            self.preprocess,
            self.indexing,
            self.cluster,
            self.notify_completion
        ]
        self.status = "success"

    def run(self):
        try:
            for step in self.steps[:-1]:  # Run all steps except the last one (notify_completion)
                step()
        except Exception as e:
            log(f"Error during pipeline execution: {e}")
            self.status = "failure"
        finally:
            self.notify_completion()
            return self.status

    def clean_bucket(self):
        log(f"Starting cleaning step for bucket {BUCKET_NAME} and session {self.session_key}")
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blobs = bucket.list_blobs(prefix=self.session_key)
            for blob in blobs:
                if not blob.name.startswith(f"{self.session_key}/{RAW_DATA_FOLDER}/"):
                    try:
                        blob.delete()
                    except NotFound:
                        log(f"Object {blob.name} not found, skipping deletion.")
                    except Exception as e:
                        log(f"Failed to delete {blob.name}: {e}")
            log("Cleaning completed successfully")
        except Exception as e:
            log(f"Cleaning bucket failed: {e}")
            raise

    def preprocess(self):
        log("Starting preprocessing step")
        try:
            client = run_v2.JobsClient()

            # Determine job name based on environment
            if self.environment == "prod":
                job_name = "projects/peoples-software/locations/us-east1/jobs/preprocessing-job"
            else:
                job_name = "projects/peoples-software/locations/us-east1/jobs/preprocessing-job-dev"

            override_spec = {
                'container_overrides': [
                    {
                        'env': [
                            {'name': 'SESSION_KEY', 'value': self.session_key}
                        ]
                    }
                ],
                "timeout": str(Config.TIMEOUT) + "s",
            }

            request = RunJobRequest(
                name=job_name,
                overrides=override_spec
            )

            operation = client.run_job(request=request, timeout=Config.TIMEOUT)
            log("Waiting for operation to complete...")

            response = operation.result(timeout=Config.TIMEOUT)
            log(f"Operation result: {response}")

            log("Preprocessing job completed successfully")
        except Exception as e:
            log(f"Preprocessing job failed: {e}")
            raise

    def indexing(self):
        log("Starting indexing step")
        try:
            client = run_v2.JobsClient()

            # Determine job name based on environment
            if self.environment == "prod":
                job_name = "projects/peoples-software/locations/us-central1/jobs/indexing-job"
            else:
                job_name = "projects/peoples-software/locations/us-central1/jobs/indexing-job-dev"

            override_spec = {
                'container_overrides': [
                    {
                        'env': [
                            {'name': 'SESSION_KEY', 'value': self.session_key}
                        ]
                    }
                ],
                "timeout": str(Config.TIMEOUT) + "s",
            }

            request = RunJobRequest(
                name=job_name,
                overrides=override_spec
            )

            operation = client.run_job(request=request, timeout=Config.TIMEOUT)
            log("Waiting for operation to complete...")

            response = operation.result(timeout=Config.TIMEOUT)
            log(f"Operation result: {response}")

            log("Indexing job completed successfully")
        except Exception as e:
            log(f"Indexing job failed: {e}")
            raise

    def cluster(self):
        log("Starting clustering step")
        try:
            client = run_v2.JobsClient()

            # Determine job name based on environment
            if self.environment == "prod":
                job_name = "projects/peoples-software/locations/us-east1/jobs/clustering-job"
            else:
                job_name = "projects/peoples-software/locations/us-east1/jobs/clustering-job-dev"

            override_spec = {
                'container_overrides': [
                    {
                        'env': [
                            {'name': 'SESSION_KEY', 'value': self.session_key}
                        ]
                    }
                ],
                "timeout": str(Config.TIMEOUT) + "s",
            }

            request = RunJobRequest(
                name=job_name,
                overrides=override_spec
            )

            operation = client.run_job(request=request, timeout=Config.TIMEOUT)
            log("Waiting for operation to complete...")

            response = operation.result(timeout=Config.TIMEOUT)
            log(f"Operation result: {response}")

            log("Clustering job completed successfully")
        except Exception as e:
            log(f"Clustering job failed: {e}")
            raise

    def notify_completion(self):
        if self.status == "failure":
            log(f"Pipeline execution failed for session key: {self.session_key}")
            return
        log(f"Starting notification step for session key: {self.session_key}")
        try:
            response = requests.post(self.NOTIFICATION_URL, json={"session_key": self.session_key})
        except requests.exceptions.RequestException as e:
            log(f"Failed to send notification for session key {self.session_key}: {e}")
            return
        if response.status_code == 200:
            log(f"Notification sent successfully for session key {self.session_key}")
        else:
            log(f"Failed to send notification for session key {self.session_key}: {response.text}")
