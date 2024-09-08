import io
import json
import logging
import time
from vector_indexing.algo_units.face_indexer import FaceIndexer
import cv2
import numpy as np
from google.cloud import storage

from common.consts_and_utils import BUCKET_NAME, PREPROCESS_FOLDER

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('numba.core').setLevel(logging.INFO)
# change level of debug for urllib3.connectionpool to INFO
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
logging.getLogger('urllib3.urllib3.util').setLevel(logging.INFO)

logging.getLogger('google.auth').setLevel(logging.INFO)

class IndexingService:

    def __init__(self, session_key: str):
        self.session_key = session_key

        try:
            self.storage_client = storage.Client()
        except Exception as e:
            logging.error("Error initializing Cloud Storage client: %s", e)
            raise
        try:
            self.bucket = self.storage_client.bucket(BUCKET_NAME)
        except Exception as e:
            logging.error("Error getting bucket: %s", e)
            raise
        self.preprocess_folder = f"{self.session_key}/{PREPROCESS_FOLDER}"
        
        # Load data from GCS
        self.embeddings, self.orig_image_paths, self.metrics = self.load_data_from_gcs()

        #validate that the lists are of the same length
        if len(self.embeddings) != len(self.orig_image_paths) or len(self.embeddings) != len(self.metrics):
            raise ValueError("The number of embeddings, original image paths, and metrics are not equal")

        #init indexer
        self.face_indexer = FaceIndexer(self.session_key, dimension=len(self.embeddings[0]))

        #log amount of face embeddings at the start of the service
        logging.info(f"Number of face embeddings at init of indexing service: {len(self.embeddings)}")

    def load_data_from_gcs(self):
        embeddings = self.download_embeddings()
        orig_image_paths = self.download_orig_paths()
        metrics  = self.download_metrics()
        return embeddings, orig_image_paths, metrics

    def download_embeddings(self):
        embeddings_blob = self.bucket.blob(f"{self.session_key}/preprocess/embeddings.npy")
        embeddings_bytes = embeddings_blob.download_as_bytes()
        embeddings = np.load(io.BytesIO(embeddings_bytes), allow_pickle=True)
        return embeddings

    def download_orig_paths(self):
        paths_blob = self.bucket.blob(f"{self.session_key}/preprocess/original_paths.json")
        paths_json = paths_blob.download_as_text()
        orig_paths = json.loads(paths_json)
        return orig_paths

    def download_metrics(self):
        """Download and return the face metrics from GCS as a list of dictionaries."""
        metrics_blob = self.bucket.blob(f"{self.session_key}/preprocess/metrics.json")
        metrics_json = metrics_blob.download_as_text()
        
        # Deserialize JSON string to Python objects
        metrics_list = json.loads(metrics_json)

        # Convert each dictionary's fields to the appropriate data types
        for metrics in metrics_list:
            metrics['face_alignment_score'] = float(metrics['face_alignment_score'])
            metrics['face_distance_score'] = float(metrics['face_distance_score'])
            metrics['face_detection_prob'] = float(metrics['face_detection_prob'])
            metrics['faces_count'] = int(metrics['faces_count'])
            metrics['face_position_score'] = float(metrics['face_position_score'])
            metrics['laplacian_variance_image'] = float(metrics['laplacian_variance_image'])
            metrics['laplacian_variance_face'] = float(metrics['laplacian_variance_face'])
            metrics['tag_position'] = tuple(map(int, metrics['tag_position']))
            metrics['face_brightness'] = float(metrics['face_brightness'])
            metrics['face_contrast'] = float(metrics['face_contrast'])
            metrics['face_to_image_ratio'] = float(metrics['face_to_image_ratio'])

        return metrics_list

    def index_faces(self):
        try:
            # Send embeddings, original image paths, and metrics to the indexer
            self.face_indexer.index_faces(embeddings=self.embeddings, image_paths=self.orig_image_paths, metrics=self.metrics)
        except Exception as e:
            logging.error(f"Error indexing faces: {e}")
            raise

    def execute(self):
        # Index the faces for vector search in real-time
        self.index_faces()

        # Wait for index to be ready before letting the service finish
        while not self.face_indexer.check_index_ready():
            # sleep for 5 seconds
            time.sleep(5)        
