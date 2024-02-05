# real_time_image_retriever.py
import json
import logging
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics import euclidean_distances
import torch
from google.cloud import storage

from common.consts_and_utils import BUCKET_NAME

from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def chebyshev_distance(a, b):
    return np.max(np.abs(a - b))


class PeepsClusterRetriever:
    def __init__(self):
        self.session_key = None
        self.selfie_image = None  # Assuming selfie_image is a numpy array

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
        self.mtcnn = MTCNN(
            image_size=160, margin=80, min_face_size=85,
            thresholds=[0.6, 0.7, 0.7], factor=0.65, post_process=True,
            device=self.device
        ).eval()

    def load_new_image(self, session_key:str, new_selfie_image):
        self.session_key = session_key
        self.selfie_image = new_selfie_image

    def download_cluster_centroids(self):
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(f'{self.session_key}/clusters/cluster_centroids.json')
        centroids_json = blob.download_as_string()
        return json.loads(centroids_json)

    def process_image(self):
        try:
            rgb_image = cv2.cvtColor(self.selfie_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (160, 160))  # Resize for optimization

            with torch.no_grad():  # Use torch.no_grad() for both MTCNN and ResNet
                # Detect and extract face
                face_crop, prob = self.mtcnn(rgb_image, return_prob=True)
                if face_crop is None:
                    return {"error": "No face detected. Please try a different image."}

                # Embedding extraction
                face_crop = face_crop.unsqueeze(0).to(self.device)
                embedding = self.resnet(face_crop).squeeze().cpu().numpy()

            return {"embedding": embedding}
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return {"error": "An error occurred while processing the image. Please try again."}

    def retrieve_top_k_candidates(self, embedding, k=4, distance_metric='cosine'):
        centroids = self.download_cluster_centroids()

        # Distance function mappings
        distance_funcs = {
            'cosine': (cosine_similarity, True),
            'euclidean': (euclidean_distance, False),
            'manhattan': (manhattan_distance, False),
            'chebyshev': (chebyshev_distance, False)
        }

        if distance_metric not in distance_funcs:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        distance_func, reverse_sort = distance_funcs[distance_metric]

        # Compute distances
        distances = {cluster_id: distance_func(embedding, np.array(centroid))
                    for cluster_id, centroid in centroids.items()}

        # Sort and select top k clusters
        sorted_clusters = sorted(distances.items(), key=lambda item: item[1], reverse=reverse_sort)[:k]

        return [(cluster_id, distance) for cluster_id, distance in sorted_clusters]
