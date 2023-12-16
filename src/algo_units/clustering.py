import io
import json
import logging
import numpy as np
from google.cloud import storage
from sklearn.cluster import DBSCAN
import umap
from scipy.spatial.distance import cosine

from sklearn.preprocessing import normalize
from consts_and_utils import BUCKET_NAME
from typing import List
import hdbscan

logging.basicConfig(level=logging.DEBUG)

class FaceClustering:

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
    
        self.face_paths: List[str] = []
        self.orig_image_paths: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def load_data(self):
        try:
            # logging.info("Loading face embeddings and paths from Cloud Storage")
            
            prefix = f"{self.session_key}/preprocess"

            blobs = self.bucket.list_blobs(prefix=prefix)
            blob_count = sum(1 for _ in self.bucket.list_blobs(prefix=prefix))
            # logging.info("Found %s blobs with prefix %s", blob_count, prefix)

            for blob in blobs:
                try:
                    if blob.name.endswith("embedding.npy"):
                        
                        # logging.debug("Loading embedding for %s", blob.name)
                        embedding = self.download_embedding_from_gcs(blob.name)
                        
                        face_path = blob.name.replace("embedding.npy", "face.jpg")
                        
                        # logging.debug("Getting original image path for %s", blob.name)
                        orig_path = self.get_orig_path(blob)
                        
                        self.face_paths.append(face_path)
                        self.orig_image_paths.append(orig_path)
                        self.embeddings.append(embedding)
                        
                except Exception as e:
                    logging.error("Error processing blob %s: %s", blob.name, e)
                    continue
                    
            logging.info("Loaded %s embeddings", len(self.embeddings))
            
        except Exception as e:
            logging.error("Error loading data: %s", e)
            raise

    def get_orig_path(self, blob):
        try:
            orig_txt = blob.name.replace("embedding.npy", "original_path.txt")
            
            # logging.debug("Downloading original text file %s", orig_txt)
            text = self.bucket.blob(orig_txt).download_as_string()
            
            # logging.debug("Decoded text file %s", orig_txt)
            return text.decode("utf-8")
        
        except Exception as e:
            logging.error("Error getting original path for %s: %s", blob.name, e)
            raise

    def download_embedding_from_gcs(self, embedding_path: str) -> np.ndarray:
        try:
            # Download the bytes from GCS
            blob = self.bucket.blob(embedding_path)
            embedding_bytes = blob.download_as_bytes()

            # Use BytesIO to convert bytes back into a NumPy array
            embedding_buffer = io.BytesIO(embedding_bytes)
            embedding = np.load(embedding_buffer, allow_pickle=True)
            return embedding
        except Exception as e:
            logging.error(f"Error downloading embedding: {e}")
            raise
    
    def calculate_noise_threshold(self, distances):
        # Implement your method to define the noise threshold
        # Example: Use a percentile based threshold
        return np.percentile(distances, 95)

    def handle_noise_points(self, noise_points):
        # Assuming self.face_paths and self.bucket are defined
        # noise_points = np.where(cluster_labels == -1)[0]
        logging.info("Identified %s noise points", len(noise_points))

        logging.info("Saving noise points to Cloud Storage")
        noise_folder = f"{self.session_key}/clusters/noise"
        
        # Iterate over all noise points
        for i in noise_points:
            try:
                # Construct the GCS blob name for the face image
                face_blob_name = self.face_paths[i]  # Full GCS URI for the face image
                noise_face_blob_name = f"{noise_folder}/{i}.jpg"  # New blob name in the noise folder
                
                # Copy the face image to the noise folder
                face_blob = self.bucket.blob(face_blob_name)
                self.bucket.copy_blob(face_blob, self.bucket, noise_face_blob_name)
                logging.debug(f"Moved face image to {noise_face_blob_name}")

                # Now, upload the original image path as a text file to the noise folder
                orig_path = self.orig_image_paths[i]  # The original image path
                orig_blob_content = orig_path.encode('utf-8')  # Encoding the path to bytes
                orig_blob_name = f"{noise_folder}/{i}.txt"  # The blob name for the original path text file
                
                # Create a new blob for the original image path and upload the content
                orig_blob = self.bucket.blob(orig_blob_name)
                orig_blob.upload_from_string(orig_blob_content, content_type='text/plain')
                # logging.debug(f"Uploaded original image path to {orig_blob_name}")
                
            except Exception as e:
                logging.error(f"Error saving noise point {i}: {e}")

    def cluster(self):
        try:
            array_embeddings = np.array(self.embeddings)
            normalized_embeddings = normalize(array_embeddings)

            # Using UMAP for dimensionality reduction
            umap_model = umap.UMAP(metric='cosine', n_components=75)
            umap_embeddings = umap_model.fit_transform(normalized_embeddings)

            # Clustering with HDBSCAN
            clusterer = hdbscan.HDBSCAN(min_cluster_size=4, cluster_selection_method='eom', core_dist_n_jobs=-1)
            cluster_labels = clusterer.fit_predict(umap_embeddings)
            # cluster_labels = clusterer.fit_predict(normalized_embeddings)

            # Clustering with DBSCAN
            # clusterer = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
            # cluster_labels = clusterer.fit_predict(array_embeddings)

            # Filter out noise (-1) cluster
            valid_cluster_labels = cluster_labels[cluster_labels != -1]
            unique_clusters = set(valid_cluster_labels)
            numUniqueFaces = len(unique_clusters)
            logging.info(f"# unique faces (excluding noise): {numUniqueFaces}")

            return cluster_labels
        except Exception as e:
            logging.error(f"Error clustering embeddings: {e}")
            raise


    def get_cluster_reps(self, cluster_labels):

        cluster_reps = {}
        
        try:
            logging.info("Computing representative for each cluster")
            
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:
                    continue
                # logging.debug("Getting embeddings for cluster %s", cluster_id)
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_embeddings = np.array(self.embeddings)[cluster_indices]
                
                # logging.debug("Computing centroid for cluster %s", cluster_id)
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # logging.debug("Finding closest embedding to centroid in cluster %s", cluster_id)
                closest_index = cluster_indices[np.argmin(np.linalg.norm(cluster_embeddings - centroid, axis=1))]
                
                try:
                    # logging.debug("Getting paths for representative embedding in cluster %s", cluster_id)
                    
                    rep_face_path = self.face_paths[closest_index]
                    rep_orig_path = self.orig_image_paths[closest_index]
                    
                except Exception as e:
                    logging.error("Error getting paths for cluster %s: %s", cluster_id, e)
                    continue
            
                # logging.debug("Adding representative for cluster %s to dictionary", cluster_id)  
                cluster_reps[cluster_id] = {
                    "face_path": rep_face_path,
                    "orig_path": rep_orig_path,
                    "centroid": centroid,
                    "cluster_embeddings": cluster_embeddings.tolist() # convert numpy array to list for JSON serialization
                }
            
            logging.info("Computed %s cluster representatives", len(cluster_reps))
            
        except Exception as e:
            logging.error("Error computing cluster representatives: %s", e)
            raise
            
        return cluster_reps

    def get_cluster_medoids(self, cluster_labels):
        cluster_medoids = {}
        try:
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:
                    continue

                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_embeddings = np.array(self.embeddings)[cluster_indices]

                # Compute pairwise distances and find the medoid
                pairwise_distances = np.linalg.norm(cluster_embeddings[:, None] - cluster_embeddings, axis=2)
                avg_distances = np.mean(pairwise_distances, axis=1)
                medoid_index = cluster_indices[np.argmin(avg_distances)]

                # Store necessary information
                cluster_medoids[cluster_id] = {
                    "face_path": self.face_paths[medoid_index],
                    "orig_path": self.orig_image_paths[medoid_index],
                    "centroid": self.embeddings[medoid_index],
                    "cluster_embeddings": cluster_embeddings.tolist(),
                }
        except Exception as e:
            logging.error("Error computing cluster medoids: %s", e)
            raise

        return cluster_medoids

    def save_noise_cluster_faces(self, cluster_labels):
        noise_cluster_folder = f"{self.session_key}/clusters/noise_cluster/"
        noise_indices = np.where(cluster_labels == -1)[0]

        for idx in noise_indices:
            # Use the stored face path for the noise index
            cropped_face_path = self.face_paths[idx]

            # The destination path in the bucket for the noise cluster
            destination_blob_name = f"{noise_cluster_folder}noise_face_{idx}.jpg"

            # Copy the cropped face image to the noise cluster folder
            cropped_face_blob = self.bucket.blob(cropped_face_path)
            self.bucket.copy_blob(cropped_face_blob, self.bucket, destination_blob_name)
            logging.debug(f"Copied cropped face to {destination_blob_name}")

    def cosine_distance(self, vec1, vec2):
        """
        Calculate the adjusted cosine distance between two vectors.
        """
        return cosine(vec1, vec2)


    def save_clusters(self, cluster_labels, cluster_reps):
        try:
            logging.info("Saving %s clusters to Cloud Storage", len(cluster_reps))
            
            cluster_sizes = {}
            #save noise cluster faces
            self.save_noise_cluster_faces(cluster_labels)
            # Filter out the noise cluster
            valid_cluster_ids = [cluster_id for cluster_id in np.unique(cluster_labels) if cluster_id != -1]
            
            for cluster_id in np.unique(valid_cluster_ids):

                cluster_folder = f"{self.session_key}/clusters/{cluster_id}/"
                faces_folder = f"{self.session_key}/faces/"
                    
                try:
                    rep_face_path = cluster_reps[cluster_id]["face_path"]
                    rep_face_blob = self.bucket.blob(rep_face_path)
                    destination_blob_name = f"{faces_folder}{cluster_id}.jpg"
                    self.bucket.copy_blob(rep_face_blob, self.bucket, destination_blob_name)
                    
                except Exception as e:
                    logging.error("Error uploading representative face to %s: %s", cluster_folder, e)
                    
                try:  
                    centroid = cluster_reps[cluster_id]["centroid"]
                    blob = self.bucket.blob("{}centroid.npy".format(cluster_folder))
                    blob.upload_from_string(centroid.tobytes())
                    
                except Exception as e:
                    logging.error("Error uploading centroid to %s: %s", cluster_folder, e)
                    
                try:
                    embeddings = cluster_reps[cluster_id]["cluster_embeddings"]
                    orig_paths = [self.orig_image_paths[i] for i in np.where(cluster_labels == cluster_id)[0]]

                    # Calculate the cosine distances of each embedding to the centroid
                    distances = [self.cosine_distance(embed, centroid) for embed in embeddings]

                    # Create a list of tuples (path, distance), then sort by distance
                    path_distance_pairs = sorted(zip(orig_paths, distances), key=lambda x: x[1])

                    # Include distances in the metadata
                    image_info = [{"path": path, "distance": distance} for path, distance in path_distance_pairs]
                    metadata = {"image_paths": image_info}                    
                    blob = self.bucket.blob(f"{cluster_folder}metadata.json")
                    blob.upload_from_string(json.dumps(metadata))
                    
                except Exception as e:
                    logging.error("Error uploading image paths to %s: %s", cluster_folder, e)

                # Count the number of images in the cluster
                num_images = len(np.where(cluster_labels == cluster_id)[0])
                cluster_sizes[cluster_id] = num_images  

            # Sort clusters by size
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True)
            sorted_by_amount = [str(cluster_id) for cluster_id, _ in sorted_clusters]

            # Add sorted list to metadata
            cluster_summary = {
                "clusters": {str(cluster_id): {"amount": size} for cluster_id, size in cluster_sizes.items()},
                "sorts": {"sortedByAmount": sorted_by_amount}
            }

            # Save the cluster summary to metadata.json
            metadata_blob = self.bucket.blob(f"{self.session_key}/faces/metadata.json")
            metadata_blob.upload_from_string(json.dumps(cluster_summary))    
            logging.info("Finished saving clusters")
            
        except Exception as e:
            logging.error("Error saving clusters: %s", e)
            raise

    def execute(self):
        self.load_data()
        cluster_labels = self.cluster()
        # cluster_reps = self.get_cluster_reps(cluster_labels)
        cluster_reps = self.get_cluster_medoids(cluster_labels)
        self.save_clusters(cluster_labels, cluster_reps)