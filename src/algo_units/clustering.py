import io
import json
import logging
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from google.cloud import storage
from io import BytesIO

from sklearn.metrics import pairwise_distances_argmin_min
from consts_and_utils import BUCKET_NAME
from typing import List

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
            logging.info("Loading face embeddings and paths from Cloud Storage")
            
            prefix = f"{self.session_key}/preprocess"

            blobs = self.bucket.list_blobs(prefix=prefix)
            blob_count = sum(1 for _ in self.bucket.list_blobs(prefix=prefix))
            logging.info("Found %s blobs with prefix %s", blob_count, prefix)

            for blob in blobs:
                try:
                    if blob.name.endswith("embedding.npy"):
                        
                        logging.debug("Loading embedding for %s", blob.name)
                        embedding = self.download_embedding_from_gcs(blob.name)
                        
                        face_path = blob.name.replace("embedding.npy", "face.jpg")
                        
                        logging.debug("Getting original image path for %s", blob.name)
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
            orig_txt = blob.name.replace("embedding.npy", "orig.txt")
            
            logging.debug("Downloading original text file %s", orig_txt)
            text = self.bucket.blob(orig_txt).download_as_string()
            
            logging.debug("Decoded text file %s", orig_txt)
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
                logging.debug(f"Uploaded original image path to {orig_blob_name}")
                
            except Exception as e:
                logging.error(f"Error saving noise point {i}: {e}")

    def cluster(self):
        try:
            # Convert list of embeddings to a 2D NumPy array if not already
            array_embbedings = np.array(self.embeddings)
            if array_embbedings.ndim == 1:
                    array_embbedings = np.stack(array_embbedings)
            logging.info("Clustering %s embeddings using DBSCAN", len(array_embbedings))

            # dbscan = DBSCAN(eps=0.575, min_samples=2)
            # Initialize KMeans 
            kmeans = KMeans(n_clusters=25, random_state=0)

            #log the shape of self.embbbeding tto ensure it's 2D
            logging.info("Embeddings shape before clustering: %s", array_embbedings.shape)

            # logging.info("Fitting DBSCAN model")
            # dbscan.fit(array_embbedings)

            logging.info("Fitting KMeans model")
            kmeans.fit(array_embbedings)
    
            logging.info("Extracting cluster labels")
            # cluster_labels = dbscan.labels_
            cluster_labels = kmeans.labels_

            logging.debug("Cluster labels: %s", cluster_labels)

            # determine the total number of unique faces found in the dataset
            labelIDs = np.unique(cluster_labels)
            numUniqueFaces = len(np.where(labelIDs > -1)[0])
            logging.info("# unique faces: {}".format(numUniqueFaces))

            # Calculate distance to nearest cluster center
            _, distances = pairwise_distances_argmin_min(array_embbedings, kmeans.cluster_centers_)

            # Identify noise points
            noise_threshold = self.calculate_noise_threshold(distances)
            noise_points = np.where(distances > noise_threshold)[0]
            logging.info("Identified %s noise points", len(noise_points))

            # Handle noise points
            self.handle_noise_points(noise_points)

            logging.info("Removing noise points from labels")
            
            # Exclude noise points from labels
            valid_points = np.where(distances <= noise_threshold)[0]
            valid_labels = cluster_labels[valid_points]
            # cluster_labels = cluster_labels[np.where(cluster_labels != -1)[0]]

            logging.info("Clustering completed with %s labels", len(valid_labels))
            return valid_labels

            # logging.info("Clustering completed with %s labels", len(cluster_labels))
            # logging.debug("Cluster labels: %s", cluster_labels)

            # return cluster_labels

        except Exception as e:
            logging.error(f"Error clustering embeddings: {e}")
            raise

    def find_closest_embedding_pairs(self):
        logging.info("Finding the 30 closest unique embedding pairs")

        # Convert list of embeddings to a 2D NumPy array if not already
        embeddings_array = np.array(self.embeddings)
        
        # Calculate the pairwise distance matrix
        distance_matrix = np.linalg.norm(embeddings_array[:, None] - embeddings_array, axis=2)
        
        # Ignore the diagonal by setting them to infinity
        np.fill_diagonal(distance_matrix, np.inf)

        # Find the index of the closest embedding for each face
        closest_indices = np.argmin(distance_matrix, axis=1)
        
        # Generate pairs and sort them by distance
        pairs = [(i, closest_indices[i]) for i in range(len(closest_indices))]
        pairs = sorted(pairs, key=lambda x: np.linalg.norm(embeddings_array[x[0]] - embeddings_array[x[1]]))

        # Filter out duplicates (each face should only appear once)
        unique_pairs = []
        used_indices = set()
        for i, j in pairs:
            if i not in used_indices and j not in used_indices:
                unique_pairs.append((i, j))
                used_indices.add(i)
                used_indices.add(j)
            if len(unique_pairs) == 30:
                break
        
        # Process and copy the face images to the test_embeddings folder
        test_embeddings_folder = f"{self.session_key}/clusters/test_embeddings"
        for idx, (i, j) in enumerate(unique_pairs):
            try:
                # Construct the GCS blob name for the face images
                face_i_blob_name = self.face_paths[i]
                face_j_blob_name = self.face_paths[j]

                # Define the new blob names in the test_embeddings folder
                test_face_i_blob_name = f"{test_embeddings_folder}/closest_pair_{idx}_1.jpg"
                test_face_j_blob_name = f"{test_embeddings_folder}/closest_pair_{idx}_2.jpg"
                
                # Copy the face images to the test_embeddings folder
                face_i_blob = self.bucket.blob(face_i_blob_name)
                face_j_blob = self.bucket.blob(face_j_blob_name)

                self.bucket.copy_blob(face_i_blob, self.bucket, test_face_i_blob_name)
                self.bucket.copy_blob(face_j_blob, self.bucket, test_face_j_blob_name)

                logging.debug(f"Copied face image to {test_face_i_blob_name}")
                logging.debug(f"Copied face image to {test_face_j_blob_name}")

            except Exception as e:
                logging.error(f"Error processing closest pair {idx}: {e}")

        logging.info("Finished saving 10 closest unique embedding pairs")

    def get_cluster_reps(self, cluster_labels):

        cluster_reps = {}
        
        try:
            logging.info("Computing representative for each cluster")
            
            for cluster_id in np.unique(cluster_labels):
            
                logging.debug("Getting embeddings for cluster %s", cluster_id)
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_embeddings = np.array(self.embeddings)[cluster_indices]
                
                logging.debug("Computing centroid for cluster %s", cluster_id)
                centroid = np.mean(cluster_embeddings, axis=0)
                
                logging.debug("Finding closest embedding to centroid in cluster %s", cluster_id)
                closest_index = cluster_indices[np.argmin(np.linalg.norm(cluster_embeddings - centroid, axis=1))]
                
                try:
                    logging.debug("Getting paths for representative embedding in cluster %s", cluster_id)
                    
                    rep_face_path = self.face_paths[closest_index]
                    rep_orig_path = self.orig_image_paths[closest_index]
                    
                except Exception as e:
                    logging.error("Error getting paths for cluster %s: %s", cluster_id, e)
                    continue
            
                logging.debug("Adding representative for cluster %s to dictionary", cluster_id)  
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

    def save_clusters(self, cluster_labels, cluster_reps):

        try:
            logging.info("Saving %s clusters to Cloud Storage", len(cluster_reps))
            
            for cluster_id in np.unique(cluster_labels):
            
                cluster_folder = f"{self.session_key}/clusters/{cluster_id}/"
                    
                try:
                    logging.debug("Uploading representative face to %s", cluster_folder)
                    
                    rep_face_path = cluster_reps[cluster_id]["face_path"]
                    rep_face_blob = self.bucket.blob(rep_face_path)
                    destination_blob_name = f"{cluster_folder}rep_face.jpg"
                    # Copy the representative face image within GCS
                    self.bucket.copy_blob(rep_face_blob, self.bucket, destination_blob_name)

                    logging.debug("Copied face image to %s", destination_blob_name)
                    
                except Exception as e:
                    logging.error("Error uploading representative face to %s: %s", cluster_folder, e)
                    
                try:  
                    logging.debug("Uploading centroid embedding to %s", cluster_folder)
                    
                    centroid = cluster_reps[cluster_id]["centroid"]
                    blob = self.bucket.blob("{}centroid.npy".format(cluster_folder))
                    blob.upload_from_string(centroid.tobytes())
                    
                except Exception as e:
                    logging.error("Error uploading centroid to %s: %s", cluster_folder, e)
                    
                try:
                    logging.debug("Uploading ordered image paths to %s", cluster_folder)
                    
                    # Retrieve the embeddings and original paths for the current cluster
                    embeddings = cluster_reps[cluster_id]["cluster_embeddings"]
                    orig_paths = [self.orig_image_paths[i] for i in np.where(cluster_labels == cluster_id)[0]]

                    # Calculate the distances of each embedding to the centroid
                    distances = np.linalg.norm(embeddings - centroid, axis=1)

                    # Create a list of tuples (path, distance), then sort by distance
                    path_distance_pairs = sorted(zip(orig_paths, distances), key=lambda x: x[1])

                    # Save the sorted paths (along with distances if you want) to GCS
                    blob = self.bucket.blob(f"{cluster_folder}orig_paths.json")
                    blob.upload_from_string(json.dumps(path_distance_pairs))
                    
                except Exception as e:
                    logging.error("Error uploading image paths to %s: %s", cluster_folder, e)
                
            logging.info("Finished saving clusters")
            
        except Exception as e:
            logging.error("Error saving clusters: %s", e)
            raise

    def execute(self):
        self.load_data()
        cluster_labels = self.cluster()
        cluster_reps = self.get_cluster_reps(cluster_labels)
        self.save_clusters(cluster_labels, cluster_reps)

        # validation
        self.find_closest_embedding_pairs()