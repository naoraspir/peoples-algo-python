import io
import json
import logging
import os
import traceback
import cv2
import numpy as np
from google.cloud import storage
import torch

from sklearn.preprocessing import normalize
from facenet_pytorch import MTCNN
from algo_units.best_face_utils import calculate_sharpness, detect_glasses, evaluate_face_alignment, normalize_scores
from algo_units.face_uniter import FaceUniter
from consts_and_utils import ALIGNMENT_WEIGHT, BUCKET_NAME, DISTANCE_METRIC_HDBSCAN, DISTANCE_WEIGHT, DROPOUT_THRESHOLD, GLASSES_DEDUCTION_WEIGHT, GRAY_SCALE_DEDUCTION_WEIGHT, MIN_CLUSTER_SIZE_HDBSCAN, N_DIST_JOBS_HDBSCAN, N_NEIGHBORS_FACE_UNITER, SHARPNNES_WEIGHT, is_grayscale
from typing import List, Optional, Tuple
import hdbscan
from scipy.spatial.distance import euclidean
import face_recognition

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('numba.core').setLevel(logging.INFO)
# change level of debug for urllib3.connectionpool to INFO
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

class FaceClustering:

    def __init__(self, session_key: str, all_results: List[dict]):
        self.session_key = session_key
        self.all_results = all_results
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
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.embeddings, self.faces, self.orig_image_paths = self.extract_data(all_results)
        self.mtcnn = MTCNN(
                image_size=160, margin=80, min_face_size=85,
                thresholds=[0.6, 0.7, 0.7], factor=0.65, post_process=True,
                device=self.device
            ).eval()

    def extract_data(self, results):
        embeddings = [result['embedding'] for result in results]
        faces = [result['face'] for result in results]
        orig_image_paths = [result['original_path'] for result in results]
        return embeddings, faces, orig_image_paths

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
                        
                        embedding = self.download_embedding_from_gcs(blob.name)
                        
                        face_path = blob.name.replace("embedding.npy", "face.jpg")
                        
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
            
            text = self.bucket.blob(orig_txt).download_as_string()
            
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

                # Now, upload the original image path as a text file to the noise folder
                orig_path = self.orig_image_paths[i]  # The original image path
                orig_blob_content = orig_path.encode('utf-8')  # Encoding the path to bytes
                orig_blob_name = f"{noise_folder}/{i}.txt"  # The blob name for the original path text file
                
                # Create a new blob for the original image path and upload the content
                orig_blob = self.bucket.blob(orig_blob_name)
                orig_blob.upload_from_string(orig_blob_content, content_type='text/plain')
                
            except Exception as e:
                logging.error(f"Error saving noise point {i}: {e}")

    def cluster(self):
        try:
            array_embeddings = np.array(self.embeddings).astype(np.float64)  # Convert to float64 for clustering
            # Apply L2 normalization
            normalized_embeddings = normalize(array_embeddings, norm='l2')

            # normalized_embeddings = normalize(array_embeddings)

            # Using UMAP for dimensionality reduction
            # Skip UMAP if the dataset is small
            # if array_embeddings.shape[0] < 50:
            #     logging.info("Skipping UMAP due to small dataset size")
            #     processed_embeddings = array_embeddings
            # else:
            #     normalized_embeddings = normalize(array_embeddings)
            #     umap_model = umap.UMAP(metric='euclidean' , n_components=min(50, array_embeddings.shape[0] - 1),n_neighbors=30 , min_dist=0.0, random_state=42)
            #     processed_embeddings = umap_model.fit_transform(normalized_embeddings)
            #     logging.info(f"UMAP embeddings array shape: {processed_embeddings.shape}")

            # Compute cosine distance matrix
            # distance_matrix = cosine_distances(normalized_embeddings)

            # Clustering with HDBSCAN
            # clusterer = hdbscan.HDBSCAN(min_cluster_size=5, cluster_selection_method='eom', core_dist_n_jobs=-1)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE_HDBSCAN, metric=DISTANCE_METRIC_HDBSCAN, core_dist_n_jobs=N_DIST_JOBS_HDBSCAN)
            cluster_labels = clusterer.fit_predict(normalized_embeddings)

            #log noise point amount
            logging.info("noise point amount: "+str(len(np.where(cluster_labels == -1)[0])))

            # Filter out noise (-1) cluster
            valid_cluster_labels = cluster_labels[cluster_labels != -1]
            unique_clusters = set(valid_cluster_labels)
            numUniqueFaces = len(unique_clusters)
            logging.info(f"# unique faces (excluding noise): {numUniqueFaces}")


            return cluster_labels
        except Exception as e:
            logging.error(f"Error clustering embeddings: {e}")
            raise

    def choose_best_face(self, faces: List[np.ndarray], embeddings: List[np.ndarray], centroid: np.ndarray, cluster_id:int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        try:
            best_score = -1.0
            best_face: Optional[np.ndarray] = None
            best_embedding: Optional[np.ndarray] = None
            best_face_landmarks: dict = {}
            faces_with_glasses = []  # To store faces with glasses
            best_face_sharpness = -1.0
            best_face_alignment = -1.0
            best_face_distance = -1.0
            best_face_glasses = -1.0
            best_face_gray_scale = -1.0

            # Check if centroid is None or contains nan
            if centroid is None or np.isnan(centroid).any():
                logging.warning(f"Centroid is None or contains nan: {centroid}")
            
            sharpness_scores = np.array([calculate_sharpness(face) for face in faces])
            sharpness_scores = normalize_scores(sharpness_scores)

            # Calculate the distance of each embedding to the centroid
            distance_scores_numpy = np.array([euclidean(embedding, centroid) for embedding in embeddings])
            distance_scores = normalize_scores(distance_scores_numpy , is_distance_score=True)
            
            alignment_scores = np.array([evaluate_face_alignment(face) for face in faces])
            alignment_scores = normalize_scores(alignment_scores)
            
            for idx, (face, embedding) in enumerate(zip(faces, embeddings)):
                # Check if the face has glasses
                # Glass detection and score adjustment
                landmark_list = face_recognition.face_landmarks(face, model="large")
                if not landmark_list:
                    glasses_deduction = 0.0                    
                elif detect_glasses(face, landmark_list[0]):
                    glasses_deduction = 1.0
                    faces_with_glasses.append(face)  # Save face for debugging
                else:
                    glasses_deduction = 0.0

                # Check for grayscale and deduct from the score
                is_gray = is_grayscale(face)
                grayscale_deduction = 1.0 if is_gray else 0.0

                # Calculate the composite score
                score = (SHARPNNES_WEIGHT * sharpness_scores[idx] + 
                        ALIGNMENT_WEIGHT * alignment_scores[idx] + 
                        DISTANCE_WEIGHT * (1 - distance_scores[idx]) +
                        GLASSES_DEDUCTION_WEIGHT * glasses_deduction +
                        GRAY_SCALE_DEDUCTION_WEIGHT * grayscale_deduction)  # glasses and gray scale images are scored much lower to give perfect face candidates a chance

                # Log scores for debugging
                # logging.debug(f"Face index {idx}: Sharpness={sharpness_scores[idx]}, Alignment={alignment_scores[idx]}, Distance={distance_scores[idx]},\n\
                #                Glasses Deduction= {glasses_deduction}, Gray Scale Deduction = {grayscale_deduction} , Score={score}")

                if score > best_score:
                    best_score = score
                    best_face = face
                    best_embedding = embedding
                    best_face_landmarks = landmark_list[0] if landmark_list else {}
                    best_face_sharpness = sharpness_scores[idx]  * SHARPNNES_WEIGHT
                    best_face_alignment = alignment_scores[idx] * ALIGNMENT_WEIGHT
                    best_face_distance = distance_scores[idx] * DISTANCE_WEIGHT
                    best_face_glasses = glasses_deduction * GLASSES_DEDUCTION_WEIGHT
                    best_face_gray_scale = grayscale_deduction * GRAY_SCALE_DEDUCTION_WEIGHT
                   

            # Fallback: if no best face found, choose the medoid face
            if best_face is None:
                logging.warning("No best face found based on scores, choosing medoid face instead.")
                medoid_index = np.argmin(distance_scores)
                best_face = faces[medoid_index]
                best_embedding = embeddings[medoid_index]
            else:
                logging. info(f"Best face found for cluster {cluster_id}: Sharpness={best_face_sharpness}, Alignment={best_face_alignment}, Distance={best_face_distance},\n")
                logging. info(f"Glasses Deduction= {best_face_glasses}, Gray Scale Deduction = {best_face_gray_scale} , Score={best_score}")
                #save the best face with landmarks to gcs
                # self.save_best_face_landmarks_to_gcs(best_face, best_face_landmarks, cluster_id)    
            return best_face, best_embedding, faces_with_glasses  # Return faces with glasses for debugging
        except Exception as e:
            logging.error("Error choosing best face: %s", e)
            raise e

    def save_faces_with_glasses(self, faces_with_glasses, cluster_id):
        try:
            glasses_dir = f"{self.session_key}/faces/faces_with_glasses/{cluster_id}/"
            for idx, face in enumerate(faces_with_glasses):
                # Resize and encode the face image
                # resized_face = cv2.resize(face, (244, 244), interpolation=cv2.INTER_LINEAR)
                _, buffer = cv2.imencode('.jpg', face)
                face_data = buffer.tobytes()

                # Define the destination path in GCS
                destination_blob_name = f"{glasses_dir}{idx}.jpg"

                # Create a blob and upload the face image
                face_blob = self.bucket.blob(destination_blob_name)
                face_blob.upload_from_string(face_data, content_type='image/jpeg')

        except Exception as e:
            logging.error(f"Error uploading face with glasses in cluster {cluster_id}, image {idx}: {e}")
    
    def save_best_face_landmarks_to_gcs(self, face: np.ndarray, landmarks, cluster_id: int):
        try:
            face_with_drawn_landmarks = face.copy()
            # Draw landmarks on the face image
            for feature in landmarks:
                for point in landmarks[feature]:
                    cv2.circle(face_with_drawn_landmarks, tuple(point), 2, (255, 0, 0), -1)
            
            # Encode the image with landmarks to bytes
            _, buffer = cv2.imencode('.jpg', face_with_drawn_landmarks)
            face_data = buffer.tobytes()
            
            # Define the destination path in GCS
            destination_blob_name = f"{self.session_key}/faces/with_landmarks/{cluster_id}.jpg"
            
            # Create a blob and upload the face image with landmarks
            face_blob = self.bucket.blob(destination_blob_name)
            face_blob.upload_from_string(face_data, content_type='image/jpeg')
        except Exception as e:
            logging.error(f"Error uploading face with landmarks in cluster {cluster_id}: {e}")
            raise e

    def get_cluster_reps(self, cluster_labels):
        cluster_reps = {}
        try:
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:
                    continue
                logging.info(f"Processing rep for cluster {cluster_id}")    
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                # Check if the cluster has any faces
                if len(cluster_indices) == 0:
                    logging.warning(f"No faces found for cluster {cluster_id}")
                    continue

                cluster_faces = [self.faces[i] for i in cluster_indices]
                cluster_embeddings = np.array(self.embeddings)[cluster_indices]

                # Calculate the centroid for the current cluster
                centroid = np.mean(cluster_embeddings, axis=0)

                # Choose the best face
                best_face, best_embedding, face_with_glaasses = self.choose_best_face(cluster_faces, cluster_embeddings, centroid, cluster_id)

                # save faces with glasses for debugging
                if len(face_with_glaasses) > 0:
                    # self.save_faces_with_glasses(face_with_glaasses, cluster_id)
                    pass

                if best_face is not None:
                    # Find the index of the best embedding
                    best_face_index = None
                    for i, embedding in enumerate(cluster_embeddings):
                        if np.array_equal(embedding, best_embedding):
                            best_face_index = cluster_indices[i]
                            break

                    if best_face_index is not None:
                        # Store necessary information including the best face image
                        cluster_reps[cluster_id] = {
                            "face_image": best_face,  # Storing the best face image
                            "orig_path": self.orig_image_paths[best_face_index],
                            "rep_embbeding": best_embedding,
                            "cluster_embeddings": cluster_embeddings.tolist(),
                        }
                    else:
                        logging.warning(f"Best embedding not found for cluster {cluster_id}")
                else:
                    logging.warning(f"No best face found for cluster {cluster_id}")

        except Exception as e:
            logging.error("Error computing cluster reps: %s", e)
            raise

        return cluster_reps

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

    def save_undetected_face(self, undetected_folder, rep_face_image, cluster_id, probs):
        # Method to save undetected or low confidence faces
        try:
            _, buffer = cv2.imencode('.jpg', rep_face_image)
            face_data = buffer.tobytes()

            # Determine the file name based on face detection confidence
            if probs is None:
                prob_str = "not_detected"
            elif isinstance(probs, float):
                prob_str = f"{probs:.2f}"
            else:
                prob_str = "weird_probs"    
            destination_blob_name = f"{undetected_folder}{cluster_id}_{prob_str}.jpg"
            undetected_blob = self.bucket.blob(destination_blob_name)
            undetected_blob.upload_from_string(face_data, content_type='image/jpeg')
        except Exception as e:
            logging.error(f"Error saving undetected or low confidence face for cluster {cluster_id}: {e}")

    def save_clusters(self, cluster_labels, cluster_reps):
        try:
            logging.info("Saving %s clusters to Cloud Storage", len(cluster_reps))
            
            cluster_sizes = {}
            #save noise cluster faces
            # self.save_noise_cluster_faces(cluster_labels)
            faces_folder = f"{self.session_key}/faces/"
            undetected_folder = f"{faces_folder}undetected_or_low_conf_faces/"

            # Filter out the noise cluster
            valid_cluster_ids = [cluster_id for cluster_id in cluster_reps if cluster_id != -1]

            logging.info("Available cluster IDs in cluster_reps: %s", list(cluster_reps.keys()))
            
            for cluster_id in np.unique(valid_cluster_ids):
                if cluster_id in cluster_reps:
                    logging.info("Current cluster ID for saving: %s", cluster_id)

                    cluster_folder = f"{self.session_key}/clusters/{cluster_id}/"
                    
                    # Get the representative face image for the cluster
                    rep_face_image = cluster_reps[cluster_id]["face_image"]

                    # Perform face detection on the representative image
                    try:
                        # boxes, probs = self.mtcnn.detect(rep_face_image)
                        _, probs = self.mtcnn(rep_face_image, return_prob=True)
                        #log the conffidence
                        logging.info("probs: "+str(probs))
                    except Exception as e:
                        logging.error("Error detecting faces in cluster %s: %s", cluster_id, e)
                        self.save_undetected_face(undetected_folder, rep_face_image, cluster_id, None)
                        continue
                    if probs is not None:
                        if probs >= DROPOUT_THRESHOLD:    
                            try:

                                # Encode the image to bytes
                                #resize face_crop to 244x244
                                rep_face_image = cv2.resize(rep_face_image, (244,244), interpolation=cv2.INTER_LINEAR)
                                _, buffer = cv2.imencode('.jpg', rep_face_image)
                                face_data = buffer.tobytes()

                                destination_blob_name = f"{faces_folder}{cluster_id}.jpg"

                                # Create a blob and upload the face image
                                rep_face_blob = self.bucket.blob(destination_blob_name)
                                rep_face_blob.upload_from_string(face_data, content_type='image/jpeg')
                                
                            except Exception as e:
                                logging.error("Error uploading representative face to %s: %s", cluster_folder, e)
                                
                            try:  
                                centroid = cluster_reps[cluster_id]["rep_embbeding"]
                                # Create a buffer
                                buffer = io.BytesIO()

                                # Save the array to the buffer
                                np.save(buffer, centroid)

                                # Upload the buffer content to GCS
                                blob = self.bucket.blob("{}centroid.npy".format(cluster_folder))
                                buffer.seek(0)  # Make sure to seek to the start of the buffer
                                blob.upload_from_file(buffer, content_type='application/octet-stream')
                                
                            except Exception as e:
                                logging.error("Error uploading centroid to %s: %s", cluster_folder, e)
                                
                            try:
                                embeddings = cluster_reps[cluster_id]["cluster_embeddings"]
                                orig_paths = [self.orig_image_paths[i] for i in np.where(cluster_labels == cluster_id)[0]]

                                # Calculate the Euclidean distances of each embedding to the centroid
                                distances = [euclidean(embed, centroid) for embed in embeddings]


                                # Create a list of tuples (path, distance), then sort by distance
                                path_distance_pairs = sorted(zip(orig_paths, distances), key=lambda x: x[1])
                                #include look alikes in the json
                                look_alikes = cluster_reps[cluster_id]["look_alikes"]
                                #ensure the each look alike id is pythonic int
                                look_alikes = [int(look_alike) for look_alike in look_alikes]
                                # Include distances in the metadata
                                image_info = [{"file_name": os.path.basename(path), "distance": distance} for path, distance in path_distance_pairs]
                                metadata = {
                                    "images": image_info,
                                    "looks_alike": look_alikes
                                }                  
                                # json_metadata = json.dumps(metadata)
                                blob = self.bucket.blob(f"{cluster_folder}metadata.json")
                                blob.upload_from_string(json.dumps(metadata))
                                
                            except Exception as e:
                                logging.error("Error uploading image paths to %s: %s", cluster_folder, e)

                            # Count the number of images in the cluster
                            num_images = len(np.where(cluster_labels == cluster_id)[0])
                            cluster_sizes[cluster_id] = num_images
                        else:
                            logging.info(f"Cluster {cluster_id} skipped due to low face confidence or no face detected.")
                            # Save undetected or low confidence faces
                            self.save_undetected_face(undetected_folder, rep_face_image, cluster_id, probs)
                    else:
                        logging.info(f"Cluster {cluster_id} skipped due to strange in detection.")
                        # Save undetected or low confidence faces
                        self.save_undetected_face(undetected_folder, rep_face_image, cluster_id, probs)

                else:
                    logging.warning("Cluster ID %s not found in cluster_reps", cluster_id)
                    continue  # Skip this cluster ID  

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
            logging.error("Exception in save_clusters: %s", e)
            traceback_str = traceback.format_exc()
            logging.error("Exception traceback: %s", traceback_str)
            raise

    def execute(self):
        self.load_data()
        cluster_labels = self.cluster()
        # cluster_reps = self.get_cluster_reps(cluster_labels)
        cluster_reps = self.get_cluster_reps(cluster_labels)
        
        
        # Instantiate and run FaceUniter
        face_uniter = FaceUniter(cluster_reps, n_neighbors=N_NEIGHBORS_FACE_UNITER )
        updated_cluster_reps = face_uniter.run()

        # Now use the updated_cluster_reps for further processing
        self.save_clusters(cluster_labels, updated_cluster_reps)