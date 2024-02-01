import io
import json
import logging
import cv2
import numpy as np
from google.cloud import storage

from sklearn.preprocessing import normalize
from algo_units import cluster_saver
from algo_units.best_face_utils import calculate_sharpness, detect_glasses, evaluate_face_alignment, normalize_scores
from algo_units.face_uniter import FaceUniter
from common.consts_and_utils import ALIGNMENT_WEIGHT, BUCKET_NAME, DISTANCE_METRIC_HDBSCAN, DISTANCE_WEIGHT, GLASSES_DEDUCTION_WEIGHT, GRAY_SCALE_DEDUCTION_WEIGHT, MIN_CLUSTER_SIZE_HDBSCAN, N_DIST_JOBS_HDBSCAN, N_NEIGHBORS_FACE_UNITER, PREPROCESS_FOLDER, SHARPNNES_WEIGHT, is_grayscale
from typing import List, Optional, Tuple
import hdbscan
from scipy.spatial.distance import euclidean
import face_recognition

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('numba.core').setLevel(logging.INFO)
# change level of debug for urllib3.connectionpool to INFO
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
logging.getLogger('urllib3.urllib3.util').setLevel(logging.INFO)

logging.getLogger('google.auth').setLevel(logging.INFO)

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
        self.preprocess_folder = f"{self.session_key}/{PREPROCESS_FOLDER}"
        
        # Load data from GCS
        self.embeddings, self.faces, self.orig_image_paths = self.load_data_from_gcs()

        self.cluster_saver = cluster_saver.ClusterSaver(self.session_key, self.bucket, self.orig_image_paths)

    def load_data_from_gcs(self):
        embeddings = self.download_embeddings()
        faces = self.download_faces()
        orig_image_paths = self.download_orig_paths()
        return embeddings, faces, orig_image_paths

    def download_embeddings(self):
        embeddings_blob = self.bucket.blob(f"{self.session_key}/preprocess/embeddings.npy")
        embeddings_bytes = embeddings_blob.download_as_bytes()
        embeddings = np.load(io.BytesIO(embeddings_bytes), allow_pickle=True)
        return embeddings

    def download_faces(self):
        faces_blob = self.bucket.blob(f"{self.session_key}/preprocess/faces.npy")
        faces_bytes = faces_blob.download_as_bytes()
        encoded_faces = np.load(io.BytesIO(faces_bytes), allow_pickle=True)
        # Decode each face image
        faces = [cv2.imdecode(np.frombuffer(face, dtype=np.uint8), cv2.IMREAD_COLOR) for face in encoded_faces]
        return faces

    def download_orig_paths(self):
        paths_blob = self.bucket.blob(f"{self.session_key}/preprocess/original_paths.json")
        paths_json = paths_blob.download_as_text()
        orig_paths = json.loads(paths_json)
        return orig_paths

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
                self.save_best_face_landmarks_to_gcs(best_face, best_face_landmarks, cluster_id)    
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
    
    def execute(self):
        # self.load_data()
        cluster_labels = self.cluster()
        # cluster_reps = self.get_cluster_reps(cluster_labels)
        cluster_reps = self.get_cluster_reps(cluster_labels)
        
        
        # Instantiate and run FaceUniter
        face_uniter = FaceUniter(cluster_reps, n_neighbors=N_NEIGHBORS_FACE_UNITER )
        updated_cluster_reps = face_uniter.run()

        # Now use the updated_cluster_reps for further processing
        self.cluster_saver.save_clusters(cluster_labels, updated_cluster_reps)
