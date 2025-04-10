from collections import defaultdict
import io
import logging
import os
import json
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import torch
from facenet_pytorch import MTCNN

from common.consts_and_utils import DROPOUT_THRESHOLD

class ClusterSaver:
    def __init__(self, session_key, bucket, orig_image_paths):
        self.session_key = session_key
        self.bucket = bucket
        self.orig_image_paths = orig_image_paths
        self.faces_folder = f"{self.session_key}/faces/"
        self.undetected_folder = f"{self.faces_folder}undetected_or_low_conf_faces/"
        self.clusters_folder = f"{self.session_key}/clusters/"
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(
                image_size=160, margin=80, min_face_size=85,
                thresholds=[0.6, 0.7, 0.7], factor=0.65, post_process=True,
                device=self.device
            ).eval()
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    #prossesing the clusters functions
    def process_cluster(self, cluster_id, rep_info, cluster_labels, image_to_clusters):
        try:
            cluster_folder = f"{self.clusters_folder}{cluster_id}/"
            rep_face_image = rep_info["face_image"]
            sorted_faces = rep_info["sorted_faces"]
            rep_prob = rep_info['best_face_prob']
            probs = rep_prob

            if probs is not None and probs >= DROPOUT_THRESHOLD:
                # self.upload_representative_face(rep_face_image, cluster_id)
                self.upload_faces_for_cluster(sorted_faces, cluster_id)
                centroid = rep_info["rep_embbeding"]
                
                if centroid is not None:
                    self.upload_centroid(centroid, cluster_folder)
                else:
                    logging.error(f"Invalid centroid for cluster ID {cluster_id}. Skipping...")
                    return None, {}
                

                        # Retrieve the information from the rep_info dictionary
                embeddings, orig_paths, metrics, sorting_scores = (
                    rep_info["cluster_embeddings"],
                    rep_info["orig_paths"],
                    rep_info["metrics"],
                    rep_info["sorting_scores"]
                )
                _, image_info = self.calculate_distances_and_image_info(embeddings, orig_paths, centroid, sorting_scores)
                look_alikes = [str(look_alike) for look_alike in rep_info["look_alikes"]]
                
                metadata = {
                    "images": image_info,
                    "looks_alike": look_alikes
                }
                
                num_images = len(orig_paths)

                for image_info_item in image_info:
                    image_path = image_info_item['file_name']
                    image_to_clusters[image_path].append(str(cluster_id))

                logging.info(f"Processed cluster ID {cluster_id} with {num_images} images.")
                return metadata, {cluster_id: num_images} , image_to_clusters

            else:
                self.handle_low_confidence_or_no_face(rep_face_image, cluster_id, probs)
                return None, {},{}
        except Exception as e:
            logging.error(f"Error processing cluster ID {cluster_id}: {e}")
            self.save_undetected_face(self.undetected_folder, rep_face_image, cluster_id, None)
            return None, {}, {}
    
    def get_face_prob(self, rep_face_image):
        try:
            # Replace with actual face detection logic
            with torch.no_grad():
                _, probs = self.mtcnn(rep_face_image, return_prob=True)
                logging.info(f"Face detection confidence for cluster: {probs}")
                return probs
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            return None

    def upload_representative_face(self, rep_face_image, cluster_id):
        try:
            rep_face_image = cv2.resize(rep_face_image, (244, 244), interpolation=cv2.INTER_LINEAR)
            _, buffer = cv2.imencode('.jpg', rep_face_image)
            face_data = buffer.tobytes()
            destination_blob_name = f"{self.faces_folder}{cluster_id}.jpg"
            rep_face_blob = self.bucket.blob(destination_blob_name)
            rep_face_blob.upload_from_string(face_data, content_type='image/jpeg')
            logging.info(f"Uploaded representative face for cluster ID {cluster_id}")
        except Exception as e:
            logging.error(f"Error uploading representative face for cluster ID {cluster_id}: {e}")

    def upload_faces_for_cluster(self, sorted_faces, cluster_id):
        try:
            for index, face in enumerate(sorted_faces):
                resized_face = cv2.resize(face, (244, 244), interpolation=cv2.INTER_LINEAR)
                _, buffer = cv2.imencode('.jpg', resized_face)
                face_data = buffer.tobytes()
                destination_blob_name = f"{self.faces_folder}{cluster_id}_{index}.jpg"
                face_blob = self.bucket.blob(destination_blob_name)
                face_blob.upload_from_string(face_data, content_type='image/jpeg')
                logging.info(f"Uploaded face {index} for cluster ID {cluster_id}")
        except Exception as e:
            logging.error(f"Error uploading faces for cluster ID {cluster_id}: {e}")

    def upload_centroid(self, centroid, cluster_folder):
        try:
            buffer = io.BytesIO()
            np.save(buffer, centroid)
            buffer.seek(0)
            blob = self.bucket.blob(f"{cluster_folder}centroid.npy")
            blob.upload_from_file(buffer, content_type='application/octet-stream')
            logging.info(f"Uploaded centroid for cluster folder {cluster_folder}")
        except Exception as e:
            logging.error(f"Error uploading centroid to {cluster_folder}: {e}")

    def get_embeddings_and_paths(self, rep_info, cluster_labels, cluster_id):
        embeddings = rep_info["cluster_embeddings"]
        orig_paths = [self.orig_image_paths[i] for i in np.where(cluster_labels == cluster_id)[0]]
        logging.info(f"Number of images in cluster {cluster_id}: {len(orig_paths)}")
        return embeddings, orig_paths

    def calculate_distances_and_image_info(self, embeddings, orig_paths, centroid, sorting_scores):
        try:
            distances = [euclidean(embed, centroid) for embed in embeddings]

            # Pair each original path with its corresponding sorting score and distance
            path_score_distance_pairs = zip(orig_paths, sorting_scores, distances)

            # Sort the pairs by score in descending order (highest first)
            sorted_pairs = sorted(path_score_distance_pairs, key=lambda x: x[1], reverse=True)

            image_info = [{"file_name": os.path.basename(path), "distance": distance, "sort_score": score} for path , score, distance  in sorted_pairs]
            logging.info(f"Calculated distances and image info for cluster with {len(embeddings)} images.")
            return distances, image_info
        except Exception as e:
            logging.error(f"Error calculating distances and image info: {e}")
            return [], [], []
    
    def handle_low_confidence_or_no_face(self, rep_face_image, cluster_id, probs):
        logging.info(f"Cluster {cluster_id} skipped due to low face confidence or no face detected, confidence: {probs}")
        self.save_undetected_face(self.undetected_folder, rep_face_image, cluster_id, probs)

    def save_undetected_face(self, undetected_folder, face_image, cluster_id, probs):
        # Method to save undetected or low confidence faces
        try:
            _, buffer = cv2.imencode('.jpg', face_image)
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
            logging.info(f"Uploaded undetected or low confidence face for cluster {cluster_id}")
        except Exception as e:
            logging.error(f"Error saving undetected or low confidence face for cluster {cluster_id}: {e}")

    def create_related_peeps(self, cluster_metadata, image_to_clusters):
        for cluster_id, data in cluster_metadata.items():
            related_peeps_counter = defaultdict(int)

            # Iterate through each image and its associated cluster IDs
            for image_info in data['images']:
                image_path = image_info['file_name']
                if image_path in image_to_clusters:
                    for related_cluster_id in image_to_clusters[image_path]:
                        if related_cluster_id != cluster_id:
                            related_peeps_counter[related_cluster_id] += 1

            # Sort related clusters by appearance count
            related_peeps_sorted = sorted(related_peeps_counter.items(), key=lambda item: item[1], reverse=True)
            
            # Exclude the current cluster_id and create the final list
            data['related_peeps'] = [str(cid) for cid, _ in related_peeps_sorted if str(cid) != str(cluster_id)]

    #saving the clusters metadata functions
    def upload_json_to_gcs(self, data, blob_name):
        """Helper method to upload JSON to GCS."""
        try:
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(json.dumps(data))
            logging.info(f"Uploaded JSON to {blob_name}.")
        except Exception as e:
            logging.error(f"Error uploading JSON to {blob_name}: {e}")

    def save_metadata_for_env(self, data, base_name, environments):
        """Save metadata JSON for different environments."""
        for env in environments:
            # Construct the appropriate file name based on the environment
            env_blob_name = f"{self.session_key}/{base_name}{'' if env == '' else '_' + env}.json"
            self.upload_json_to_gcs(data, env_blob_name)

    def upload_faces_metadata_json(self, cluster_sizes):
        """Save cluster summary for default, dev, and prod environments."""
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True)
        sorted_by_amount = [str(cluster_id) for cluster_id, _ in sorted_clusters]

        cluster_summary = {
            "clusters": {str(cluster_id): {"amount": size, "rep" : "0"} for cluster_id, size in cluster_sizes.items()},
            "sorts": {"sortedByAmount": sorted_by_amount}
        }

        # Save for default, dev, and prod
        self.save_metadata_for_env(cluster_summary, 'faces/metadata', ['dev', 'prod', ''])

    def upload_clusters_metadata_json(self, cluster_metadata , cluster_centroids):
        """Upload combined metadata JSON for all clusters for default, dev, and prod environments."""
        self.save_metadata_for_env(cluster_metadata, 'clusters/metadata', ['dev', 'prod', ''])
    

    def upload_root_metadata_json(self, image_to_clusters):
        """Upload images metadata JSON for default, dev, and prod environments."""
        images_metadata = {image: [str(cluster_id) for cluster_id in set(clusters)] for image, clusters in image_to_clusters.items()}
        self.save_metadata_for_env(images_metadata, 'metadata', ['dev', 'prod', ''])
                

    def save_clusters(self, cluster_labels, cluster_reps):
        logging.info(f"Saving {len(cluster_reps)} clusters to gcs")
        cluster_centroids = {}
        cluster_metadata = {}
        cluster_sizes = {}
        image_to_clusters = defaultdict(list)
        valid_cluster_ids = [cluster_id for cluster_id in cluster_reps if cluster_id != -1]

        for cluster_id in valid_cluster_ids:
            metadata, sizes, image_clusters = self.process_cluster(cluster_id, cluster_reps[cluster_id], cluster_labels, image_to_clusters)
            if metadata:  # Only add if metadata was successfully created
                cluster_metadata[str(cluster_id)] = metadata
                cluster_sizes.update(sizes)
                image_to_clusters.update(image_clusters)
                # Save centroid to cluster_centroids dictionary
                centroid = cluster_reps[cluster_id]["rep_embbeding"]
                if centroid is not None:
                    cluster_centroids[str(cluster_id)] = centroid.tolist()
                else:
                    logging.error(f"Invalid centroid for cluster ID {cluster_id}. Skipping...")
            else:
                logging.warning(f"Skipping cluster ID {cluster_id}")
                continue

        logging.info("Saving related peeps...")
        self.create_related_peeps(cluster_metadata, image_to_clusters)

        self.upload_clusters_metadata_json(cluster_metadata, cluster_centroids)  # Upload once after adding "related_peeps"
        # Upload the cluster centroids as a single JSON file
        self.upload_json_to_gcs(cluster_centroids, f"{self.clusters_folder}cluster_centroids.json")
        logging.info("related peeps saved successfully.")        
        self.upload_faces_metadata_json(cluster_sizes)
        logging.info("faces summary saved successfully.")
        self.upload_root_metadata_json(image_to_clusters)
        logging.info("root images metadata saved successfully.")

        logging.info("Cluster saving completed successfully.")
