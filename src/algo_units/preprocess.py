import cv2
import asyncio
from deepface import DeepFace
import numpy as np
from google.cloud import storage
from requests import RequestException
from consts_and_utils import BATCH_SIZE, BUCKET_NAME, DETECTORS, RAW_DATA_FOLDER, is_clear
import logging
import os
import hashlib

logging.basicConfig(level=logging.INFO)


class PeepsPreProcessor:
    def __init__(self, session_key: str):
        try:
            self.session_key = session_key
            self.storage_client = storage.Client()
            self.bucket_name = BUCKET_NAME
            self.source_bucket = self.storage_client.get_bucket(self.bucket_name)
            self.image_paths = self.get_image_paths_from_bucket()
            self.preprocess_folder = 'preprocess'
        except Exception as e:
            logging.error(f"Initialization error: {e}")
    
    def download_image_from_gcs(self, image_path: str) -> np.array:
        if not image_path or not isinstance(image_path, str):
            raise ValueError("Invalid image path provided.")  

        try:
            blob = self.source_bucket.blob(image_path)
            img_data = blob.download_as_bytes()
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)
            return img
        except Exception as e:
            logging.error(f"Error downloading image: {e}")
            return np.array([])


    def get_image_paths_from_bucket(self) -> list:
        if not self.session_key or not isinstance(self.session_key, str):
            raise ValueError("Invalid session key.")
        
        try:
            folder_path = f'{self.session_key}/{RAW_DATA_FOLDER}'
            blobs = list(self.storage_client.list_blobs(self.bucket_name, prefix=folder_path))
            return [blob.name for blob in blobs if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except Exception as e:
            logging.error(f"Error fetching images from bucket: {e}")
            return []
        
    async def upload_to_gcs(self, data, destination_path, content_type='image/jpeg'):
        try:
            blob = self.source_bucket.blob(destination_path)
            blob.upload_from_string(data, content_type=content_type)
        except RequestException as re:
            logging.error(f"HTTP request failed during upload: {re}")
        except Exception as e:
            logging.error(f"Error uploading to GCS: {e}")

    @staticmethod
    def crop_faces_from_image(img) -> list:
        if not isinstance(img, np.ndarray) or len(img.shape) != 3:
           raise ValueError("Invalid image provided.")
        
        try:
            detected_faces_img = DeepFace.extract_faces(img, detector_backend=DETECTORS[4], enforce_detection=False, align=True)
            logging.info(f"Number of extracted faces: {len(detected_faces_img)}")
            cropped_faces = []
            for face in detected_faces_img:
                x, y, w, h = face["facial_area"].values()
                face_crop = img[y:y+h, x:x+w]
                if is_clear(image=img, face=face_crop):
                    cropped_faces.append(face_crop)
            return cropped_faces
        except Exception as e:
            logging.error(f"Error cropping faces: {e}")
            return []

    @staticmethod
    def get_face_embedding(face_img) -> np.array:
        if not isinstance(face_img, np.ndarray) or len(face_img.shape) != 3:
            raise ValueError("Invalid face image provided.")
        
        try:
            embedding = DeepFace.represent(face_img, model_name='VGG-Face', enforce_detection=False)
            return np.array(embedding[0]['embedding'])
        except Exception as e:
            logging.error(f"Error getting face embedding: {e}")
            return np.array([])
    
    def preprocess_entire_image(self, img) -> np.array:
        """
        Apply preprocessing steps on the entire image to improve face detection.
        
        Args:
            img: The original image.
        
        Returns:
            np.array: Preprocessed image.
        """
        if not isinstance(img, np.ndarray) or len(img.shape) != 3:
            raise ValueError("Invalid image provided.")
        try:
            # Histogram Equalization for improving contrast
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.equalizeHist(img_gray)
            img = cv2.merge([img_gray, img_gray, img_gray])
            
            # Denoising
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            
            # Resizing (if needed)
            max_dimension = max(img.shape)
            scale_factor = 800 / max_dimension
            if max_dimension > 800:
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            return img
        except Exception as e:
            logging.error(f"Error preprocessing entire image: {e}")
            return img

    def preprocess_face(self, face_img) -> np.array:
        """
        Apply preprocessing steps on the cropped face to prepare for embedding extraction.
        
        Args:
            face_img: The cropped face image.
        
        Returns:
            np.array: Preprocessed face image.
        """
        if not isinstance(face_img, np.ndarray) or len(face_img.shape) != 3:
             raise ValueError("Invalid face image paths.")
        try:
            # Histogram Equalization for improving contrast
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.equalizeHist(face_gray)
            face_img = cv2.merge([face_gray, face_gray, face_gray])
            
            # Denoising
            face_img = cv2.fastNlMeansDenoisingColored(face_img, None, 10, 10, 7, 21)
            
            # Resizing
            standard_size = (224, 224)
            face_img = cv2.resize(face_img, standard_size, interpolation=cv2.INTER_LINEAR)

            return face_img
        except Exception as e:
            logging.error(f"Error preprocessing face: {e}")
            return face_img

    async def preprocess_and_upload_batch(self, batch_image_paths):
        if not batch_image_paths or not isinstance(batch_image_paths, list):
            raise ValueError("Invalid batch image paths provided.")

        for image_path in batch_image_paths:
            try:
                img = self.download_image_from_gcs(image_path)

            except RequestException as re:
                logging.error(f"HTTP request failed while downloading image {image_path}: {re}")
                continue  # Move to next image if this one fails
            except Exception as e:
                logging.error(f"Error downloading image {image_path}: {e}")
                continue

            try:
                img = self.preprocess_entire_image(img)
            except Exception as e:
                logging.error(f"Error preprocessing entire image {image_path}: {e}")
                continue

            try:
                cropped_faces = self.crop_faces_from_image(img)
            except Exception as e:
                logging.error(f"Error cropping faces from image {image_path}: {e}")
                continue

            for idx, face in enumerate(cropped_faces):
                try:
                    processed_face = self.preprocess_face(face)
                except Exception as e:
                    logging.error(f"Error preprocessing face {idx} from image {image_path}: {e}")
                    continue

                try:
                    # Get embedding and generate a unique hash for it
                    embedding = self.get_face_embedding(processed_face)
                    embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
                except Exception as e:
                    logging.error(f"Error getting embedding for face {idx} from image {image_path}: {e}")
                    raise(e)

                try:
                    # Upload cropped face
                    img_encoded = cv2.imencode('.jpg', processed_face)[1]
                    face_data = img_encoded.tobytes()
                    face_destination_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_hash}/face.jpg"
                    await self.upload_to_gcs(face_data, face_destination_path)
                except Exception as e:
                    logging.error(f"Error uploading cropped face {idx} from image {image_path}: {e}")
                    raise(e)

                try:
                    # Upload embedding
                    embedding_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_hash}/embedding.npy"
                    await self.upload_to_gcs(embedding.tobytes(), embedding_path, content_type='application/octet-stream')
                except Exception as e:
                    logging.error(f"Error uploading embedding for face {idx} from image {image_path}: {e}")
                    continue

                try:
                    # Save the original image path
                    original_image_data = image_path.encode()  # Assuming image_path is a string
                    original_image_destination_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_hash}/original_image_path.txt"
                    await self.upload_to_gcs(original_image_data, original_image_destination_path)
                except Exception as e:
                    logging.error(f"Error saving original image path for face {idx} from image {image_path}: {e}")
                    continue


    async def process_all_batches(self):
        if not self.image_paths or not isinstance(self.image_paths, list):
            raise ValueError("Invalid image paths.")

        batch_size = BATCH_SIZE
        for i in range(0, len(self.image_paths), batch_size):
            batch = self.image_paths[i: i+batch_size]
            try:
                await self.preprocess_and_upload_batch(batch)
            except Exception as e:
                logging.error(f"Error processing batch starting at index {i}: {e}")
        logging.info("Processing complete.")
    
    async def execute(self):
        await self.process_all_batches()