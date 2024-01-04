import io
from multiprocessing import Pool
from typing import List, Tuple
import cv2
import asyncio
import numpy as np
from google.cloud import storage
from requests import RequestException
import torch
from consts_and_utils import BATCH_SIZE, BUCKET_NAME, CONF_THRESHOLD, MAX_WEB_IMAGE_HEIGHT, MAX_WEB_IMAGE_WIDTH, PREPROCESS_FOLDER, RAW_DATA_FOLDER, SEMAPHORE_ALLOWED, WEB_DATA_FOLDER, is_clear
import logging
import os
# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import InceptionResnetV1, MTCNN
import psutil
from gcs_utils import download_image_from_gcs, get_image_paths_from_bucket, upload_to_gcs

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('numba.core').setLevel(logging.INFO)
# change level of debug for urllib3.connectionpool to INFO
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

class PeepsPreProcessor:

    def __init__(self, session_key: str)-> None:
        try:
            self.session_key = session_key
            self.storage_client = storage.Client()
            self.bucket_name = BUCKET_NAME
            self.source_bucket = self.storage_client.get_bucket(self.bucket_name)
            self.preprocess_folder = PREPROCESS_FOLDER
            self.raw_folder = RAW_DATA_FOLDER
            self.cropped_faces_count = 0  # Initialize a counter for cropped faces
            self.failed_to_detect_index = 0  # Initialize a counter for failed faces
            self.failed_to_embbed_index = 0  # Initialize a counter for failed faces  
            self.low_conf_face_index = 0  # Initialize a counter for low confidence faces
            self.not_clear_face_index = 0  # Initialize a counter for not clear faces
            self.no_faces_images_index = 0  # Initialize a counter for no faces detected in image
            self.image_paths = get_image_paths_from_bucket(self.session_key, self.storage_client, self.bucket_name, self.raw_folder)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device}")
            # Load facial recognition model
            self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
            self.embeddings_lambda = lambda input: self.resnet(input)
            self.mtcnn = MTCNN(
                image_size=160, margin=80, min_face_size=85,
                thresholds=[0.6, 0.7, 0.7], factor=0.65, post_process=True,
                device=self.device
            ).eval()
            self.deferred_tasks = []  # Initialize the deferred tasks list
            self.results = []  # New attribute to store results

        except Exception as e:
            logging.error(f"Initialization error: {e}")
    
    def queue_deferred_upload(self, data, destination_path, content_type='image/jpeg')-> None:
        # Queue the task for later
        self.deferred_tasks.append((self.source_bucket, data, destination_path, content_type))

    def _save_failed_image(self, face_crop, failure_type)-> None:
        try:
            _, buffer = cv2.imencode('.jpg', face_crop)
            face_data = buffer.tobytes()
            unique_identifier_path = f"{failure_type}/face_{getattr(self, f'{failure_type}_index')}.jpg"
            destination_path = f"{self.session_key}/{unique_identifier_path}"
            
            # Add the upload task to deferred tasks
            self.queue_deferred_upload(face_data, destination_path, 'image/jpeg')
        except Exception as e:
            logging.error(f"Error saving {failure_type} image: {e}")

    def resize_and_upload_for_web(self, img, original_path):
        
        # Calculate the scaling factor
        height_factor = MAX_WEB_IMAGE_HEIGHT / img.shape[0]
        width_factor = MAX_WEB_IMAGE_WIDTH / img.shape[1]
        scale_factor = min(height_factor, width_factor)

        # Resize the image preserving the aspect ratio
        img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Convert the image to bytes
        _, img_encoded = cv2.imencode('.jpg', img_resized)
        web_image_data = img_encoded.tobytes()

        # Create the new path for the web-optimized image
        # Change the directory to 'web' instead of 'raw' and keep the same file name
        path_parts = original_path.split('/')
        path_parts[-2] = WEB_DATA_FOLDER  # Change the second last element from 'raw' to 'web'
        web_image_path = '/'.join(path_parts)

        # Upload the resized image
        try:
            self.queue_deferred_upload(web_image_data, web_image_path, 'image/jpeg')
        except Exception as e:
            logging.error(f"Error uploading web-optimized image: {e}")

    def crop_faces_from_image_and_embed(self, img) -> Tuple[List[np.array], List[np.array]]:
        if not isinstance(img, np.ndarray) or len(img.shape) != 3:
            raise ValueError("Invalid image provided.")
        
        cropped_faces, face_encodings = [], []
        # logging.info("start of processing new image")
        
        # Process the image
        image_for_extraction = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            boxes, probs = self.mtcnn.detect(image_for_extraction)
        except Exception as ex:
            logging.error(f"Error in faces detection: {ex}")
            raise (ex)

        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                x, y, w, h = box.tolist()
                x, y, w, h = map(int, [x, y, w, h])  # Convert to int

                # Determine padding size
                padding = int(0.4 * (w - x))  # Example: 40% of the face width

                # Apply padding and ensure coordinates are within image bounds
                x_padded = max(x - padding, 0)
                y_padded = max(y - padding, 0)
                w_padded = min(w + padding, img.shape[1])
                h_padded = min(h + padding, img.shape[0])
                if h_padded <= img.shape[0] and w_padded <= img.shape[1] and w_padded > x_padded and h_padded > y_padded:
                    try:
                        face_crop = image_for_extraction[y_padded:h_padded, x_padded:w_padded]
                    
                    except Exception as ex:
                        logging.error(f"Error in face extraction: {ex}")
                        raise(ex)

                    if prob >= CONF_THRESHOLD:
                        if is_clear(image=img, face=face_crop): # Check if the face is clear    
                            try:
                                with torch.no_grad():
                            
                                    face_for_embedding, prob = self.mtcnn(face_crop, return_prob=True)
                                    # #resize face_crop to 244x244
                                    # face_crop_resized = cv2.resize(face_crop, (244,244), interpolation=cv2.INTER_LINEAR)
                                    if face_for_embedding is not None:
                                        face_for_embedding = face_for_embedding.unsqueeze(0).to(self.device)
                                        e = self.embeddings_lambda(face_for_embedding).squeeze().cpu().numpy()
                                        face_encodings.append(e)
                                        cropped_faces.append(face_crop)
                                        self.cropped_faces_count += 1
                                    else:                                        
                                        # self._save_failed_image(face_crop, 'failed_to_detect')
                                        self.failed_to_detect_index += 1
                            except Exception as e:
                                logging.error(f"Error getting face encoding: {e}")
                                # self._save_failed_image(face_crop, 'failed_to_embbed')
                                self.failed_to_embbed_index += 1
                        else:
                            # self._save_failed_image(face_crop, 'not_clear_face') 
                            self.not_clear_face_index += 1  
                    else:
                        # self._save_failed_image(face_crop, 'low_conf_face')
                        self.low_conf_face_index += 1
                else:
                    logging.error("Face coordinates are out of bounds")
        else:
            # self._save_failed_image(image_for_extraction, 'no_faces_images')
            self.no_faces_images_index += 1

        logging.info(f"Number of extracted faces: {len(cropped_faces)}")
        return cropped_faces, face_encodings   

    async def process_single_image(self, image_path)-> None:
        """
        Process a single image: download the image, resize for web, crop faces,
        extract embeddings, and queue everything for upload.
        
        Parameters:
        image_path (str): The path of the image to process.
        """
        try:
            img = await download_image_from_gcs(self.source_bucket, image_path)
            self.resize_and_upload_for_web(img, image_path)
            await self.process_faces_and_embeddings(img, image_path)
        except RequestException as re:
            logging.error(f"HTTP request failed while downloading image {image_path}: {re}")
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")

    async def process_faces_and_embeddings(self, img, image_path)-> None:
        """
        Crop faces from an image, extract embeddings, and queue for upload.
        
        Parameters:
        img (np.array): The image from which faces will be cropped.
        image_path (str): The path of the image being processed.
        """
        cropped_faces, face_encodings = self.crop_faces_from_image_and_embed(img)
        for face_crop_resized, encoding in zip(cropped_faces, face_encodings):
            self.results.append({
            "embedding": encoding,
            "face": face_crop_resized,
            "original_path": image_path
            })

    # Helper methods for each step of the process
    def get_embedding_folder_name(self, image_path, idx)-> str:
        # Extracts the folder name for storing embeddings
        return image_path.split('/')[-1].split('.')[0] + f"_face_{idx}"

    def upload_cropped_face(self, face, embedding_folder_name)-> None:
        # Uploads the cropped face to the storage bucket
        try:
            img_encoded = cv2.imencode('.jpg', face)[1]
            face_data = img_encoded.tobytes()
            face_destination_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_folder_name}/face.jpg"
            self.queue_deferred_upload(face_data, face_destination_path, content_type='image/jpeg')
        except Exception as e:
            logging.error(f"Error uploading cropped face for {embedding_folder_name}: {e}")

    def upload_face_embedding(self, encoding, embedding_folder_name)-> None:
        # Uploads the face embedding to the storage bucket
        try:
            embedding_bytes = io.BytesIO()
            np.save(embedding_bytes, encoding, allow_pickle=True)
            embedding_bytes.seek(0)
            embedding_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_folder_name}/embedding.npy"
            self.queue_deferred_upload(embedding_bytes.read(), embedding_path, content_type='application/octet-stream')
        except Exception as e:
            logging.error(f"Error uploading embedding for {embedding_folder_name}: {e}")

    def save_original_image_path(self, image_path, embedding_folder_name)-> None:
        # Saves the path of the original image
        try:
            original_image_data = image_path.encode()
            original_image_name = "original_path.txt"
            original_image_destination_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_folder_name}/{original_image_name}"
            self.queue_deferred_upload(original_image_data, original_image_destination_path, content_type='text/plain')
        except Exception as e:
            logging.error(f"Error saving original image path for {embedding_folder_name}: {e}")

    async def preprocess_and_upload_batch(self, batch_image_paths)-> None:
        """
        Process a batch of image paths: download the images, preprocess them,
        and queue them for upload to GCS.
        
        Parameters:
        batch_image_paths (List[str]): A list of image paths to process.
        """
        for image_path in batch_image_paths:
            await self.process_single_image(image_path)

    async def preprocess_and_upload_batch_async(self, batch_image_paths, semaphore_value)-> None:
        # semaphore = asyncio.Semaphore(semaphore_value)
        # async with semaphore:
        await self.preprocess_and_upload_batch(batch_image_paths)

        # Ensure that deferred tasks are iterables before gathering
        if self.deferred_tasks:
            logging.info("Uploading deferred tasks...")
            await asyncio.gather(*(upload_to_gcs(*task) for task in self.deferred_tasks if task is not None))
            # pass
            self.deferred_tasks.clear()
        else:
            logging.info("No deferred tasks to upload.")

    async def process_all_batches(self)-> list:
        if not self.image_paths or not isinstance(self.image_paths, list):
            raise ValueError("Invalid image paths.")

        # Number of CPUs
        num_cpus = os.cpu_count()
        logging.info(f"Number of CPUs: {num_cpus}")
        # Total Memory
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total / (1024**3)  # Convert bytes to GB
        # Free Memory in GB
        free_memory_gb = memory_info.free / (1024**3)
        logging.info(f"Total Memory: {total_memory}")
        logging.info(f"Free Memory: {free_memory_gb}")

        batch_size = BATCH_SIZE 
        pool = Pool(processes=num_cpus)

        batches = [self.image_paths[i: i + batch_size] for i in range(0, len(self.image_paths), batch_size)]
        total_faces_extracted = 0
        total_failed_to_detect = 0
        total_failed_to_embbed = 0
        total_low_conf_face = 0
        total_not_clear_face = 0
        total_no_faces_images = 0

        # Process batches in parallel
        results = [pool.apply_async(process_batch, args=(batch, self.session_key, SEMAPHORE_ALLOWED)) for batch in batches]

        

        pool.close()
        pool.join()

        all_results = []

        # Wait for all processes to complete
        for result in results:
            result_tupple = result.get()  # get() will block until the result is ready
            total_faces_extracted += result_tupple[0][0]
            total_failed_to_detect += result_tupple[0][1]
            total_failed_to_embbed += result_tupple[0][2]
            total_low_conf_face += result_tupple[0][3]
            total_not_clear_face += result_tupple[0][4]
            total_no_faces_images += result_tupple[0][5]
            all_results.extend(result_tupple[1])

        # Logging
        logging.info(f"Total number of faces extracted: {total_faces_extracted}")
        logging.info(f"Total number of failed to detect faces: {total_failed_to_detect}")
        logging.info(f"Total number of failed to embbed faces: {total_failed_to_embbed}")
        logging.info(f"Total number of low confidence faces: { total_low_conf_face}")
        logging.info(f"Total number of not clear faces: {total_not_clear_face}")
        logging.info(f"Total number of no faces images: {total_no_faces_images}")  

        return all_results 
    
    async def execute(self)-> list:
        all_results = await self.process_all_batches()
        logging.info("Preprocessing completed successfully.")
        return all_results

# multi process helper function
def process_batch(batch_image_paths, session_key, semaphore_value):
    preprocessor = PeepsPreProcessor(session_key)
    asyncio.run(preprocessor.preprocess_and_upload_batch_async(batch_image_paths, semaphore_value))   
    result_tupple = (preprocessor.cropped_faces_count, preprocessor.failed_to_detect_index, preprocessor.failed_to_embbed_index, preprocessor.low_conf_face_index, preprocessor.not_clear_face_index, preprocessor.no_faces_images_index)     
    return (result_tupple, preprocessor.results)