import io
from multiprocessing import Pool
from typing import List, Tuple
from PIL import Image
import cv2
import asyncio
from deepface import DeepFace
import numpy as np
from google.cloud import storage
from requests import RequestException
import torch
from consts_and_utils import BATCH_SIZE, BUCKET_NAME, CONF_THRESHOLD, DETECTORS, MODELS, RAW_DATA_FOLDER, SEMAPHORE_ALLOWED, is_clear
import logging
import os
import hashlib
import face_recognition
# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision.transforms import ToTensor
import psutil

logging.basicConfig(level=logging.DEBUG)
def np_array_to_tensor(np_image):
    # Convert np.array image to PIL Image
    pil_image = Image.fromarray(np_image)
    # Apply transformation
    return ToTensor()(pil_image).unsqueeze(0)

def prepare_face(image_array) -> np.array:#TO RGB
        # Check if the image data is normalized (0.0 to 1.0)
        if image_array.max() <= 1.0:
            # Scale to 0-255 and convert to uint8
            image_array = (image_array * 255).astype(np.uint8)

        # If the image has a single channel, convert it to a 3-channel image by duplicating the channels
        if image_array.ndim == 2 or (image_array.ndim == 3 and image_array.shape[2] == 1):
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        # If the image is in BGR format (common in OpenCV), convert it to RGB for proper display
        if image_array.shape[2] == 3:  # Check if there are three channels
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        return image_array

class PeepsPreProcessor:

    def __init__(self, session_key: str):
        try:
            self.session_key = session_key
            self.storage_client = storage.Client()
            self.bucket_name = BUCKET_NAME
            self.source_bucket = self.storage_client.get_bucket(self.bucket_name)
            self.image_paths = self.get_image_paths_from_bucket()
            self.preprocess_folder = 'preprocess'
            self.cropped_faces_count = 0  # Initialize a counter for cropped faces
            self.failed_to_detect_index = 0  # Initialize a counter for failed faces
            self.failed_to_embbed_index = 0  # Initialize a counter for failed faces  
            self.low_conf_face_index = 0  # Initialize a counter for low confidence faces
            self.not_clear_face_index = 0  # Initialize a counter for not clear faces
            self.no_faces_images_index = 0  # Initialize a counter for no faces detected in image
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

            

        except Exception as e:
            logging.error(f"Initialization error: {e}")
    
    async def download_image_from_gcs(self, image_path: str) -> np.array:
        if not image_path or not isinstance(image_path, str):
            raise ValueError("Invalid image path provided.")  

        try:
            blob = self.source_bucket.blob(image_path)
            img_data = blob.download_as_bytes()
            # Decode the image data from BGR (default in OpenCV) to RGB
            img_bgr = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            logging.error(f"Error downloading image: {e}")
            raise(e)

    def get_image_paths_from_bucket(self) -> list:
        if not self.session_key or not isinstance(self.session_key, str):
            raise ValueError("Invalid session key.")
        
        try:
            folder_path = f'{self.session_key}/{RAW_DATA_FOLDER}'
            blobs = list(self.storage_client.list_blobs(self.bucket_name, prefix=folder_path))
            return [blob.name for blob in blobs if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except Exception as e:
            logging.error(f"Error fetching images from bucket: {e}")
            #raise the releveant error for not being able to download
            raise(e)
        
    async def upload_to_gcs(self, data, destination_path, content_type='image/jpeg'):
        try:
            blob = self.source_bucket.blob(destination_path)
            blob.upload_from_string(data, content_type=content_type)
        except RequestException as re:
            logging.error(f"HTTP request failed during upload: {re}")
        except Exception as e:
            logging.error(f"Error uploading to GCS: {e}")
    
    def queue_deferred_upload(self, data, destination_path, content_type='image/jpeg'):
        # Queue the task for later
        self.deferred_tasks.append((data, destination_path, content_type))

    def _save_failed_image(self, face_crop, failure_type):
        try:
            _, buffer = cv2.imencode('.jpg', face_crop)
            face_data = buffer.tobytes()
            unique_identifier_path = f"{failure_type}/face_{getattr(self, f'{failure_type}_index')}.jpg"
            destination_path = f"{self.session_key}/{unique_identifier_path}"
            
            # Add the upload task to deferred tasks
            # self.deferred_tasks.append((face_data, destination_path, 'image/jpeg'))
            #log the task tuple 
            #logging.info("task tuple: "+str((face_data, destination_path, 'image/jpeg')))
            self.queue_deferred_upload(face_data, destination_path, 'image/jpeg')
            # Increment the counter
            # setattr(self, f'{failure_type}_index', getattr(self, f'{failure_type}_index') + 1)
        except Exception as e:
            logging.error(f"Error saving {failure_type} image: {e}")

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
                                    #resize face_crop to 244x244
                                    face_crop_resized = cv2.resize(face_crop, (244,244), interpolation=cv2.INTER_LINEAR)
                                    if face_for_embedding is not None:
                                        face_for_embedding = face_for_embedding.unsqueeze(0).to(self.device)
                                        e = self.embeddings_lambda(face_for_embedding).squeeze().cpu().numpy()
                                        face_encodings.append(e)
                                        cropped_faces.append(face_crop_resized)
                                        self.cropped_faces_count += 1
                                    else:
                                        
                                        # asyncio.create_task(self._save_failed_image(face_crop_resized, 'failed_to_detect'))
                                        self._save_failed_image(face_crop_resized, 'failed_to_detect')
                                        self.failed_to_detect_index += 1
                            except Exception as e:
                                logging.error(f"Error getting face encoding: {e}")
                                # asyncio.create_task(self._save_failed_image(face_crop, 'failed_to_embbed'))
                                self._save_failed_image(face_crop, 'failed_to_embbed')
                                self.failed_to_embbed_index += 1
                        else:
                            #  asyncio.create_task(self._save_failed_image(face_crop, 'not_clear_face'))
                            self._save_failed_image(face_crop, 'not_clear_face') 
                            self.not_clear_face_index += 1  
                    else:
                        # asyncio.create_task(self._save_failed_image(face_crop, 'low_conf_face'))
                        self._save_failed_image(face_crop, 'low_conf_face')
                        self.low_conf_face_index += 1
                else:
                    logging.error("Face coordinates are out of bounds")
        else:
            # logging.info("No face detected in entire image")
            # asyncio.create_task(self._save_failed_image(image_for_extraction, 'no_faces_images'))
            self._save_failed_image(image_for_extraction, 'no_faces_images')
            self.no_faces_images_index += 1

        logging.info(f"Number of extracted faces: {len(cropped_faces)}")#TODO: erase logging
        return cropped_faces, face_encodings   

    async def preprocess_and_upload_batch(self, batch_image_paths):
        if not batch_image_paths or not isinstance(batch_image_paths, list):
            raise ValueError("Invalid batch image paths provided.")


        for image_path in batch_image_paths:
            try:
                img = await self.download_image_from_gcs(image_path)

            except RequestException as re:
                logging.error(f"HTTP request failed while downloading image {image_path}: {re}")
                continue  # Move to next image if this one fails
            except Exception as e:
                logging.error(f"Error downloading image {image_path}: {e}")
                continue


            try:
                # Now we receive both faces and their encodings
                cropped_faces, face_encodings = self.crop_faces_from_image_and_embed(img)
            except Exception as e:
                logging.error(f"Error cropping faces and embedding from image {image_path}: {e}")
                raise(e)
                # continue
        
            # Iterate over cropped faces and their encodings
            for idx, (face, encoding) in enumerate(zip(cropped_faces, face_encodings)):
                try:
                    embedding_folder_name = image_path.split('/')[-1].split('.')[0] + f"_face_{idx}"
                except Exception as e:
                    logging.error(f"Error getting hash for embedding for face {idx} from image {image_path}: {e}")
                    raise(e)

                try:
                    # Convert from BGR to RGB
                    processed_face = face
                    # Upload cropped face
                    img_encoded = cv2.imencode('.jpg', processed_face)[1]
                    face_data = img_encoded.tobytes()
                    face_destination_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_folder_name}/face.jpg"
                    # asyncio.create_task(self.upload_to_gcs(face_data, face_destination_path))
                    self.queue_deferred_upload(face_data, face_destination_path)
                except Exception as e:
                    logging.error(f"Error uploading cropped face {idx} from image {image_path}: {e}")
                    raise(e)

                try:
                    # Prepare the bytes to upload
                    embedding_bytes = io.BytesIO()
                    np.save(embedding_bytes, encoding, allow_pickle=True)
                    embedding_bytes.seek(0)  # Important: move back to the start of the BytesIO object

                    # Upload to GCS
                    embedding_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_folder_name}/embedding.npy"
                    # asyncio.create_task(self.upload_to_gcs(embedding_bytes.read(), embedding_path, content_type='application/octet-stream'))
                    self.queue_deferred_upload(embedding_bytes.read(), embedding_path, content_type='application/octet-stream')
                except Exception as e:
                    logging.error(f"Error uploading embedding for face {idx} from image {image_path}: {e}")
                    continue

                try:
                    # Save the original image path
                    original_image_data = image_path.encode()  # Assuming image_path is a string
                    original_image_name = "original_path.txt"
                    original_image_destination_path = f"{self.session_key}/{self.preprocess_folder}/{embedding_folder_name}/{original_image_name}"
                    # asyncio.create_task(self.upload_to_gcs(original_image_data, original_image_destination_path,content_type='text/plain'))
                    self.queue_deferred_upload(original_image_data, original_image_destination_path,content_type='text/plain')

                except Exception as e:
                    logging.error(f"Error saving original image path for face {idx} from image {image_path}: {e}")
                    continue
        

    async def preprocess_and_upload_batch_async(self, batch_image_paths, semaphore_value):
        semaphore = asyncio.Semaphore(semaphore_value)
        async with semaphore:
            await self.preprocess_and_upload_batch(batch_image_paths)

        # Ensure that deferred tasks are iterables before gathering
        if self.deferred_tasks:
            await asyncio.gather(*(self.upload_to_gcs(*task) for task in self.deferred_tasks if task is not None))
        self.deferred_tasks.clear()

    async def process_all_batches(self):
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

        # Wait for all processes to complete
        for result in results:
            result_tupple = result.get()  # get() will block until the result is ready
            total_faces_extracted += result_tupple[0]
            total_failed_to_detect += result_tupple[1]
            total_failed_to_embbed += result_tupple[2]
            total_low_conf_face += result_tupple[3]
            total_not_clear_face += result_tupple[4]
            total_no_faces_images += result_tupple[5]

        # Logging
        logging.info(f"Total number of faces extracted: {total_faces_extracted}")
        logging.info(f"Total number of failed to detect faces: {total_failed_to_detect}")
        logging.info(f"Total number of failed to embbed faces: {total_failed_to_embbed}")
        logging.info(f"Total number of low confidence faces: { total_low_conf_face}")
        logging.info(f"Total number of not clear faces: {total_not_clear_face}")
        logging.info(f"Total number of no faces images: {total_no_faces_images}")   
    
    async def execute(self):
        await self.process_all_batches()
        
        logging.info("Preprocessing completed successfully.")

def process_batch(batch_image_paths, session_key, semaphore_value):
    preprocessor = PeepsPreProcessor(session_key)
    asyncio.run(preprocessor.preprocess_and_upload_batch_async(batch_image_paths, semaphore_value))   
    result_tupple = (preprocessor.cropped_faces_count, preprocessor.failed_to_detect_index, preprocessor.failed_to_embbed_index, preprocessor.low_conf_face_index, preprocessor.not_clear_face_index, preprocessor.no_faces_images_index)     
    return result_tupple