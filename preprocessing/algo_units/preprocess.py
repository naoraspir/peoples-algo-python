import copy
from gc import collect
import io
import json
# from multiprocessing import Pool
import concurrent
from typing import List, Tuple
import cv2
import asyncio
import numpy as np
from google.cloud import storage
import torch
import logging
import os
# See github.com/timesler/facenet-pytorch:
import psutil
from common.consts_and_utils import BATCH_SIZE, BUCKET_NAME, CONF_THRESHOLD, MAX_WEB_IMAGE_HEIGHT, MAX_WEB_IMAGE_WIDTH, PREPROCESS_FOLDER, RAW_DATA_FOLDER, WEB_DATA_FOLDER, is_clear
from preprocessing.gcs_utils import get_image_paths_from_bucket, upload_to_gcs, download_images_batch
from facenet_pytorch import InceptionResnetV1, MTCNN



logging.basicConfig(level=logging.DEBUG)
logging.getLogger('numba.core').setLevel(logging.INFO)
# change level of debug for urllib3.connectionpool to INFO
logging.getLogger('urllib3').setLevel(logging.INFO)

logging.getLogger('google.auth').setLevel(logging.INFO)

class PeepsPreProcessor:

    def __init__(self, session_key: str,)-> None:
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device}")
    
            self.session_key = session_key
            self.bucket_name = BUCKET_NAME
            self.preprocess_folder = PREPROCESS_FOLDER
            self.raw_folder = RAW_DATA_FOLDER

            self.cropped_faces_count = 0  # Initialize a counter for cropped faces
            self.failed_to_detect_index = 0  # Initialize a counter for failed faces
            self.failed_to_embbed_index = 0  # Initialize a counter for failed faces  
            self.low_conf_face_index = 0  # Initialize a counter for low confidence faces
            self.not_clear_face_index = 0  # Initialize a counter for not clear faces
            self.no_faces_images_index = 0  # Initialize a counter for no faces detected in image
            
            self.deferred_tasks = []  # Initialize the deferred tasks list
            self.results = []  # New attribute to store results

        except Exception as e:
            logging.error(f"Initialization error: {e}")
    
    def queue_deferred_upload(self, data, destination_path, content_type='image/jpeg')-> None:
        # Queue the task for later
        storage_client = storage.Client()
        source_bucket = storage_client.get_bucket(self.bucket_name)
        self.deferred_tasks.append((source_bucket, data, destination_path, content_type))

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
        return img_resized

    def crop_faces_from_image_and_embed(self, img, embeddings_lambda , mtcnn  ) -> Tuple[List[np.array], List[np.array]]:
        if not isinstance(img, np.ndarray) or len(img.shape) != 3:
            raise ValueError("Invalid image provided.")
        
        

        cropped_faces, face_encodings = [], []
        # logging.info("start of processing new image")
        
        # Process the image
        # image_for_extraction = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_for_extraction = img

        try:
            with torch.no_grad():
                boxes, probs = mtcnn.detect(image_for_extraction)
        except Exception as ex:
            logging.error(f"Error in faces detection: {ex}")
            raise (ex)

        if boxes is not None and probs is not None:
            #log the number of initial faces detected
            # logging.info("Number of initial faces detected: "+str(len(boxes)))
            for box, prob in zip(boxes, probs):
                x, y, w, h = box.tolist()
                x, y, w, h = map(int, [x, y, w, h])  # Convert to int

                # Determine padding size
                padding = int(0.6 * (w - x))  # Example: 60% of the face width

                # Apply padding and ensure coordinates are within image bounds
                x_padded = max(x - padding, 0)
                y_padded = max(y - padding, 0)
                w_padded = min(w + padding, img.shape[1])
                h_padded = min(h + padding, img.shape[0])
                if h_padded <= img.shape[0] and w_padded <= img.shape[1] and w_padded > x_padded and h_padded > y_padded:
                    try:
                        face_crop = image_for_extraction[y_padded:h_padded, x_padded:w_padded]
                        face_for_clear_check = image_for_extraction[y:h, x:w]
                    
                    except Exception as ex:
                        logging.error(f"Error in face extraction: {ex}")
                        raise(ex)

                    if prob >= CONF_THRESHOLD:
                        if is_clear(image=img, face=face_for_clear_check): # Check if the face is clear    
                            try:
                                with torch.no_grad():
                                    # #resize face_crop to 244x244
                                    face_crop = cv2.resize(face_crop, (244,244), interpolation=cv2.INTER_LINEAR)
                                    face_for_embedding, prob = mtcnn(face_crop, return_prob=True)
                                    if face_for_embedding is not None:
                                        face_for_embedding = face_for_embedding.unsqueeze(0).to(self.device)
                                        e = embeddings_lambda(face_for_embedding).squeeze().cpu().numpy()
                                        face_encodings.append(e)
                                        cropped_faces.append(face_crop)
                                        self.cropped_faces_count += 1
                                    else:                                        
                                        self._save_failed_image(face_crop, 'failed_to_detect')
                                        self.failed_to_detect_index += 1
                            except Exception as e:
                                logging.error(f"Error getting face encoding: {e}")
                                self._save_failed_image(face_crop, 'failed_to_embbed')
                                self.failed_to_embbed_index += 1
                        else:
                            self._save_failed_image(face_crop, 'not_clear_face') 
                            self.not_clear_face_index += 1  
                    else:
                        self._save_failed_image(face_crop, 'low_conf_face')
                        self.low_conf_face_index += 1
                else:
                    logging.error("Face coordinates are out of bounds")
        else:
            self._save_failed_image(image_for_extraction, 'no_faces_images')
            self.no_faces_images_index += 1

        logging.info(f"Number of taken faces: {len(cropped_faces)}")
        return cropped_faces, face_encodings   

    async def process_faces_and_embeddings(self, img, image_path , embeddings_lambda, mtcnn)-> None:
        """
        Crop faces from an image, extract embeddings, and queue for upload.
        
        Parameters:
        img (np.array): The image from which faces will be cropped.
        image_path (str): The path of the image being processed.
        """
        cropped_faces, face_encodings = self.crop_faces_from_image_and_embed(img, embeddings_lambda , mtcnn)
        for face_crop_resized, encoding in zip(cropped_faces, face_encodings):
            self.results.append({
            "embedding": encoding,
            "face": face_crop_resized,
            "original_path": image_path
            })

    # Helper methods for each step of the process
    async def preprocess_and_upload_batch(self, batch_image_paths)-> None:
        """
        Process a batch of image paths: download the images, preprocess them,
        and queue them for upload to GCS.
        
        Parameters:
        batch_image_paths (List[str]): A list of image paths to process.
        """
        storage_client = storage.Client()
        source_bucket = storage_client.get_bucket(self.bucket_name)
        images = await download_images_batch(source_bucket, batch_image_paths)
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total / (1024**3)  # Convert bytes to GB
        # Free Memory in GB
        free_memory_gb = memory_info.free / (1024**3)
        logging.info(f"Total Memory for process in the start of the new batch: {total_memory}")
        logging.info(f"Free Memory for process in the start of the new batch: {free_memory_gb}")
        try:
            # logging.info("Loading the facial recognition model...")
            resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
            embeddings_lambda = lambda input: resnet(input)
            mtcnn = MTCNN(
                image_size=160, margin=80, min_face_size=85,
                thresholds=[0.6, 0.7, 0.7], factor=0.65, post_process=True,
                device=self.device
            ).eval()

        except Exception as e:
            logging.error(f"Error loading the facial recognition model: {e}")
            raise(e)

        for img, image_path in zip(images, batch_image_paths):
            try:
                web_sized_image = self.resize_and_upload_for_web(img, image_path)  # RGB
                image_for_extraction = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB
                await self.process_faces_and_embeddings(image_for_extraction, image_path, embeddings_lambda, mtcnn)  # always run the algo with web resolution.
            except Exception as e:
                logging.error(f"Error processing image {image_path}: {e}")

    async def preprocess_and_upload_batch_async(self, batch_image_paths)-> None:
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

    # multi process helper function
    def process_batch(self, batch_image_paths):
        preprocessor = PeepsPreProcessor(self.session_key)
        asyncio.run(preprocessor.preprocess_and_upload_batch_async(batch_image_paths))

        # Create a copy of the results
        result_tuple = (
            preprocessor.cropped_faces_count,
            preprocessor.failed_to_detect_index,
            preprocessor.failed_to_embbed_index,
            preprocessor.low_conf_face_index,
            preprocessor.not_clear_face_index,
            preprocessor.no_faces_images_index
        )
        results_copy = copy.deepcopy(preprocessor.results)

        # Delete the preprocessor object
        del preprocessor

        # Run garbage collection
        collect()

        return (result_tuple, results_copy)

    def process_all_batches(self)-> list:
        # Initialize the storage client and get the source bucket
        storage_client = storage.Client()
        source_bucket = storage_client.get_bucket(self.bucket_name)
        image_paths = get_image_paths_from_bucket(self.session_key, storage_client, self.bucket_name, self.raw_folder)
        if not image_paths or not isinstance(image_paths, list):
            raise ValueError("Invalid image paths.")
        logging.info(f"Number of images to process: {len(image_paths)}")
        # return None
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
        logging.info(f"Batch size: {batch_size}")

        batches = [image_paths[i: i + batch_size] for i in range(0, len(image_paths), batch_size)]
        # total_faces_extracted = 0
        # total_failed_to_detect = 0
        # total_failed_to_embbed = 0
        # total_low_conf_face = 0
        # total_not_clear_face = 0
        # total_no_faces_images = 0

        # Process batches in parallel
        # results = [pool.apply_async(process_batch, args=(batch, self.session_key)) for idx,batch in enumerate(batches)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(self.process_batch, batches))

        all_results = []

        # Wait for all processes to complete
        for result_tupple in results:
            # result_tupple = result # get() will block until the result is ready
            # total_faces_extracted += result_tupple[0][0]
            # total_failed_to_detect += result_tupple[0][1]
            # total_failed_to_embbed += result_tupple[0][2]
            # total_low_conf_face += result_tupple[0][3]
            # total_not_clear_face += result_tupple[0][4]
            # total_no_faces_images += result_tupple[0][5]
            all_results.extend(result_tupple[1])

        return all_results 
    
    #save all the data to gcs:
    async def store_preprocess_artifacts_to_gcs(self):
       
        embeddings, faces, original_paths = self.aggregate_preprocess_artifacts()
        preprocess_folder = f"{self.session_key}/{self.preprocess_folder}"
        
        storage_client = storage.Client()
        source_bucket = storage_client.get_bucket(self.bucket_name)

        try:
            # Process embeddings
            embeddings_array = np.array(embeddings)
            embeddings_bytes = io.BytesIO()
            np.save(embeddings_bytes, embeddings_array, allow_pickle=True)
            embeddings_bytes.seek(0)
            await upload_to_gcs(source_bucket, embeddings_bytes.read(), f"{preprocess_folder}/embeddings.npy", content_type='application/octet-stream')
        except Exception as e:
            logging.error(f"Error uploading embeddings: {e}")
            raise(e)
        
        try:
            # Process faces
            # Note: We will store the faces as a list of bytes objects in an object array, not a regular ndarray.
            encoded_faces = [cv2.imencode('.jpg', face)[1].tobytes() for face in faces]  # Encode each face and store in a list
            faces_array = np.array(encoded_faces, dtype=object)  # Create an object array of bytes objects
            faces_bytes = io.BytesIO()
            np.save(faces_bytes, faces_array, allow_pickle=True)  # Save the object array to BytesIO buffer
            faces_bytes.seek(0)
            await upload_to_gcs(source_bucket, faces_bytes.read(), f"{preprocess_folder}/faces.npy", content_type='application/octet-stream')
        except Exception as e:
            logging.error(f"Error uploading faces: {e}")
            raise e
        
        try:
            # Process original paths
            original_paths_json = json.dumps(original_paths)
            original_paths_bytes = original_paths_json.encode()
            await upload_to_gcs(source_bucket, original_paths_bytes, f"{preprocess_folder}/original_paths.json", content_type='application/json')
        except Exception as e:
            logging.error(f"Error uploading original paths: {e}")
            raise(e)
        
    def aggregate_preprocess_artifacts(self):
        embeddings = []
        faces = []
        original_paths = []
        for result in self.results:
            embeddings.append(result['embedding'])
            faces.append(result['face'])
            original_paths.append(result['original_path'])
        return embeddings, faces, original_paths
    
    #entry point for the class
    def execute(self):
        all_results = self.process_all_batches()
        logging.info("Preprocessing completed successfully.")
        # return all_results
        self.results = all_results
        asyncio.run(self.store_preprocess_artifacts_to_gcs())
        logging.info("Preprocessing artifacts uploaded successfully.")
