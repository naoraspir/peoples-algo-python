import gc
import io
import json
# from multiprocessing import Pool
from threading import Lock
import time
import asyncio
import numpy as np
from google.cloud import storage
import torch
import logging
import os
# See github.com/timesler/facenet-pytorch:
import psutil
from PIL import Image
from io import BytesIO
from common.consts_and_utils import BATCH_SIZE, BUCKET_NAME, CONF_THRESHOLD, MAX_WEB_IMAGE_HEIGHT, MAX_WEB_IMAGE_WIDTH, PREPROCESS_FOLDER, RAW_DATA_FOLDER, WEB_DATA_FOLDER, get_laplacian_variance, is_clear, process_images_with_pil
from preprocessing.algo_units.metric_calculator import PeepsMetricCalculator
from preprocessing.gcs_utils import upload_to_gcs, download_images_batch
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
            self.web_folder = WEB_DATA_FOLDER

            self.cropped_faces_count = 0  # Initialize a counter for cropped faces
            self.failed_to_detect_index = 0  # Initialize a counter for failed faces
            self.failed_to_embbed_index = 0  # Initialize a counter for failed faces  
            self.low_conf_face_index = 0  # Initialize a counter for low confidence faces
            self.not_clear_face_index = 0  # Initialize a counter for not clear faces
            self.no_faces_images_index = 0  # Initialize a counter for no faces detected in image
            
            self.deferred_tasks = []  # Initialize the deferred tasks list
            self.results = []  # New attribute to store results
            self.results_lock = Lock()  # Thread lock for safe access to self.results

            self.paths_times = []  # New attribute to store time image was taken by photographer for each image

            self.storage_client = storage.Client()
            self.source_bucket = self.storage_client.get_bucket(self.bucket_name)
            
            self.metrics_calculator = PeepsMetricCalculator()
            self.metrics_calculator_lock = Lock()
            try:
                # logging.info("Loading the facial recognition model...")
                self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
                # self.embeddings_lambda = lambda input: self.resnet(input)
                self.mtcnn = MTCNN(
                    image_size=160, margin=0, min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7], factor=0.65, post_process=True,
                    device=self.device, keep_all=True , selection_method="largest"
                ).eval()
                # Locks for thread-safe access to models
                self.mtcnn_lock = Lock()
                self.resnet_lock = Lock()

            except Exception as e:
                logging.error(f"Error loading the facial recognition model: {e}")
                raise(e)

        except Exception as e:
            logging.error(f"Initialization error: {e}")

    # Helper methods for each step of the process
    def batch_process_faces_and_embeddings(self, images, images_paths):
        # Images are RGB PIL images at entrance to this function
        try:
            cropped_faces, origin_image_paths, additional_metrics, face_batch_to_embbed_list = [], [], [], []

            # Initialize lists for batch data
            with torch.no_grad():
                try:
                    # Log start of detection
                    start_time = time.time()
                    # Detect faces using MTCNN (directly with PIL images)
                    batch_boxes, batch_probs, batch_landmarks = self.mtcnn.detect(images, landmarks=True)
                    elapsed_time = time.time() - start_time
                    logging.info(f"Detecting faces took {elapsed_time:.2f} seconds")

                    if batch_boxes is not None and len(batch_boxes) > 0 and batch_boxes[0] is not None:
                        # Log time it took to extract faces
                        start_time = time.time()
                        # Extract faces using MTCNN (directly with PIL images)
                        batch_faces = self.mtcnn.extract(images, batch_boxes, save_path=None)
                        elapsed_time = time.time() - start_time
                        logging.info(f"Extracting faces took {elapsed_time:.2f} seconds")
                    else:
                        batch_faces = [None] * len(images)

                except Exception as e:
                    logging.error(f"Error detecting faces in micro-batch of size {len(images)}: {e}")
                    raise(e)

            # Ensure the lengths of the lists are consistent
            assert len(images_paths) == len(batch_faces), "Mismatch in the number of image paths and faces detected"

            # For each image in the batch
            for img, images_path, faces, boxes, probs, landmarks in zip(images, images_paths, batch_faces, batch_boxes, batch_probs, batch_landmarks):
                if faces is None:
                    continue

                # Calculate laplacian variance for the image (convert PIL to NumPy just for this function)
                img_np = np.array(img)
                img_laplacian_variance = get_laplacian_variance(img_np)
                img_faces_count = len(faces)

                logging.info(f"There are {img_faces_count} faces in the image: {images_path}")

                # For each face in the image
                for index, (face, box, prob, landmark) in enumerate(zip(faces, boxes, probs, landmarks)):
                    x, y, w, h = box.tolist()
                    x, y, w, h = map(int, [x, y, w, h])

                    # Crop face using PIL
                    face_original_crop_sized = img.crop((x, y, w, h))

                    # get laplacian variance for the face (convert to NumPy for this function)
                    face_np = np.array(face_original_crop_sized)
                    laplacian_variance_face = get_laplacian_variance(face_np)

                    #check if image is_clear and prob is higher the CONF_THRESHOLD
                    # Check if the face is large enough relative to the original image and clear
                    # if prob < CONF_THRESHOLD or not is_clear(img_np, face_np):
                    #     continue  

                    # Crop face with 0.6 padding using PIL
                    padding = int(0.6 * (w - x))
                    x_padded, y_padded, w_padded, h_padded = max(x - padding, 0), max(y - padding, 0), min(w + padding, img.width), min(h + padding, img.height)
                    face_crop_for_ui = img.crop((x_padded, y_padded, w_padded, h_padded))

                    # Resize face for UI using PIL
                    face_resized = face_crop_for_ui.resize((244, 244), Image.LANCZOS)
                    cropped_faces.append(face_resized)

                    # Calculate additional metrics
                    try:
                        face_metrics = self.metrics_calculator.calculate_face_metrics(
                            box, img_np.shape, prob, img_faces_count, img_laplacian_variance, laplacian_variance_face, 
                            face_landmarks=landmark, face_idx=index, face_image=face_np
                        )
                    except Exception as e:
                        logging.error(f"Error calculating additional metrics: {e}")
                        raise(e)
                    additional_metrics.append(face_metrics)

                    # Add face to the list of faces for batch embedding
                    face_batch_to_embbed_list.append(face)

                    # Add original image path to the list of original image paths
                    origin_image_paths.append(images_path)

            # Batch process embeddings for all cropped faces
            if face_batch_to_embbed_list:
                try:
                    face_batch_to_embbed_tensor = torch.stack(face_batch_to_embbed_list).to(self.device)
                    # Log time it takes the resnet to extract embeddings
                    start_time = time.time()
                    with torch.no_grad():
                        embeddings = self.resnet(face_batch_to_embbed_tensor)
                    embeddings = embeddings.cpu().numpy().tolist()
                    elapsed_time = time.time() - start_time
                    logging.info(f"Batch extracting embeddings took {elapsed_time:.2f} seconds for a batch of size {len(face_batch_to_embbed_list)}")
                except Exception as e:
                    logging.error(f"Error extracting embeddings: {e}")
                    raise(e)

                # Check that all lists are equal in length
                if len(embeddings) == len(cropped_faces) == len(additional_metrics) == len(origin_image_paths):
                    for emb, face, metrics, path in zip(embeddings, cropped_faces, additional_metrics, origin_image_paths):
                        # Use the lock to safely append to self.results
                        self.results.append({
                            "embedding": emb,
                            "face": face,  # Already a PIL image
                            "original_path": path,
                            "metrics": metrics  # Include the additional metrics here
                        })
                else:
                    logging.error(f"Error extracting embeddings: not equal length of lists :")
                    logging.error(f"embeddings: {len(embeddings)}")
                    logging.error(f"cropped_faces: {len(cropped_faces)}")
                    logging.error(f"additional_metrics: {len(additional_metrics)}")
                    logging.error(f"origin_image_paths: {len(origin_image_paths)}")
                    raise ValueError("Error extracting embeddings: not equal length of lists")
               
                # Minimal change: free temporary variables before exiting
                del cropped_faces, additional_metrics, origin_image_paths, face_batch_to_embbed_list, embeddings
                gc.collect()

        except Exception as e:
            logging.error(f"Error processing faces and embeddings: {e}")
            raise(e)

    

        # Helper methods for each step of the process
    
    def preprocess_and_upload_batch(self, batch_image_paths):
        """
        Process a batch of image paths: download the images, preprocess them,
        and queue them for upload to GCS.
        
        Parameters:
        batch_image_paths (List[str]): A list of image paths to process.
        """
        
        #log the time it takes to download the images
        logging.info(f"Downloading images for batch of size {len(batch_image_paths)}...")
        start_time = time.time()
        all_images_path_pairs = asyncio.run(download_images_batch(self.source_bucket, batch_image_paths))
        elapsed_time = time.time() - start_time
        logging.info(f"Downloading images took {elapsed_time:.2f} seconds")

        memory_info = psutil.virtual_memory()
        # total_memory = memory_info.total / (1024**3)  # Convert bytes to GB
        free_memory_gb = memory_info.free / (1024**3)
        # logging.info(f"Total Memory at start of batch: {total_memory} GB")
        logging.info(f"Free Memory at start of batch: {free_memory_gb} GB")
        #measure the time it takes to process the images
        start_time = time.time()
        # Group images by size
        size_to_images = {}
        for img, path, datetime_taken in all_images_path_pairs:
            self.paths_times.append((path, datetime_taken))
            
            # Get size using PIL's size attribute (width, height)
            size = img.size  # This returns (width, height)
            
            if size not in size_to_images:
                size_to_images[size] = {'images': [], 'paths': []}
            
            # Append the PIL image and path
            size_to_images[size]['images'].append(img)
            size_to_images[size]['paths'].append(path)

        # Process each size-based micro-batch
        for size, data in size_to_images.items():
            micro_batch_images = data['images']
            micro_batch_paths = data['paths']
            logging.info(f"Processing micro-batch of size {size} with {len(micro_batch_images)} images")
            self.process_micro_batch(micro_batch_images, micro_batch_paths)
        
        # Clear the grouping dictionary after processing
        del size_to_images, all_images_path_pairs
        gc.collect()
        
        elapsed_time = time.time() - start_time
        logging.info(f"Processing images took {elapsed_time:.2f} seconds")
        # New part - Upload all deferred tasks at the end of processing a batch
        # measure the time it takes to upload the deferred tasks
        start_time = time.time()    
        asyncio.run(self.upload_deferred_tasks())
        elapsed_time = time.time() - start_time
        logging.info(f"Uploading deferred tasks took {elapsed_time:.2f} seconds")

    # Multi-process helper function
    def process_micro_batch(self, micro_batch_images, micro_batch_paths):
        try:
            # RGB PIL images are passed to this function
            rgb_images = [img for img in micro_batch_images]

            # Dynamically downscale images based on optimal resolution
            rgb_downscaled_images = process_images_with_pil(rgb_images)
            
            # Resize images to a fixed size of 800x800
            # rgb_resized_images = resize_images_fixed(rgb_images,target_size=(1024, 1024))

            # Process all images in the micro-batch together
            self.batch_process_faces_and_embeddings(rgb_downscaled_images, micro_batch_paths)

            # Optionally, add web resizing and uploading here
            self.batch_resize_and_upload_for_web(rgb_images, micro_batch_paths)
        except Exception as e:
            logging.error(f"Error processing micro-batch: {e}")
    
    def process_all_batches(self , images_paths):
        # Get all image paths from the raw folder
        image_paths = images_paths
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
        # logging.info(f"Total Memory: {total_memory}")
        logging.info(f"Free Memory at start of chunk: {free_memory_gb}")

        batch_size = BATCH_SIZE 
        logging.info(f"Batch size: {batch_size}")

        batches = [image_paths[i: i + batch_size] for i in range(0, len(image_paths), batch_size)]
        self.results = []  # Reset results for a new processing session
        
        for batch in batches:
            # self.process_batch(batch)
            self.preprocess_and_upload_batch(batch)
            gc.collect()

        logging.info("All batches processed successfully.")
    
    # Function to aggregate data for each face
    def aggregate_face_data(self):
        embeddings = []
        faces = []
        original_paths = []
        metrics_list = []

        for result in self.results:
            embeddings.append(result['embedding'])
            faces.append(result['face'])
            original_paths.append(result['original_path'])
            metrics_list.append(result['metrics'])  # Assuming 'metrics' is a dictionary

        return embeddings, faces, original_paths, metrics_list
    
    #save all the data to gcs:
    async def store_preprocess_artifacts_to_gcs(self):
        try:
            embeddings, faces, original_paths, metrics_list = self.aggregate_face_data()
            preprocess_folder = f"{self.session_key}/{self.preprocess_folder}"

            # storage_client = storage.Client()
            source_bucket = self.source_bucket  #storage_client.get_bucket(self.bucket_name)

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
            encoded_faces = []
            for face in faces:
                # Use BytesIO to encode the face as JPEG using PIL
                face_byte_arr = BytesIO()
                face.save(face_byte_arr, format='JPEG')  # Save the PIL image to a BytesIO stream in JPEG format
                encoded_faces.append(face_byte_arr.getvalue())  # Store the bytes object
            faces_array = np.array(encoded_faces, dtype=object)  # Create an object array of bytes objects
            faces_bytes = io.BytesIO()
            np.save(faces_bytes, faces_array, allow_pickle=True)  # Save the object array to BytesIO buffer
            faces_bytes.seek(0)
            await upload_to_gcs(source_bucket, faces_bytes.read(), f"{preprocess_folder}/faces.npy", content_type='application/octet-stream')

            # Save original paths
            original_paths_json = json.dumps(original_paths).encode()
            await upload_to_gcs(source_bucket, original_paths_json, f"{preprocess_folder}/original_paths.json", content_type='application/json')

            # Save metrics
            metrics_json = json.dumps([dict(metric) for metric in metrics_list]).encode()
            await upload_to_gcs(source_bucket, metrics_json, f"{preprocess_folder}/metrics.json", content_type='application/json')
        except Exception as e:
            logging.error(f"Error uploading preprocessing artifacts: {e}")
            raise e
    
    def store_aggregated_artifacts_to_gcs(self, aggregated_results, aggregated_paths_times):
        try:
            self.results = aggregated_results
            self.paths_times = aggregated_paths_times
            asyncio.run(self.store_preprocess_artifacts_to_gcs())
            asyncio.run(self.finalize_and_upload_metadata())
            logging.info("Preprocessing artifacts uploaded successfully.")
        except Exception as e:
            logging.error(f"Error storing aggregated preprocessing artifacts: {e}")
            raise e
        
    #entry point for the class
    def execute(self, images_paths: list):
        self.process_all_batches(images_paths= images_paths)
        # Now that all batches are processed, finalize and upload the metadata
        #measure the time it takes to finalize and upload the metadata
        start_time = time.time()
        elapsed_time = time.time() - start_time
        logging.info(f"Finalizing and uploading metadata took {elapsed_time:.2f} seconds")
        logging.info("Preprocessing completed successfully.")

        return (self.results.copy(), self.paths_times.copy())  # Return a copy of the results to avoid 
        # asyncio.run(self.store_preprocess_artifacts_to_gcs())
        # logging.info("Preprocessing artifacts uploaded successfully.")

    #resize and upload methods
    def batch_resize_and_upload_for_web(self, images, original_paths):
        """
        Resize and upload a batch of images for web use using PIL.
        
        Parameters:
        images (List[PIL.Image]): A list of PIL images to process.
        original_paths (List[str]): A list of original image paths.
        """
        web_images_data = []
        web_image_paths = []

        for img, original_path in zip(images, original_paths):
            # Calculate the scaling factor
            width, height = img.size
            height_factor = MAX_WEB_IMAGE_HEIGHT / height
            width_factor = MAX_WEB_IMAGE_WIDTH / width
            scale_factor = min(height_factor, width_factor)

            # Resize the image preserving the aspect ratio using PIL
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img_resized = img.resize(new_size, Image.LANCZOS)

            # Convert the image to bytes (JPEG format)
            img_byte_arr = BytesIO()
            img_resized.save(img_byte_arr, format='JPEG')
            web_image_data = img_byte_arr.getvalue()
            web_images_data.append(web_image_data)

            # Create the new path for the web-optimized image
            path_parts = original_path.split('/')
            path_parts[-2] = WEB_DATA_FOLDER
            web_image_path = '/'.join(path_parts)
            web_image_paths.append(web_image_path)

        # Upload the resized images
        try:
            for web_image_data, web_image_path in zip(web_images_data, web_image_paths):
                self.queue_deferred_upload(web_image_data, web_image_path, 'image/jpeg')
        except Exception as e:
            logging.error(f"Error uploading web-optimized images: {e}")


    #upload metadata to gcs
    async def finalize_and_upload_metadata(self):
        # Sort by datetime, placing None values at the end
        self.paths_times.sort(key=lambda x: (x[1] is None, x[1]))

        # Generate a list of dictionaries with image names and their datetimes
        sorted_metadata = [
            {"image_name": os.path.basename(path), "datetime": dt if dt else None}
            for path, dt in self.paths_times
        ]

        # Create metadata content
        metadata_content = json.dumps(sorted_metadata, indent=4)  # Using indent for better readability

        # Define the path for metadata.json in the "web" folder
        metadata_path = f"{self.session_key}/{WEB_DATA_FOLDER}/metadata.json"
        
        # Upload to GCS using the shared utility function
        await upload_to_gcs(self.source_bucket, metadata_content, metadata_path, content_type='application/json')

        logging.info("Uploaded metadata.json successfully.")

    async def upload_deferred_tasks(self):
        # Check if there are deferred tasks to upload
        if self.deferred_tasks:
            logging.info("Uploading deferred tasks...")
            # Upload all tasks concurrently
            await asyncio.gather(*(upload_to_gcs(self.source_bucket, *task) for task in self.deferred_tasks))
            # Clear the tasks list once all uploads are done
            self.deferred_tasks.clear()
        else:
            logging.info("No deferred tasks to upload.")
    
    def queue_deferred_upload(self, data, destination_path, content_type='image/jpeg')-> None:
        # Queue the task for later upload
        self.deferred_tasks.append((data, destination_path, content_type))
        
    
