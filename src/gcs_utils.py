import logging
import cv2
import numpy as np
from requests import RequestException


logging.basicConfig(level=logging.DEBUG)

def get_image_paths_from_bucket(session_key, storage_client, bucket_name, data_folder) -> list[str]:
        if not session_key or not isinstance(session_key, str):
            raise ValueError("Invalid session key.")
        
        try:
            folder_path = f'{session_key}/{data_folder}'
            blobs = list(storage_client.list_blobs(bucket_name, prefix=folder_path))
            return [blob.name for blob in blobs if blob.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except Exception as e:
            logging.error(f"Error fetching images from bucket: {e}")
            #raise the releveant error for not being able to download
            raise(e)

async def download_image_from_gcs(source_bucket, image_path: str) -> np.array:
        if not image_path or not isinstance(image_path, str):
            raise ValueError("Invalid image path provided.")  

        try:
            blob = source_bucket.blob(image_path)
            img_data = blob.download_as_bytes()
            # Decode the image data from BGR (default in OpenCV) to RGB
            img_bgr = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            logging.error(f"Error downloading image: {e}")
            raise(e)

async def download_images_batch(source_bucket, batch_image_paths):
        """
        Download a batch of images from GCS.
        
        Parameters:
        batch_image_paths (List[str]): A list of image paths to download.
        
        Returns:
        List[np.array]: A list of images.
        """
        images = []
        for image_path in batch_image_paths:
            try:
                img = await download_image_from_gcs(source_bucket, image_path)  # RGB
                images.append(img)
            except RequestException as re:
                logging.error(f"HTTP request failed while downloading image {image_path}: {re}")
            except Exception as e:
                logging.error(f"Error downloading image {image_path}: {e}")
        return images


async def upload_to_gcs(source_bucket, data, destination_path, content_type='image/jpeg'):
        try:
            blob = source_bucket.blob(destination_path)
            blob.upload_from_string(data, content_type=content_type)
        except RequestException as re:
            logging.error(f"HTTP request failed during upload: {re}")
        except Exception as e:
            logging.error(f"Error uploading to GCS: {e}")        

