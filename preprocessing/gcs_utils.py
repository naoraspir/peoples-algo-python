import logging
from requests import RequestException
import asyncio
from PIL import Image, ExifTags
import io


logging.basicConfig(level=logging.INFO)

def get_image_paths_from_bucket(session_key, storage_client, bucket_name, data_folder) -> list:
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

async def download_image_from_gcs(source_bucket, image_path: str) -> tuple:
    if not image_path or not isinstance(image_path, str):
        raise ValueError("Invalid image path provided.")  

    try:
        blob = source_bucket.blob(image_path)
        img_data = blob.download_as_bytes()
        img_pil = Image.open(io.BytesIO(img_data))  # Open directly as a PIL image
        logging.info("After download from gcs (PIL format)")

        # Try to extract EXIF datetime
        exif_data = img_pil._getexif()
        datetime_taken = None
        if exif_data:
            for tag, value in exif_data.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'DateTimeOriginal':
                    datetime_taken = value
                    break

        # No need to convert to BGR and then back to RGB
        # Return the PIL image directly
        return img_pil, image_path, datetime_taken
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        raise(e)

async def download_images_batch(source_bucket, batch_image_paths):
    """
    Download a batch of images from GCS.
    
    Parameters:
    batch_image_paths (List[str]): A list of image paths to download.
    
    Returns:
    List[tuple]: A list of tuples containing PIL images, paths, and datetime taken.
    """
    images_and_paths_pairs = []
    download_tasks = [download_image_from_gcs(source_bucket, image_path) for image_path in batch_image_paths]

    for future in asyncio.as_completed(download_tasks):
        try:
            img, path, datetime_taken = await future
            images_and_paths_pairs.append((img, path, datetime_taken))
        except RequestException as re:
            logging.error(f"HTTP request failed while downloading image: {re}")
        except Exception as e:
            logging.error(f"Error downloading image: {e}")

    return images_and_paths_pairs

async def upload_to_gcs(source_bucket, data, destination_path, content_type='image/jpeg'):
    try:
        blob = source_bucket.blob(destination_path)
        blob.upload_from_string(data, content_type=content_type)
    except RequestException as re:
        logging.error(f"HTTP request failed during upload: {re}")
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")        


async def upload_with_retry(data, destination_path, content_type='image/jpeg', max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            await upload_to_gcs(data, destination_path, content_type)
            break  # If upload succeeds, break out of the loop
        except RequestException as e:
            logging.error(f"Upload attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)  # Wait for a short period before retrying
            else:
                logging.error("Max retries reached. Upload failed.")
                raise
