# Import necessary libraries
import logging
import cv2
# from deepface import DeepFace
from PIL import Image
import numpy as np

# Define constants
MAX_WORKERS = 4 # Maximum number of workers for parallel processing
BATCH_SIZE = 20 # Batch size for face detection
MICRO_BATCH_SIZE = 10 # Micro batch size for face detection
SEMAPHORE_ALLOWED = 10
BUCKET_NAME = 'cdn-album-wedding'  # production
# BUCKET_NAME = 'album-weddings'  # debug
# RAW_DATA_FOLDER = 'raw'
RAW_DATA_FOLDER = 'raw'
WEB_DATA_FOLDER = 'web'
PREPROCESS_FOLDER = 'preprocess'
MODELS = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
DETECTORS= [
    'opencv', 
    'ssd',
    'mtcnn', 
    'dlib', 
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    "skip"
    ]
DISTANCE_METRICS = ["cosine", "euclidean", "euclidean_l2"]

#resize constants
SCALE_PRECENT = 25 # Scale percent for downscaling images

# Face detection parameters(preprocess)
CONF_THRESHOLD = 0.75 # Face detection confidence threshold

#UMAP parameters
N_COMPONENTS_UMAP = 300  # Number of components for UMAP
N_NEIGHBORS_UMAP = 20  # Number of neighbors for UMAP
MIN_DIST_UMAP = 0.1  # Minimum distance for UMAP
METRIC_UMAP = "euclidean"  # Metric for UMAP
N_EPOCHS_UMAP = 500  # Number of epochs for UMAP
LEARNING_RATE_UMAP = 1.0  # Learning rate for UMAP

# HDBSCAN parameters
MIN_CLUSTER_SIZE_HDBSCAN = 3  # Minimum cluster size for HDBSCAN
DISTANCE_METRIC_HDBSCAN = "euclidean"  # Distance metric for HDBSCAN
N_DIST_JOBS_HDBSCAN = -1  # Number of parallel jobs for HDBSCAN
MIN_CLUSTER_SAMPLES_HDBSCAN = 5  # Minimum number of samples in a cluster for HDBSCAN  
CLUSTER_SELECTION_EPSILON = 0.65 # Epsilon for selecting the best cluster from HDBSCAN 

# choose the best face from the detected faces parameters
SHARPNNES_WEIGHT = 0.35# Sharpness weight for face selection
ALIGNMENT_WEIGHT = 0.45# Alignment weight for face selection
DISTANCE_WEIGHT = 0.25 # Distance weight for face selection distance from centroid
GLASSES_DEDUCTION_WEIGHT = -0.3 # Glasses deduction weight for face selection
GRAY_SCALE_DEDUCTION_WEIGHT = -0.7 # Gray scale deduction weight for face selection
DROPOUT_THRESHOLD = 0.80  # Dropout threshold for saving images
DETECTION_WEIGHT = 0.2 # Detection weight for face selection
POSITION_WEIGHT = 0 # Position weight for face selection
FACE_DISTANCE_WEIGHT = 0 # Face distance weight for face selection distance from CAMERRA
FACE_RATIO_WEIGHT = 0 # score for face ratio in comparison to origin image
FACE_COUNT_WEIGHT = 0.45 # Face count weight for face selection less faces is better

# sorting faces parameters
FACE_COUNT_WEIGHT_SORTING = 0.65#0.35# Face count weight for sorting less faces is better
DISTANCE_WEIGHT_SORTING = 0.3# Distance weight for sorting distance from centroid
FACE_DISTANCE_WEIGHT_SORTING = 0# Face distance weight for sorting distance from CAMERRA
FACE_SHARPNESS_WEIGHT_SORTING = 0.35# Face sharpness weight for sorting
IMAGE_SHARPNESS_WEIGHT_SORTING = 0.0# Image sharpness weight for sorting
DETECTION_WEIGHT_SORTING = 0.2# Detection prob weight for sorting
POSITION_WEIGHT_SORTING = 0.0# Position relative to center of image weight for sorting
ALIGNMENT_WEIGHT_SORTING = 0.3# face alignment weight for sorting
FACE_RATIO_WEIGHT_SORTING = 0# score for face ratio in comparison to origin image

# web image parameters
MAX_WEB_IMAGE_HEIGHT = 1350  # Maximum height of web images
MAX_WEB_IMAGE_WIDTH = 1200  # Maximum width of web images

# Face Uniter parameters
FACE_UNITER_THRESHOLD = 0.5  # Face uniter similarity threshold
N_NEIGHBORS_FACE_UNITER = 6  # Number of neighbors for face uniter

# face indexer parameters
PINCONE_API_KEY = '5c6b8bec-2199-47b1-aaff-473485abee08' #pinecone api key 
PINCONE_ENNVIROMENT = 'us-central1' #pinecone environment
PINECONE_DEFAULT_EMBBEDING_DIM = 512 #embedding dimension
PINECONE_DISTANCE_METRIC =  'euclidean'  # Distance metric for pinecone index
PINECONE_INDEX_NAME = 'peeps0' #pinecone index name
PINECONE_UPSERT_BATCH_SIZE = 1000 #pinecone upsert batch size
PINECONE_SIMILARITY_THRESHOLD = 0.85 #pinecone similarity threshold

# Define utility functions
def is_clear(image, face, laplacian_threshold=60, min_size_ratio=0.00085):
    """
    Check if a cropped face image is clear based on various criteria that are relative to the original image size.

    :param image: The original image.(RGB)
    :param face: The cropped face image.
    :param laplacian_threshold_ratio: Laplacian variance threshold as a ratio relative to original image area.
    :param min_size_ratio: Minimum face size as a ratio relative to original image dimensions.
    :return: True if the face meets the criteria, False otherwise.
    """
   
    img_area = get_image_area(image)
    face_area = get_image_area(face)

    min_face_area = img_area * min_size_ratio
    # laplacian_threshold = img_area * laplacian_threshold_ratio

    # Check if the face is large enough relative to the original image
    if face_area < min_face_area:
        return False

    # Check the Laplacian variance relative to the original image
    laplacian_variance = get_laplacian_variance(face)
    #log the laplacian variance
    #logging.info("laplacian_variance: "+str(laplacian_variance))
    if laplacian_variance < laplacian_threshold:
        return False
    

    return True

#get laplacian variance for an image
def get_laplacian_variance(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def get_image_area(image):
    return image.shape[0] * image.shape[1]

def is_grayscale(image, scale_percent=25):
    # Downscale the image to reduce computational load
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return True
    elif image.ndim == 3 and image.shape[2] == 3:
        downscaled_image = downscale_image(image, scale_percent)
        # Check if all channels have the same values
        return np.allclose(downscaled_image[:, :, 0], downscaled_image[:, :, 1]) and np.allclose(downscaled_image[:, :, 1], downscaled_image[:, :, 2])
    return False

def downscale_image(image, scale_percent):
    """
    Downscale an image while maintaining its aspect ratio.
    """
    # Calculate new dimensions
    new_width = int(image.shape[1] * scale_percent)
    new_height = int(image.shape[0] * scale_percent)
    
    # Resize the image using high-quality resampling
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return resized_image

# Helper function to dynamically determine the optimal scale
def determine_optimal_scale(image):
    #RGP PIL image
    original_height, original_width = image.shape[:2]
    max_dimension = max(original_width, original_height)
    if max_dimension > 1600:
        scale_percent = 1600 / max_dimension  # Scale down to a maximum of 1600px in any dimension
    elif max_dimension < 640:
        scale_percent = 640 / max_dimension  # Scale up to a minimum of 640px in any dimension
    else:
        scale_percent = 1.0  # Keep the original size if it's within the optimal range
    return scale_percent

def determine_optimal_scale_pil(image):
    """
    Determine the optimal scale for a PIL image based on its size.
    """
    original_width, original_height = image.size
    max_dimension = max(original_width, original_height)
    
    if max_dimension > 1600:
        scale_percent = 1600 / max_dimension  # Scale down to a maximum of 1600px in any dimension
    elif max_dimension < 640:
        scale_percent = 640 / max_dimension  # Scale up to a minimum of 640px in any dimension
    else:
        scale_percent = 1.0  # Keep the original size if it's within the optimal range
    
    return scale_percent

def downscale_image_pil(image, scale_percent):
    """
    Downscale a PIL image while maintaining its aspect ratio.
    """
    # Calculate new dimensions
    new_width = int(image.width * scale_percent)
    new_height = int(image.height * scale_percent)
    
    # Resize the image using high-quality resampling
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

# Example usage
def process_images_with_pil(images):
    """
    Process a list of PIL images by downscaling them based on optimal scale.
    """
    downscaled_images = []
    for img in images:
        # Determine optimal scale
        scale_percent = determine_optimal_scale_pil(img)
        
        # Downscale the image
        downscaled_image = downscale_image_pil(img, scale_percent)
        
        # Add the downscaled image to the list
        downscaled_images.append(downscaled_image)
    
    return downscaled_images

def normalize_scores(scores: np.ndarray, is_distance_score: bool = False) -> np.ndarray:
    try:
        if is_distance_score:
            scores = np.where(np.isnan(scores) | np.isinf(scores), np.finfo(scores.dtype).max, scores)
        else:
            scores = np.nan_to_num(scores)  # Convert nan to 0 for non-distance scores

        min_score = np.min(scores)
        max_score = np.max(scores)
        return (scores - min_score) / (max_score - min_score) if max_score != min_score else np.zeros_like(scores)
    except Exception as e:
        logging.error("Error normalizing scores: %s", e)
        raise e