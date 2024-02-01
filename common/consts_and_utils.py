# Import necessary libraries
import logging
import cv2
# from deepface import DeepFace
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

# Define constants
BATCH_SIZE = 40
SEMAPHORE_ALLOWED = 10
BUCKET_NAME = 'cdn-album-wedding'  # production
# BUCKET_NAME = 'album-weddings'  # debug
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

# Face detection parameters(preprocess)
CONF_THRESHOLD = 0.75  # Face detection confidence threshold

# HDBSCAN parameters
MIN_CLUSTER_SIZE_HDBSCAN = 6  # Minimum cluster size for HDBSCAN
DISTANCE_METRIC_HDBSCAN = "euclidean"  # Distance metric for HDBSCAN
N_DIST_JOBS_HDBSCAN = -1  # Number of parallel jobs for HDBSCAN

# choose the best face from the detected faces parameters
SHARPNNES_WEIGHT = 0.4 # Sharpness weight for face selection
ALIGNMENT_WEIGHT = 0.3 # Alignment weight for face selection
DISTANCE_WEIGHT = 0.3 # Distance weight for face selection
GLASSES_DEDUCTION_WEIGHT = -0.3 # Glasses deduction weight for face selection
GRAY_SCALE_DEDUCTION_WEIGHT = -0.7 # Gray scale deduction weight for face selection

# web image parameters
MAX_WEB_IMAGE_HEIGHT = 1350  # Maximum height of web images
MAX_WEB_IMAGE_WIDTH = 1200  # Maximum width of web images

# Face Uniter parameters
FACE_UNITER_THRESHOLD = 0.5  # Face uniter similarity threshold
N_NEIGHBORS_FACE_UNITER = 2  # Number of neighbors for face uniter

#save images parameters
DROPOUT_THRESHOLD = 0.97  # Dropout threshold for saving images

# Define utility functions
def is_clear(image, face, laplacian_threshold=80, min_size_ratio=0.00085):
    """
    Check if a cropped face image is clear based on various criteria that are relative to the original image size.

    :param image: The original image.(RGB)
    :param face: The cropped face image.
    :param laplacian_threshold_ratio: Laplacian variance threshold as a ratio relative to original image area.
    :param min_size_ratio: Minimum face size as a ratio relative to original image dimensions.
    :return: True if the face meets the criteria, False otherwise.
    """
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    face_h, face_w = face.shape[:2]
    face_area = face_h * face_w

    min_face_area = img_area * min_size_ratio
    # laplacian_threshold = img_area * laplacian_threshold_ratio

    # Check if the face is large enough relative to the original image
    if face_area < min_face_area:
        return False

    # Check the Laplacian variance relative to the original image
    gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    laplacian_variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    #log the laplacian variance
    #logging.info("laplacian_variance: "+str(laplacian_variance))
    if laplacian_variance < laplacian_threshold:
        return False
    

    return True

def np_array_to_tensor(np_image):
    # Convert np.array image to PIL Image
    pil_image = Image.fromarray(np_image)
    # Apply transformation
    return ToTensor()(pil_image).unsqueeze(0)

def format_image_to_RGB(image_array) -> np.array:#TO RGB
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

def downscale_image(image, scale_percent=25):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def is_grayscale(image, scale_percent=25):
    # Downscale the image to reduce computational load
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return True
    elif image.ndim == 3 and image.shape[2] == 3:
        downscaled_image = downscale_image(image, scale_percent)
        # Check if all channels have the same values
        return np.allclose(downscaled_image[:, :, 0], downscaled_image[:, :, 1]) and np.allclose(downscaled_image[:, :, 1], downscaled_image[:, :, 2])
    return False