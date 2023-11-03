# Import necessary libraries
import cv2
from deepface import DeepFace


# Define constants
BATCH_SIZE = 50
BUCKET_NAME = 'album-weddings'  # Replace with your actual bucket name  
RAW_DATA_FOLDER = 'raw'
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
    ]
DISTANCE_METRICS = ["cosine", "euclidean", "euclidean_l2"]

# Define utility functions
def is_clear(image, face, laplacian_threshold_ratio=0.0000001, min_size_ratio=0.001):
    """
    Check if a cropped face image is clear based on various criteria that are relative to the original image size.

    :param image: The original image.
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
    laplacian_threshold = img_area * laplacian_threshold_ratio

    # Check if the face is large enough relative to the original image
    if face_area < min_face_area:
        return False

    # Check the Laplacian variance relative to the original image
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    laplacian_variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    if laplacian_variance < laplacian_threshold:
        return False

    return True

def save_to_firestore(data, phase: str, session_key: str):
    db = firestore.Client()

    # Assuming you want to save to a collection named phase
    doc_ref = db.collection(session_key+"_"+phase).document()
    doc_ref.set(data)

# def notify_next_service(data):#TODO 
#     publisher = pubsub_v1.PublisherClient()
#     topic_name = 'projects/{project_id}/topics/{next_topic_id}'

#     # Data must be a bytestring
#     data_str = str(data)
#     data_bytes = data_str.encode('utf-8')

#     publisher.publish(topic_name, data=data_bytes)