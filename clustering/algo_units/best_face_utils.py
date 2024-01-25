#helper functions
import logging
import cv2
import numpy as np
import face_recognition


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
# Define a function to calculate image sharpness
def calculate_sharpness(image: np.ndarray) -> float:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the Laplacian
    laplacian_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    # Calculate the variance of the Laplacian which is a measure of sharpness
    sharpness_score = laplacian_variance
    return sharpness_score

def detect_glasses(face: np.ndarray, landmarks) -> bool:
    try:
        # Calculate a larger region including the area between eyebrows and upper nose
        eyebrow_region_start = np.min(landmarks['left_eyebrow'], axis=0)
        eyebrow_region_end = np.max(landmarks['right_eyebrow'], axis=0)
        nose_region_end = np.max(landmarks['nose_bridge'], axis=0)

        # Expanded region to include part of the nose
        region = face[eyebrow_region_start[1]:nose_region_end[1], eyebrow_region_start[0]:eyebrow_region_end[0]]

        # Apply Gaussian Blur and Canny Filter
        img_blur = cv2.GaussianBlur(region, (5, 5), sigmaX=1.7, sigmaY=1.7)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        # Analyze the central vertical strip of the edge-detected region
        edges_center = edges.T[int(len(edges.T) / 2)]
        
        # Check for the presence of white edges (value 255) indicating glasses
        return 255 in edges_center
    except Exception as e:
        logging.error("Error detecting glasses: %s", e)
        return False#assume no glasses to avoid false positives

# Define a function to evaluate face alignment based on landmarks
def evaluate_face_alignment(face: np.ndarray) -> float:
    try:
        # Assuming face_landmarks function is defined as you provided
        landmarks_list = face_recognition.face_landmarks(face, model="large")

        if not landmarks_list:
            return 0  # No face detected or no landmarks detected

        # For simplicity, take the first face detected
        landmarks = landmarks_list[0]

        # Convert landmarks to NumPy arrays for calculations
        left_eye_center = np.mean(np.array(landmarks['left_eye']), axis=0)
        right_eye_center = np.mean(np.array(landmarks['right_eye']), axis=0)
        eye_level = abs(left_eye_center[1] - right_eye_center[1])
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        eye_level_score = max(0, 1 - (eye_level / eye_distance))

        left_eyebrow_center = np.mean(np.array(landmarks['left_eyebrow']), axis=0)
        right_eyebrow_center = np.mean(np.array(landmarks['right_eyebrow']), axis=0)
        eyebrow_level = abs(left_eyebrow_center[1] - right_eyebrow_center[1])
        inter_eyebrow_distance = np.linalg.norm(left_eyebrow_center - right_eyebrow_center)
        eyebrow_level_score = max(0, 1 - (eyebrow_level / inter_eyebrow_distance))

        nose_bridge_center = np.mean(np.array(landmarks['nose_bridge']), axis=0)
        nose_tip_center = np.mean(np.array(landmarks['nose_tip']), axis=0)
        nose_centering = abs(nose_bridge_center[0] - nose_tip_center[0])
        nose_centering_score = max(0, 1 - (nose_centering / inter_eyebrow_distance))

        mouth_left_corner = np.array(landmarks['top_lip'][0])
        mouth_right_corner = np.array(landmarks['top_lip'][6])
        mouth_level = abs(mouth_left_corner[1] - mouth_right_corner[1])
        mouth_width = np.linalg.norm(mouth_left_corner - mouth_right_corner)
        mouth_level_score = max(0, 1 - (mouth_level / mouth_width))

        # Calculate symmetry scores
        nose_center_x = np.mean(np.array(landmarks['nose_bridge']), axis=0)[0]
        
        # Symmetry for eyes
        eye_symmetry_score = 1 - abs((left_eye_center[0] - nose_center_x) - (nose_center_x - right_eye_center[0])) / eye_distance

        # Symmetry for eyebrows
        eyebrow_symmetry_score = 1 - abs((left_eyebrow_center[0] - nose_center_x) - (nose_center_x - right_eyebrow_center[0])) / inter_eyebrow_distance

        # Combining symmetry scores (50% weight) and other alignment scores (50% weight)
        symmetry_score = (eye_symmetry_score + eyebrow_symmetry_score) / 2
        alignment_score = (eye_level_score + eyebrow_level_score + nose_centering_score + mouth_level_score) / 4
        total_score = 0.5 * symmetry_score + 0.5 * alignment_score

        return total_score
    except Exception as e:
        logging.error("Error evaluating face alignment: %s", e)
        return 0