#helper functions
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from common.consts_and_utils import SHARPNNES_WEIGHT, ALIGNMENT_WEIGHT, DISTANCE_WEIGHT, DETECTION_WEIGHT, POSITION_WEIGHT, FACE_DISTANCE_WEIGHT, GLASSES_DEDUCTION_WEIGHT, GRAY_SCALE_DEDUCTION_WEIGHT, FACE_RATIO_WEIGHT, FACE_COUNT_WEIGHT, is_grayscale
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
        return False #assume no glasses to avoid false positives

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
    

def parallel_landmarks_and_glasses(face):
    landmark_list = face_recognition.face_landmarks(face, model="large")
    glasses_deduction = 1.0 if landmark_list and detect_glasses(face, landmark_list[0]) else 0.0
    return landmark_list, glasses_deduction

def parallel_grayscale(face):
    is_gray = is_grayscale(face)
    return 1.0 if is_gray else 0.0

def process_faces_parallel(cluster_id, faces, sharpness_scores, alignment_scores, detection_scores, position_scores,
                        face_distance_scores, distance_scores, face_ratio_scores, metrics, embeddings, centroid):
    
    all_scores = []
    best_score = -1
    best_face = None
    best_embedding = None
    best_face_landmarks = {}
    faces_with_glasses = []
    
    best_face_sharpness, best_face_alignment, best_face_distance, best_face_glasses, best_face_gray_scale = 0, 0, 0, 0, 0
    best_face_index = -1

    # Run landmark detection and glasses detection in parallel
    with ProcessPoolExecutor() as executor:
        futures_landmarks_glasses = {executor.submit(parallel_landmarks_and_glasses, face): idx for idx, face in enumerate(faces)}
        futures_grayscale = {executor.submit(parallel_grayscale, face): idx for idx, face in enumerate(faces)}

        landmarks_glasses_results = {futures_landmarks_glasses[future]: future.result() for future in as_completed(futures_landmarks_glasses)}
        grayscale_results = {futures_grayscale[future]: future.result() for future in as_completed(futures_grayscale)}

    for idx, (face, embedding, metric) in enumerate(zip(faces, embeddings, metrics)):
        landmark_list, glasses_deduction = landmarks_glasses_results[idx]
        grayscale_deduction = grayscale_results[idx]
        
        if glasses_deduction > 0:
            faces_with_glasses.append(face)  # Save face with glasses for debugging

        # Individual scores
        sharpness_score = sharpness_scores[idx]
        alignment_score = alignment_scores[idx]
        detection_prob = detection_scores[idx]
        position_score = position_scores[idx]
        face_distance_score = face_distance_scores[idx]
        distance_score = 1 - distance_scores[idx]  # Inverted for similarity
        face_ratio_score = face_ratio_scores[idx]
        face_count_deduction = 1 / metric['faces_count']  # Less faces is better, precompute

        # Composite score calculation
        score = (
            SHARPNNES_WEIGHT * sharpness_score +
            ALIGNMENT_WEIGHT * alignment_score +
            DISTANCE_WEIGHT * distance_score +
            DETECTION_WEIGHT * detection_prob +
            POSITION_WEIGHT * position_score +
            FACE_DISTANCE_WEIGHT * face_distance_score +
            GLASSES_DEDUCTION_WEIGHT * glasses_deduction +
            GRAY_SCALE_DEDUCTION_WEIGHT * grayscale_deduction +
            FACE_RATIO_WEIGHT * face_ratio_score +
            FACE_COUNT_WEIGHT * face_count_deduction
        )

        all_scores.append((face, score))

        # Update the best face if the current one has a higher score
        if score > best_score:
            best_score = score
            best_face = face
            best_embedding = embedding
            best_face_landmarks = landmark_list[0] if landmark_list else {}
            best_face_index = idx
    
    # Fallback: if no best face found, choose the medoid face
    if best_face is None:
        logging.warning("No best face found based on scores, choosing medoid face instead.")
        medoid_index = np.argmin(distance_scores)
        best_face = faces[medoid_index]
        best_embedding = embeddings[medoid_index]
    else:
        logging. info(f"Best face found for cluster {cluster_id}: Sharpness={best_face_sharpness}, Alignment={best_face_alignment}, Distance={best_face_distance},\n")
        logging. info(f"Glasses Deduction= {best_face_glasses}, Gray Scale Deduction = {best_face_gray_scale} , Score={best_score}")
        #save the best face with landmarks to gcs
        # self.save_best_face_landmarks_to_gcs(best_face, best_face_landmarks, cluster_id)  
    return all_scores, best_face, best_embedding, faces_with_glasses, best_face_index
