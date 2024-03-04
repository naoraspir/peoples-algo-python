import numpy as np

class PeepsMetricCalculator:
    def __init__(self):
        # Initialize any required variables or configurations here
        pass

    def calculate_face_metrics(self, box, image_shape, prob, faces_count, laplacian_variance_image, laplacian_variance_face, face_landmarks, face_idx):
        x, y, w, h = box.tolist()
        x, y, w, h = map(int, [x, y, w, h])

        # Calculate individual metrics
        face_alignment_score = self.calculate_face_alignment((x, y, w, h), face_landmarks)
        face_distance_score = self.calculate_face_distance_score(face_idx, faces_count)
        face_position_score = self.calculate_face_position_score((x, y, w, h), image_shape)
        face_tagging_position = self.calculate_tag_position((x, y, w, h), face_landmarks, image_shape)

        return {
            'face_alignment_score': float(face_alignment_score),
            'face_distance_score': float(face_distance_score),
            'face_detection_prob': float(prob),
            'faces_count': faces_count,
            'face_position_score': float(face_position_score),
            'laplacian_variance_image': float(laplacian_variance_image),
            'laplacian_variance_face': float(laplacian_variance_face),
            'tag_position': face_tagging_position,
        }

    def calculate_face_alignment(self, face_box, face_landmarks):
        # Extract landmark points
        left_eye, right_eye, nose = face_landmarks[0], face_landmarks[1], face_landmarks[2]
        face_left = face_box[0]
        face_right = face_box[2]

        # Calculate eye midpoint
        eye_midpoint = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

        # Symmetry check
        left_distance = abs(eye_midpoint[0] - face_left)
        right_distance = abs(face_right - eye_midpoint[0])
        symmetry_score = 1 - abs(left_distance - right_distance) / max(left_distance, right_distance)

        # Nose alignment
        nose_alignment = 1 - abs(nose[0] - eye_midpoint[0]) / (face_right - face_left)

        # Eye level check
        eye_level = 1 - abs(left_eye[1] - right_eye[1]) / (face_right - face_left)

        # Aggregate score
        score = (symmetry_score + nose_alignment + eye_level) / 3

        return score

    def calculate_face_distance_score(self, face_index, total_faces):
        """
        Calculate the face distance score based on the index of the face.

        Args:
            face_index (int): The index of the face in the sorted list from MTCNN.
            total_faces (int): Total number of faces detected in the image.

        Returns:
            float: A score representing the face's prominence in the image.
        """
        if total_faces <= 1:
            return 1  # If there's only one face, give it the highest score
        else:
            # Normalize the index (invert it) and then scale to a 0-1 range
            return (total_faces - face_index - 1) / (total_faces - 1)

    def calculate_face_position_score(self, face_box, image_shape):
        """
        Calculate a score based on how centered the face is within the image.

        Args:
            face_box (tuple): The bounding box of the face (x, y, w, h).
            image_shape (tuple): The dimensions of the image (height, width, channels).

        Returns:
            float: A score representing how centered the face is within the image.
        """
        img_height, img_width, _ = image_shape
        x, y, w, h = face_box

        # Calculate center of the face
        face_center_x = x + w / 2
        face_center_y = y + h / 2

        # Calculate center of the image
        img_center_x = img_width / 2
        img_center_y = img_height / 2

        # Calculate the distance of face center from image center
        distance_from_center = np.sqrt((face_center_x - img_center_x) ** 2 + (face_center_y - img_center_y) ** 2)

        # Normalize the distance: 0 means face is at the center, higher values mean face is further away
        max_distance = np.sqrt((img_width / 2) ** 2 + (img_height / 2) ** 2)
        normalized_distance = distance_from_center / max_distance

        # Convert to a score: 1 for a face at the center, decreasing towards 0 for faces at the edges
        position_score = 1 - normalized_distance

        return position_score

    def calculate_tag_position(self, face_box, face_landmarks, image_shape):
        """
        Calculate the position for a tag, typically around the chin of the person.

        Args:
            face_box (tuple): The bounding box of the face (x, y, w, h).
            landmarks (np.array): Facial landmarks detected for the face.
            image_shape (tuple): The dimensions of the image (height, width, channels).

        Returns:
            tuple: A tuple (x, y) representing the position for the tag.
        """
        x, y, w, h = face_box

        # Assuming the landmarks array contains points in the order [eyes, nose, mouth, chin]
        # The chin point would be the last point in the landmarks array
        chin_point = face_landmarks[-1]

        # Adjust the tag position to be slightly below the chin
        tag_x = chin_point[0]
        tag_y = chin_point[1] + 0.05 * h  # Adjust this value to position the tag lower or higher

        # Ensure the tag position is within the image bounds
        img_height, img_width, _ = image_shape
        tag_x = max(0, min(tag_x, img_width))
        tag_y = max(0, min(tag_y, img_height))

        return (int(tag_x), int(tag_y))


# Example usage:
# calculator = PeepsMetricCalculator()
# metrics = calculator.calculate_face_metrics(face, box, image_shape, prob, faces_count, laplacian_variance_image, laplacian_variance_face, face_landmarks, face_idx)