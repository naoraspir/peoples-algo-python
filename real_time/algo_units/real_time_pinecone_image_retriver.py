import logging
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import time
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone.core.openapi.shared.exceptions import PineconeException
from common.consts_and_utils import PINCONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_SIMILARITY_THRESHOLD


class PeepsImagesRetriever:
    def __init__(self):
        try:
            self.device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
            self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
            self.mtcnn = MTCNN(
                image_size=160, margin=80, min_face_size=85,
                thresholds=[0.6, 0.7, 0.7], factor=0.65, post_process=True,
                device=self.device
            ).eval()

            # Initialize Pinecone client for image retrieval
            self.pinecone_client = Pinecone(api_key=PINCONE_API_KEY)
            self.index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
            logging.info(f"Pinecone client initialized successfully on device {self.device}.")
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise RuntimeError(f"Failed to initialize PeepsImagesRetriever: {e}")

    def process_image(self, selfie_image: np.ndarray):
        """
        Processes the input selfie image to extract the facial embedding.
        Logs the time taken to process the image.
        """
        try:
            start_time = time.time()  # Start timing for processing

            # Convert the image to RGB and resize for the model
            rgb_image = cv2.cvtColor(selfie_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (160, 160))  # Resize for optimization

            with torch.no_grad():
                # Detect and extract face
                face_crop, prob = self.mtcnn(rgb_image, return_prob=True)
                if face_crop is None:
                    error_message = "No face detected. Please try a different image."
                    logging.error(error_message)
                    return {"error": error_message}

                # Embedding extraction
                face_crop = face_crop.unsqueeze(0).to(self.device)
                embedding = self.resnet(face_crop).squeeze().cpu().numpy()

            end_time = time.time()  # End timing
            logging.info(f"Image processed in {end_time - start_time:.4f} seconds.")
            return {"embedding": embedding}

        except cv2.error as e:
            logging.error(f"OpenCV error during image processing: {e}")
            return {"error": f"Failed to process image due to OpenCV error: {e}"}
        except torch.cuda.CudaError as e:
            logging.error(f"CUDA error: {e}")
            return {"error": f"CUDA processing failed: {e}. Check GPU availability."}
        except Exception as e:
            logging.error(f"Unexpected error during image processing: {e}")
            return {"error": f"An unexpected error occurred while processing the image: {e}"}

    def query_similar_images(self, session_key: str, embedding: np.ndarray, similarity_threshold: float = PINECONE_SIMILARITY_THRESHOLD):
        """
        Queries Pinecone for images that are similar to the provided embedding within the given threshold.
        Logs the time taken to retrieve similar images from Pinecone.
        """
        try:
            start_time = time.time()  # Start timing for querying

            # Ensure embedding is a 1D vector (flatten it if necessary)
            if len(embedding.shape) > 1:
                embedding = embedding.squeeze()

            # Convert embedding to list format
            embedding_list = embedding.tolist()

            # Perform the Pinecone query using the correct 'vector' field
            query_result = self.index.query(
                vector=embedding_list,  # Correct field for gRPC
                namespace=session_key,
                top_k=500,  # Set a large K, but we'll filter by threshold
                include_values=False,  # We only need metadata (image paths)
                include_metadata=True  # We need metadata to extract image paths
            )

            end_time = time.time()  # End timing
            logging.info(f"Pinecone query executed in {end_time - start_time:.4f} seconds.")

            # Filter results by similarity score below the threshold and sort them
            similar_images = {
                match['metadata']['image_path']: match['score']
                for match in query_result['matches']
                if match['score'] <= similarity_threshold  # Assuming Euclidean distance is used
            }
            #measure sorting time
            start_time = time.time()  # Start timing for

            # Sort by score in ascending order (lower distance means higher similarity)
            sorted_similar_images = dict(sorted(similar_images.items(), key=lambda item: item[1]))

            end_time = time.time()  # End timing    
            logging.info(f"Sorting executed in {end_time - start_time:.4f} seconds.")

            logging.info(f"Found {len(sorted_similar_images)} similar images below the threshold of {similarity_threshold}.")

            return {"image_paths": sorted_similar_images}

        except PineconeException as e:
            logging.error(f"Pinecone query error: {e}")
            return {"error": f"Failed to query Pinecone index: {e}"}
        except KeyError as e:
            logging.error(f"Key error in Pinecone response: {e}")
            return {"error": f"Unexpected structure in Pinecone query result: {e}"}
        except Exception as e:
            logging.error(f"Unexpected error during Pinecone query: {e}")
            return {"error": f"An unexpected error occurred during Pinecone query: {e}"}

    def retrieve_images(self, session_key: str, selfie_image: np.ndarray):
        """
        Main method to process the input image and retrieve all similar images from Pinecone.
        Logs the time taken for the entire retrieval process.
        """
        total_start_time = time.time()  # Start timing for the total process

        # Step 1: Process the image to extract embedding
        embedding_result = self.process_image(selfie_image)

        if "embedding" in embedding_result:
            embedding = embedding_result["embedding"]
            # Step 2: Query Pinecone for similar images
            result = self.query_similar_images(session_key, embedding)

            total_end_time = time.time()  # End timing for the total process
            logging.info(f"Total time for image retrieval: {total_end_time - total_start_time:.4f} seconds.")

            return result
        else:
            error_message = embedding_result.get("error", "An unknown error occurred")
            logging.error(f"Failed to retrieve images due to embedding error: {error_message}")
            return {"error": error_message}
