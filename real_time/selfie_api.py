import logging
import cv2
from fastapi import FastAPI, File, Form, UploadFile
import numpy as np
import time

from real_time.algo_units.real_time_cluster_retriver import PeepsClusterRetriever
from real_time.algo_units.real_time_pinecone_image_retriver import PeepsImagesRetriever

# Set up logging and FastAPI app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info("Starting FastAPI server")

app = FastAPI()
logger.info("FastAPI app initialized")

# Load environment variables from .env file (useful for local development)
from dotenv import load_dotenv
load_dotenv()

# Log initialization and environment variables
logger.info("Environment variables loaded")
logger.info(f"loading retrievers")
cluster_retriever = PeepsClusterRetriever()  # Using the same retriever class for both operations
image_retriever = PeepsImagesRetriever()
logger.info(f"retrievers loaded")

# Define the API endpoint for processing an image (existing route)
@app.post("/process-image/")
async def process_image(session_key: str = Form(...), file: UploadFile = File(...), k: int = 4):
    start_time = time.time()  # Start time measurement

    # Convert the uploaded file to a numpy array
    image_stream = await file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    selfie_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image and extract the embedding
    embedding_result = cluster_retriever.process_image(selfie_image)

    if "embedding" in embedding_result:
        top_k_candidates = cluster_retriever.retrieve_top_k_candidates(session_key, embedding_result["embedding"], k=k)
        elapsed_time = time.time() - start_time  # Measure elapsed time
        logger.info(f"Processing and retrieval took {elapsed_time:.2f} seconds")
        return {"top_k_candidates": top_k_candidates}
    else:
        return {"error": embedding_result.get("error", "An unknown error occurred")}

@app.post("/retrieve-images/")
async def retrieve_images(session_key: str = Form(...), file: UploadFile = File(...)):
    """
    API endpoint to process an uploaded selfie image and retrieve similar images.
    Returns detailed information including processing, retrieval, and re-ranking times.
    """
    start_time = time.time()  # Start total timing

    try:
        # Step 1: Convert the uploaded file to a numpy array
        image_stream = await file.read()
        nparr = np.frombuffer(image_stream, np.uint8)
        selfie_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if selfie_image is None:
            return {
                "status": "error",
                "message": "Uploaded image is not valid or corrupted.",
                "code": 400
            }

        # Step 2: Process the image to extract the embedding
        processing_start_time = time.time()  # Start timing for processing
        embedding_result = image_retriever.process_image(selfie_image)
        processing_end_time = time.time()  # End timing for processing

        if "error" in embedding_result:
            return {
                "status": "error",
                "message": embedding_result["error"],
                "code": 500
            }
        # logging.info(f"embedding_result: {embedding_result}")
        embedding = embedding_result["embedding"]

        # Step 3: Query Pinecone for similar images
        retrieval_start_time = time.time()  # Start timing for retrieval
        retrieval_result = image_retriever.query_similar_images(session_key, embedding)
        retrieval_end_time = time.time()  # End timing for retrieval

        if "error" in retrieval_result:
            return {
                "status": "error",
                "message": retrieval_result["error"],
                "code": 500
            }
      
        # Step 4: Re-rank the images using your custom re-ranking system
        rerank_start_time = time.time()  # Start timing for re-ranking

        # embeddings = [embedding] * len(retrieval_result["image_paths"])  # Assuming same query embedding
        metrics = [image["metrics"] for image in retrieval_result]  # Use Pinecone metrics
        sorting_scores = image_retriever.compute_image_sorting_score(metrics)
        
        # Combine and sort the results based on sorting scores
        sorted_images = sorted(zip(retrieval_result, sorting_scores), key=lambda x: x[1], reverse=True)
        
        rerank_end_time = time.time()  # End timing for re-ranking

        # Step 5: Calculate elapsed times
        total_elapsed_time = time.time() - start_time  # Total time taken
        processing_time = processing_end_time - processing_start_time  # Time for image processing
        retrieval_time = retrieval_end_time - retrieval_start_time  # Time for querying Pinecone
        rerank_time = rerank_end_time - rerank_start_time  # Time for re-ranking

        # Step 6: Return structured JSON response with times and result
        return {
            "status": "success",
            "message": "Images retrieved and re-ranked successfully",
            "data": {
                "session_key": session_key,
                "image_paths": [image[0]["image_path"] for image in sorted_images],  # Sorted image paths
                "processing_time_seconds": round(processing_time, 2),
                "retrieval_time_seconds": round(retrieval_time, 2),
                "reranking_time_seconds": round(rerank_time, 2),  # Add re-ranking time to response
                "total_time_seconds": round(total_elapsed_time, 2)
            },
            "code": 200
        }

    except Exception as e:
        logging.error(f"Error in /retrieve-images/ route: {e}")
        return {
            "status": "error",
            "message": f"An unexpected error occurred: {e}",
            "code": 500
        }


# Define a health route for the API
@app.get("/health/")
async def health():
    return {"status": "OK"}
