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
image_retriever = PeepsImageRetriever()
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
    start_time = time.time()

    # Convert the uploaded file to a numpy array
    image_stream = await file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    selfie_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Retrieve images where the user is present
    retrieval_result = image_retriever.retrieve_images(session_key, selfie_image)

    elapsed_time = time.time() - start_time
    logger.info(f"Image retrieval took {elapsed_time:.2f} seconds")

    return retrieval_result
# Define a health route for the API
@app.get("/health/")
async def health():
    return {"status": "OK"}
