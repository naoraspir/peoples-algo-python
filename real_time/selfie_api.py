import logging
import cv2
from fastapi import FastAPI, File, Form, UploadFile
import numpy as np
import time

from algo_units.real_time_image_retriver import PeepsClusterRetriever

# Set up logging and FastAPI app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
app = FastAPI()

# Load environment variables from .env file (useful for local development)
from dotenv import load_dotenv
load_dotenv()

retriever = PeepsClusterRetriever()

@app.post("/process-image/")
async def process_image(session_key: str = Form(...), file: UploadFile = File(...), k:int=4):
    start_time = time.time()  # Start time measurement

    # Convert the uploaded file to a numpy array
    image_stream = await file.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    selfie_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    retriever.load_new_image(session_key, selfie_image)
    embedding_result = retriever.process_image()

    if "embedding" in embedding_result:
        top_k_candidates = retriever.retrieve_top_k_candidates(embedding_result["embedding"],k=k)
        elapsed_time = time.time() - start_time  # Measure elapsed time
        logger.info(f"Processing and retrieval took {elapsed_time:.2f} seconds")
        return {"top_k_candidates": top_k_candidates}
    else:
        return {"error": embedding_result.get("error", "An unknown error occurred")}
