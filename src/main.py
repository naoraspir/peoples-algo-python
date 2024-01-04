import gc # Garbage collector
import logging
import os
from fastapi import FastAPI
from algo_units.clustering import FaceClustering
import time

from algo_units.preprocess import PeepsPreProcessor

app = FastAPI()

from pydantic import BaseModel

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/naoraspir/workspace/pepole-cluster/peoples-algo-fastapi-python/google_key.json'

logging.getLogger('numba.core').setLevel(logging.INFO)
# change level of debug for urllib3.connectionpool to INFO
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)


class PreprocessingRequest(BaseModel):
    session_key: str

async def main(request: PreprocessingRequest):
    try:
        #measure time
        startPre = time.time()
        # Initialize the PeepsPreProcessor with the provided session key
        preprocessor = PeepsPreProcessor(session_key=request.session_key)
        
        # # # # Execute the preprocessing embeding and uploading intermediate data and images to gcs
        all_results_preprocess = await preprocessor.execute()

        #measure time
        endPre = time.time()
        preprocess_time = endPre - startPre
        
        # Delete the preprocessor instance and its attributes to free memory
        del preprocessor
        gc.collect()  # Explicitly invoke garbage collection

        #priint some info on the results
        # logging.info("all_results: "+str(all_results))
        logging.info("all_results len: "+str(len(all_results_preprocess)))
        logging.info("all_results[0] len: "+str(len(all_results_preprocess[0])))

        #measure time
        startClust = time.time()

        #end of preprocesssing aand start of clustering
        face_clustering = FaceClustering(session_key=request.session_key, all_results=all_results_preprocess)

        # Execute the clustering algorithm
        face_clustering.execute()
        
        #measure time
        endClust = time.time()
        clustering_time = endClust - startClust

        return {"status": "success", "message": "Images processed and uploaded successfully.\n"+"time elapsed Clustering: "+str(clustering_time)+"\ntime elapsed Pre Process: "+str(preprocess_time)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/execute_preprocessing/")
async def execute_preprocessing(request: PreprocessingRequest):
    response = await main(request)
    return response

#curl -X POST "http://<your_server_url>/execute_preprocessing/" -H "Content-Type: application/json" -d '{"session_key": "your_session_key_here"}'
#curl -X POST "http://localhost:8000/execute_preprocessing/" -H "Content-Type: application/json" -d '{"session_key": "NsuVRhSKg5Ar4Ish7jTW"}'
#curl -X POST "http://localhost:8000/execute_preprocessing/" -H "Content-Type: application/json" -d '{"session_key": "test0"}'

@app.get("/")
async def root():
    return {"message": "Hello World"}
