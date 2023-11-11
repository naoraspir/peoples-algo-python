import os
from fastapi import FastAPI
from algo_units.clustering import FaceClustering
from algo_units.preprocess import PeepsPreProcessor

app = FastAPI()

from pydantic import BaseModel

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/naoraspir/workspace/pepole-cluster/peoples-algo-fastapi-python/google_key.json'

class PreprocessingRequest(BaseModel):
    session_key: str

@app.post("/execute_preprocessing/")
async def execute_preprocessing(request: PreprocessingRequest):
    try:
        # Initialize the PeepsPreProcessor with the provided session key
        preprocessor = PeepsPreProcessor(session_key=request.session_key)
        
        # Execute the preprocessing embeding and uploading intermediate data and images to gcs
        await preprocessor.execute()

        #end of preprocesssing aand start of clustering
        face_clustering = FaceClustering(session_key=request.session_key)

        # Execute the clustering algorithm
        face_clustering.execute()

        return {"status": "success", "message": "Images processed and uploaded successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

#curl -X POST "http://<your_server_url>/execute_preprocessing/" -H "Content-Type: application/json" -d '{"session_key": "your_session_key_here"}'
#curl -X POST "http://localhost:8000/execute_preprocessing/" -H "Content-Type: application/json" -d '{"session_key": "NsuVRhSKg5Ar4Ish7jTW"}'

@app.get("/")
async def root():
    return {"message": "Hello World"}
