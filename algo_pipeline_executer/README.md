
```sh
# for local run:
docker build -f algo_pipeline_executer/Dockerfile.dev -t peeps-algo_pipeline_executer-job-local .

docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/algo_pipeline_executer:/app/algo_pipeline_executer \
  peeps-algo_pipeline_executer-job-local

# for prod deployment:
# Build the production Docker image for the pipeline executor service
docker build -f algo_pipeline_executer/Dockerfile -t peeps-algo_pipeline_executer-job .

# Tag the image for production
docker tag peeps-algo_pipeline_executer-job:latest gcr.io/peoples-software/algo-pipeline-executer-service:prod

# Push the image to the Google Container Registry
docker push gcr.io/peoples-software/algo-pipeline-executer-service:prod


```