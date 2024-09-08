
```sh
# for dev run:
docker build -f algo_pipeline_executer/Dockerfile.dev -t peeps-algo_pipeline_executer-job-dev .

docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/algo_pipeline_executer:/app/algo_pipeline_executer \
  peeps-algo_pipeline_executer-job-dev

# for prod deployment:
docker build -f algo_pipeline_executer/Dockerfile -t peeps-algo_pipeline_executer-job .

docker tag peeps-algo_pipeline_executer-job:latest gcr.io/peoples-software/algo-pipeline-executer-service:latest

docker push gcr.io/peoples-software/algo-pipeline-executer-service:latest

```