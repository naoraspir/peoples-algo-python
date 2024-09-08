
```sh
# for dev run:
docker build -f vector_indexing/Dockerfile.dev -t peeps-indexing-job-dev .

docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/vector_indexing:/app/vector_indexing \
  peeps-indexing-job-dev

# for prod deployment:
docker build -f vector_indexing/Dockerfile -t peeps-indexing-job .

docker tag peeps-indexing-job:latest gcr.io/peoples-software/indexing-service:latest

docker push gcr.io/peoples-software/indexing-service:latest

```