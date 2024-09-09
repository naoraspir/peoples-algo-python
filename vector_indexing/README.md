
```sh
# for local run:
docker build -f vector_indexing/Dockerfile.dev -t peeps-indexing-job-local .

docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/vector_indexing:/app/vector_indexing \
  peeps-indexing-job-local

# for prod deployment:
# Build the production Docker image for the indexing service
docker build -f vector_indexing/Dockerfile -t peeps-indexing-job .

# Tag the image for production
docker tag peeps-indexing-job:latest gcr.io/peoples-software/indexing-service:prod

# Push the image to the Google Container Registry
docker push gcr.io/peoples-software/indexing-service:prod

#for dev deployment:
# Build the development Docker image for the indexing service
docker build -f vector_indexing/Dockerfile -t peeps-indexing-job-dev .

# Tag the image for development
docker tag peeps-indexing-job-dev:latest gcr.io/peoples-software/indexing-service:dev

# Push the image to the Google Container Registry
docker push gcr.io/peoples-software/indexing-service:dev


```