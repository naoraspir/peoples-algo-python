
```sh
# for local run:
docker build -f preprocessing/Dockerfile.dev -t peeps-preprocessing-job-local .

docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/preprocessing:/app/preprocessing \
  peeps-preprocessing-job-local

# for prod deployment:
# Build the production Docker image for the preprocessing service
docker build -f preprocessing/Dockerfile -t peeps-preprocess-job .

# Tag the image for production
docker tag peeps-preprocess-job:latest gcr.io/peoples-software/preprocessing-service:prod

# Push the image to the Google Container Registry
docker push gcr.io/peoples-software/preprocessing-service:prod

#for dev deployment:
# Build the development Docker image for the preprocessing service
docker build -f preprocessing/Dockerfile -t peeps-preprocess-job-dev .

# Tag the image for development
docker tag peeps-preprocess-job-dev:latest gcr.io/peoples-software/preprocessing-service:dev

# Push the image to the Google Container Registry
docker push gcr.io/peoples-software/preprocessing-service:dev


```