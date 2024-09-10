
#clustering

```sh
# for local run:
docker build -f clustering/Dockerfile.dev -t peeps-clustering-job-local .

docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/clustering:/app/clustering \
  peeps-clustering-job-local

---------------------------------------------------------------------------------------------------------

# for prod deployment:
# Build the production Docker image for the clustering service
docker build -f clustering/Dockerfile -t peeps-clustering-job .

# Tag the image for production
docker tag peeps-clustering-job:latest gcr.io/peoples-software/clustering-service:prod

# Push the image to the Google Container Registry
docker push gcr.io/peoples-software/clustering-service:prod

---------------------------------------------------------------------------------------------------------

#for dev deployment:
# Build the development Docker image for the clustering service
docker build -f clustering/Dockerfile -t peeps-clustering-job-dev .

# Tag the image for development
docker tag peeps-clustering-job-dev:latest gcr.io/peoples-software/clustering-service:dev

# Push the image to the Google Container Registry
docker push gcr.io/peoples-software/clustering-service:dev


```