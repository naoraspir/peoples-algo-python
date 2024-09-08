
```sh
# for dev run:
docker build -f preprocessing/Dockerfile.dev -t peeps-preprocessing-job-dev .

docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/preprocessing:/app/preprocessing \
  peeps-preprocessing-job-dev

# for prod deployment:
docker build -f preprocessing/Dockerfile -t peeps-preprocess-job .

docker tag peeps-preprocess-job:latest gcr.io/peoples-software/preprocessing-service

docker push gcr.io/peoples-software/preprocessing-service

```