
```sh
# for dev run:
docker run -it \
  -e SESSION_KEY=test1 \
  -v $(pwd)/google_key.json:/app/google_key.json \
  -v $(pwd)/common:/app/common \
  -v $(pwd)/clustering:/app/clustering \
  peeps-clustering-job-dev


```