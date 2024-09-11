
```sh
# for testing api:

curl --location 'https://selfie-service-dev-34y6ttkera-ue.a.run.app/retrieve-images/' \
--header 'Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImQ3YjkzOTc3MWE3ODAwYzQxM2Y5MDA1MTAxMmQ5NzU5ODE5MTZkNzEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTEyNDc3OTMzNTkzMjIxNjc3NDAwIiwiZW1haWwiOiJuYW9yYXNwaXJAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiJIZk9oR3lGclp0VHdOamVmbFVXYThBIiwiaWF0IjoxNzI2MDUwNjAwLCJleHAiOjE3MjYwNTQyMDB9.iGiOwU4IUzxGzoveX_pEezbPvhVXRtj003Ef7HclELwxmKjO8xtH2s211bpyJxYMvVwAsyVlwWDjDUYRL94gGpSixg8Rw795cLPfbi9wVnzwOFmJsR5YmBZ3N_07LbVnuKYvd4vNVV8odauk_9Ucy-5pTP62qDiEYUNWuKAVleOqrtuc0bROxAuzzviEo2ImWzjvrYTzmfrfrs1ay-LtHueIvlUZr9pn2oMS6lbWSN5Vqj5x9hbpiwlTiI81OnQqzo6738RV3VNm_zJZHlz50W3Gxp75yxDweMPityf8xPFIzENU3XatkEdjktwEXHfMZfSMlfplsTtqcS0MNoFMQg' \
--form 'file=@"/Users/naoraspir/Library/CloudStorage/OneDrive-OriginAI/originAI_ws/workspace/pepole-cluster/faces_test/test3.png"' \
--form 'session_key="test1"'

# to create token:
gcloud auth print-identity-token

# for testing aapi locally:
docker build -f real_time/Dockerfile.dev -t selfie-api-dev .

docker run -p 8000:8000 \
-v /Users/naoraspir/Library/CloudStorage/OneDrive-OriginAI/originAI_ws/workspace/pepole-cluster/peoples-algo-fastapi-python/real_time:/app/real_time \
selfie-api-dev 

docker run -p 8000:8000 \
-v google_key.json:/app/google_key.json \
-v /Users/naoraspir/Library/CloudStorage/OneDrive-OriginAI/originAI_ws/workspace/pepole-cluster/peoples-algo-fastapi-python/real_time:/app/real_time \
-e GOOGLE_APPLICATION_CREDENTIALS="/app/google_key.json" \
selfie-api-dev

# check health:
curl --location 'http://127.0.0.1:8000/health/' 

#check api:
curl --location 'http://127.0.0.1:8000/retrieve-images/' \
--form 'file=@"/Users/naoraspir/Library/CloudStorage/OneDrive-OriginAI/originAI_ws/workspace/pepole-cluster/faces_test/test3.png"' \
--form 'session_key="test1"'

```