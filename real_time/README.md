
```sh
# for testing api:

curl --location 'https://selfie-service-dev-34y6ttkera-ue.a.run.app/retrieve-images/' \
--header 'Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImQ3YjkzOTc3MWE3ODAwYzQxM2Y5MDA1MTAxMmQ5NzU5ODE5MTZkNzEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIzMjU1NTk0MDU1OS5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsImF1ZCI6IjMyNTU1OTQwNTU5LmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTEyNDc3OTMzNTkzMjIxNjc3NDAwIiwiZW1haWwiOiJuYW9yYXNwaXJAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiJoemRXRlNlWGpza0NBbi1Ld0ZrR0l3IiwiaWF0IjoxNzI2MDM2ODEyLCJleHAiOjE3MjYwNDA0MTJ9.iec96Aiqe1MAfKTLO9qM2LXZWj9OrDoL0jZmBL9GnsXuVOEbKsPfPHmO_1TsQlhSHGOgUB9rYlS1z9k3zjKJRFt196dyLzoMJZGlgsfhOLbOcXQn_1WIZDtvN5PvMNdc3pGZb8Wty5kc8FX2-x1mJHAIkVs7hdgA2QpUkX9a4Sf2r-kddSQX15gY_RAyCXSed3Z7fZno9OFHKd8ND2FlJhjKcg3al9t0iz8kYKP0JSyUvfEGwWU-5WKTXc2McKnN_FlCAXcwkFpRqRaRP4rqR1kpWMgvBQUVopzkFVYntHBtfgXwqQIB0jheedYcbK4_GBPY04Lr7OynB9GCYoyjpA' \
--form 'file=@"/Users/naoraspir/Library/CloudStorage/OneDrive-OriginAI/originAI_ws/workspace/pepole-cluster/faces_test/faces/252_52.jpg"' \
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
--form 'file=@"/Users/naoraspir/Library/CloudStorage/OneDrive-OriginAI/originAI_ws/workspace/pepole-cluster/faces_test/faces/252_52.jpg"' \
--form 'session_key="test1"'

```