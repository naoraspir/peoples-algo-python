# Stage 1: Build 
FROM --platform=linux/amd64 python:3.11-slim AS builder 

WORKDIR /build

COPY algo_pipeline_executer/requirements.txt .

# Install build dependencies if necessary
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
ENV PYTHONPATH /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# # Copy the service account key file
# COPY google_key.json /app/google_key.json
# ENV GOOGLE_APPLICATION_CREDENTIALS /app/google_key.json

# Copy application code
COPY common /app/common
COPY algo_pipeline_executer /app/algo_pipeline_executer

# Set the working directory to the algo_pipeline_executer directory
WORKDIR /app/algo_pipeline_executer

CMD ["python", "main.py"]
