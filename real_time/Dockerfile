# Stage 1: Build
FROM --platform=linux/amd64 python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake g++

# Copy the wheels and install them
COPY pytorch_facenet_wheels /build/pytorch_facenet_wheels
# Install any other Python dependencies
COPY real_time/requirements.txt .

# Install build dependencies if necessary
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir /build/pytorch_facenet_wheels/*.whl

# RUN which uvicorn

RUN apt-get purge -y --auto-remove build-essential cmake g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
ENV PYTHONPATH /app
ENV TORCH_HOME /root/.cache/torch
ENV PYTHONUNBUFFERED True


# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy the service account key file
# COPY google_key.json /app/google_key.json
# ENV GOOGLE_APPLICATION_CREDENTIALS /app/google_key.json

# copy chekpoints to torch cache inside docker
COPY checkpoints /root/.cache/torch/checkpoints

# Copy application code
COPY common /app/common
COPY real_time /app/real_time

# Set the working directory to the real_time directory
WORKDIR /app/real_time

# Command to run the uvicorn server
CMD ["sh", "-c", "uvicorn selfie_api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
