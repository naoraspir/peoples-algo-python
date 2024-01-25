# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake g++

COPY pytorch_facenet_wheels /build/pytorch_facenet_wheels
COPY precompiled_wheels /build/precompiled_wheels
COPY clustering/requirements.txt .

# Install build dependencies if necessary
RUN pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir /build/pytorch_facenet_wheels/*.whl \
                            /build/precompiled_wheels/*.whl && \
                            pip install --upgrade pip && \
                            pip install --no-cache-dir -r requirements.txt

RUN apt-get purge -y --auto-remove build-essential cmake g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the service account key file
# COPY google_key.json /app/google_key.json
# ENV GOOGLE_APPLICATION_CREDENTIALS /app/google_key.json

# copy chekpoints to torch cache inside docker
# COPY checkpoints /root/.cache/torch/checkpoints

# Copy application code
COPY common /app/common
COPY clustering /app/clustering

# Set the working directory to the clustering directory
WORKDIR /app/clustering

CMD ["python", "clustering_app.py"]