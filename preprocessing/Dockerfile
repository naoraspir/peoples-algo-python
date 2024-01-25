# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies if necessary
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential
RUN pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu
# Copy the wheels and install them
COPY pytorch_facenet_wheels /build/pytorch_facenet_wheels
RUN pip install --no-cache-dir /build/pytorch_facenet_wheels/*.whl

# Install any other Python dependencies
COPY preprocessing/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

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
COPY checkpoints /root/.cache/torch/checkpoints

# Copy application code
COPY common /app/common
COPY preprocessing /app/preprocessing

# Set the working directory to the preprocessing directory
WORKDIR /app/preprocessing

CMD ["python", "preprocess_app.py"]