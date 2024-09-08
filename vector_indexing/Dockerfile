# Stage 1: Build 
FROM --platform=linux/amd64 python:3.11-slim AS builder 

WORKDIR /build

RUN apt-get update

COPY vector_indexing/requirements.txt .

# Install build dependencies if necessary
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get purge -y --auto-remove build-essential cmake g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY common /app/common
COPY vector_indexing /app/vector_indexing

# Set the working directory to the vector_indexing directory
WORKDIR /app/vector_indexing

CMD ["python", "indexing_app.py"]