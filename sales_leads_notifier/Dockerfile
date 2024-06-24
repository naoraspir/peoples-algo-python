# Stage 1: Build 
FROM --platform=linux/amd64 python:3.11-slim as builder 

WORKDIR /build

COPY sales_leads_notifier/requirements.txt .

# Install build dependencies if necessary
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
ENV PYTHONPATH /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY common /app/common
COPY sales_leads_notifier /app/sales_lead_notifier

# Set the working directory to the sales_lead_notifier directory
WORKDIR /app/sales_lead_notifier

CMD ["python", "main.py"]

