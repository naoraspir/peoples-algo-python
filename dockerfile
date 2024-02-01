# Use a base image with necessary build tools
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /build

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

# Clone and build dlib
RUN git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git && \
    cd dlib && \
    python3 setup.py bdist_wheel && \
    mv dist/*.whl /wheels/

# Create a directory for wheel files
RUN mkdir /wheels
