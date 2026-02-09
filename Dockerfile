# ==============================================================================
# RVC-Nano RunPod Serverless Worker
# Base image with CUDA 11.8 for PyTorch compatibility
# ==============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ==============================================================================
# System Dependencies
# ==============================================================================
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    # RVC-specific dependencies
    pkg-config \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# ==============================================================================
# Python Environment Setup
# ==============================================================================
# Install uv for fast package management
RUN pip install uv

# Create virtual environment
ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.10 /.venv

# Install setuptools first (required for some dependencies)
RUN uv pip install setuptools==69.5.1

# Install PyTorch with CUDA 11.8
RUN uv pip install torch==2.4.0+cu118 torchaudio==2.4.0+cu118 torchvision==0.19.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ==============================================================================
# Python Dependencies
# ==============================================================================
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt

# Install RunPod SDK
RUN uv pip install runpod huggingface_hub

# ==============================================================================
# Application Files
# ==============================================================================
# Copy source code
COPY src /src
COPY download_models.py /

# Create assets directory
RUN mkdir -p /assets/hubert /assets/rmvpe

# Download models from HuggingFace
RUN python /download_models.py

# Runpod serverless code
COPY schemas.py handler.py exceptions.py test_input.json /

# ==============================================================================
# Runtime Configuration
# ==============================================================================
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_home

# Run the handler
CMD ["python", "-u", "/handler.py"]
