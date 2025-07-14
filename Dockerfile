FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod SDK first
RUN pip install --no-cache-dir runpod>=1.5.0

# Install diffusers from git (required for FluxKontextPipeline)
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directories
RUN mkdir -p /workspace/models

# Copy application files
COPY . .

# Set environment variables
ENV TRANSFORMERS_CACHE=/workspace/models
ENV DIFFUSERS_CACHE=/workspace/models
ENV HUGGINGFACE_HUB_CACHE=/workspace/models
ENV CUDA_VISIBLE_DEVICES=0

# Optional: Download models during build (requires HF_TOKEN build arg)
ARG HF_TOKEN
RUN if [ -n "$HF_TOKEN" ]; then \
    python setup_models.py; \
    fi

# Start the handler
CMD ["python", "-u", "handler.py"] 