FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH=/workspace
ENV TRANSFORMERS_CACHE=/workspace/models
ENV DIFFUSERS_CACHE=/workspace/models
ENV HUGGINGFACE_HUB_CACHE=/workspace/models
ENV CUDA_VISIBLE_DEVICES=0

# Hugging Face authentication (set this in RunPod environment)
# ENV HUGGINGFACE_TOKEN=your_token_here

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create model cache directory
RUN mkdir -p /workspace/models

# Copy application files
COPY handler.py .
COPY runpod_config.py .

# Pre-download models (optional - uncomment if you want to bake models into image)
# RUN python -c "
# from diffusers import FluxFillPipeline, FluxControlNetPipeline
# from diffusers.models import FluxControlNetModel
# from transformers import pipeline
# import torch
# print('Pre-downloading models...')
# FluxFillPipeline.from_pretrained('black-forest-labs/FLUX.1-Fill-dev', torch_dtype=torch.bfloat16)
# FluxControlNetModel.from_pretrained('black-forest-labs/FLUX.1-Depth-dev', torch_dtype=torch.bfloat16)
# FluxControlNetModel.from_pretrained('black-forest-labs/FLUX.1-Canny-dev', torch_dtype=torch.bfloat16)
# FluxControlNetPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16)
# pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')
# print('Models downloaded successfully')
# "

# Set permissions
RUN chmod +x handler.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Expose port (optional for debugging)
EXPOSE 8000

# Command to run the application
CMD ["python", "handler.py"] 