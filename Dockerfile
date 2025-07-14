FROM runpod/pytorch:2.2.0-py3.11-cuda12.1.1-devel-ubuntu22.04

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

# Copy requirements and install setup dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir huggingface_hub  # Only needed for setup
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directories
RUN mkdir -p /workspace/models

# Copy application files
COPY . .

# Set environment variables
ENV TRANSFORMERS_CACHE=/workspace/models
ENV DIFFUSERS_CACHE=/workspace/models
ENV HUGGINGFACE_HUB_CACHE=/workspace/models
ENV PYTHONPATH=/workspace

# Make setup script executable
RUN chmod +x setup_models.py

# Download models during build (requires HUGGINGFACE_TOKEN build arg)
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN
RUN if [ -n "$HUGGINGFACE_TOKEN" ]; then \
        echo "Downloading models during build..."; \
        python setup_models.py; \
    else \
        echo "HUGGINGFACE_TOKEN not provided - models will need to be downloaded at runtime"; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the handler
CMD ["python", "handler.py"] 