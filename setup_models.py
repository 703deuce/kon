#!/usr/bin/env python3
"""
Model setup script for FLUX.1 Kontext RunPod deployment.
Downloads models once during container initialization.
"""

import os
import logging
from huggingface_hub import login
from diffusers import FluxFillPipeline, FluxPipeline
from diffusers.models import FluxControlNetModel
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download all required models to local cache"""
    try:
        # Get HF token from environment
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is required for model download")
        
        # Login to Hugging Face
        logger.info("Authenticating with Hugging Face...")
        login(token=hf_token)
        logger.info("✅ Authentication successful")
        
        # Download FLUX.1 Fill model
        logger.info("Downloading FLUX.1 Fill model...")
        FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype="auto",
            use_safetensors=True
        )
        logger.info("✅ FLUX.1 Fill model downloaded")
        
        # Download FLUX.1 Kontext model
        logger.info("Downloading FLUX.1 Kontext model...")
        FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype="auto",
            use_safetensors=True
        )
        logger.info("✅ FLUX.1 Kontext model downloaded")
        
        # Download ControlNet models
        logger.info("Downloading FLUX.1 Depth ControlNet...")
        FluxControlNetModel.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            torch_dtype="auto",
            use_safetensors=True
        )
        logger.info("✅ FLUX.1 Depth ControlNet downloaded")
        
        logger.info("Downloading FLUX.1 Canny ControlNet...")
        FluxControlNetModel.from_pretrained(
            "black-forest-labs/FLUX.1-Canny-dev",
            torch_dtype="auto",
            use_safetensors=True
        )
        logger.info("✅ FLUX.1 Canny ControlNet downloaded")
        
        # Download base FLUX.1 model
        logger.info("Downloading FLUX.1 base model...")
        FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype="auto",
            use_safetensors=True
        )
        logger.info("✅ FLUX.1 base model downloaded")
        
        # Download depth estimator
        logger.info("Downloading depth estimator...")
        pipeline(
            "depth-estimation",
            model="Intel/dpt-hybrid-midas"
        )
        logger.info("✅ Depth estimator downloaded")
        
        logger.info("🎉 All models downloaded successfully!")
        logger.info("🔥 Ready for local inference without authentication!")
        
    except Exception as e:
        logger.error(f"❌ Error downloading models: {str(e)}")
        raise e

if __name__ == "__main__":
    download_models() 