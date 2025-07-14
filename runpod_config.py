"""
RunPod Configuration for FLUX.1 Kontext Serverless
"""

import os

# RunPod Settings
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")

# Model Configuration
MODEL_CONFIG = {
    "fill_model": "black-forest-labs/FLUX.1-Fill-dev",
    "depth_controlnet": "black-forest-labs/FLUX.1-Depth-dev", 
    "canny_controlnet": "black-forest-labs/FLUX.1-Canny-dev",
    "base_model": "black-forest-labs/FLUX.1-dev",
    "depth_estimator": "Intel/dpt-hybrid-midas"
}

# Default Parameters
DEFAULT_PARAMS = {
    "fill_image": {
        "num_inference_steps": 30,
        "guidance_scale": 30.0,
        "strength": 0.8,
        "width": 1024,
        "height": 1024
    },
    "depth_controlled_generation": {
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "controlnet_conditioning_scale": 0.6,
        "width": 1024,
        "height": 1024
    },
    "canny_controlled_generation": {
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "controlnet_conditioning_scale": 0.6,
        "low_threshold": 100,
        "high_threshold": 200,
        "width": 1024,
        "height": 1024
    },
    "multi_controlnet_generation": {
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "depth_conditioning_scale": 0.6,
        "canny_conditioning_scale": 0.6,
        "low_threshold": 100,
        "high_threshold": 200,
        "width": 1024,
        "height": 1024
    }
}

# Resource Configuration
RESOURCE_CONFIG = {
    "gpu_count": 1,
    "gpu_type": "RTX 4090",
    "memory_gb": 24,
    "disk_gb": 50,
    "container_disk_gb": 20
}

# Timeout Configuration (in seconds)
TIMEOUT_CONFIG = {
    "request_timeout": 600,  # 10 minutes
    "idle_timeout": 300,     # 5 minutes
    "max_execution_time": 900  # 15 minutes
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Cache Configuration
CACHE_CONFIG = {
    "model_cache_dir": "/runpod-volume/huggingface",
    "temp_cache_dir": "/tmp/flux_cache"
}

# Environment Variables
ENV_VARS = {
    "TRANSFORMERS_CACHE": "/runpod-volume/huggingface",
    "DIFFUSERS_CACHE": "/runpod-volume/huggingface",
    "HUGGINGFACE_HUB_CACHE": "/runpod-volume/huggingface",
    "CUDA_VISIBLE_DEVICES": "0"
}

# Health Check Configuration
HEALTH_CHECK = {
    "endpoint": "/health",
    "timeout": 30,
    "retries": 3
}

# Scaling Configuration
SCALING_CONFIG = {
    "min_workers": 0,
    "max_workers": 10,
    "scale_up_threshold": 0.8,
    "scale_down_threshold": 0.2,
    "scale_up_cooldown": 60,
    "scale_down_cooldown": 300
}

# Network Configuration
NETWORK_CONFIG = {
    "max_request_size": "100MB",
    "max_response_size": "100MB",
    "compression": True
}

# Available Endpoints
ENDPOINTS = {
    "fill_image": {
        "description": "Fill/inpaint image using FLUX.1 Fill with optional mask",
        "required_params": ["image", "prompt"],
        "optional_params": [
            "mask", "num_inference_steps", "guidance_scale", "strength", 
            "seed", "width", "height"
        ]
    },
    "instruction_edit": {
        "description": "True FLUX.1 Kontext instruction-based editing without masks",
        "required_params": ["image", "instruction"],
        "optional_params": [
            "num_inference_steps", "guidance_scale", "seed", "width", "height"
        ]
    },
    "depth_controlled_generation": {
        "description": "Generate image with depth control",
        "required_params": ["image", "prompt"],
        "optional_params": [
            "num_inference_steps", "guidance_scale", 
            "controlnet_conditioning_scale", "seed", "width", "height"
        ]
    },
    "canny_controlled_generation": {
        "description": "Generate image with Canny edge control",
        "required_params": ["image", "prompt"],
        "optional_params": [
            "num_inference_steps", "guidance_scale",
            "controlnet_conditioning_scale", "low_threshold", 
            "high_threshold", "seed", "width", "height"
        ]
    },
    "multi_controlnet_generation": {
        "description": "Generate image with multiple ControlNet conditions",
        "required_params": ["image", "prompt"],
        "optional_params": [
            "num_inference_steps", "guidance_scale",
            "depth_conditioning_scale", "canny_conditioning_scale",
            "low_threshold", "high_threshold", "seed", "width", "height"
        ]
    },
    "get_model_info": {
        "description": "Get information about loaded models",
        "required_params": [],
        "optional_params": []
    }
}

# Validation Rules
VALIDATION_RULES = {
    "image_formats": ["PNG", "JPEG", "JPG", "WEBP"],
    "max_image_size": 2048,
    "min_image_size": 512,
    "max_prompt_length": 1000,
    "max_inference_steps": 100,
    "min_inference_steps": 1,
    "max_guidance_scale": 50.0,
    "min_guidance_scale": 1.0
}

# Error Messages
ERROR_MESSAGES = {
    "missing_endpoint": "Endpoint parameter is required",
    "invalid_endpoint": "Invalid endpoint specified",
    "missing_image": "Image parameter is required",
    "invalid_image": "Invalid image format or corrupted image",
    "invalid_mask": "Invalid mask format",
    "missing_prompt": "Prompt parameter is required",
    "invalid_parameters": "Invalid parameters provided",
    "model_not_loaded": "Model not loaded or initialization failed",
    "gpu_memory_error": "Insufficient GPU memory",
    "timeout_error": "Request timed out"
} 