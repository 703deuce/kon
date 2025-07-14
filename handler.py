import runpod
import torch
import base64
import io
import json
import logging
from PIL import Image
from diffusers import FluxKontextPipeline, FluxFillPipeline, FluxControlNetPipeline
from diffusers.models import FluxControlNetModel
from transformers import pipeline
import numpy as np
import cv2
from typing import Dict, Any, Optional, List
import traceback
import gc
import os
from huggingface_hub import login
import boto3
from botocore.client import Config
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure S3 storage for model caching
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3api-eu-ro-1.runpod.io")
S3_BUCKET = os.environ.get("S3_BUCKET", "ly11hhawq7")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")
S3_REGION = os.environ.get("S3_REGION", "eu-ro-1")

# Local cache directory
MODEL_CACHE_DIR = "/workspace/models"

# Create S3 client if credentials are available
s3_client = None
if S3_ACCESS_KEY and S3_SECRET_KEY:
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name=S3_REGION
        )
        logger.info("✅ S3 client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize S3 client: {e}")
        s3_client = None
else:
    logger.warning("⚠️ S3 credentials not found, S3 storage disabled")

# Set HuggingFace cache directory to use local storage
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR
# Create the cache directory if it doesn't exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
logger.info(f"📁 Using local model cache: {MODEL_CACHE_DIR}")

def sync_models_from_s3():
    """Sync models from S3 to local cache directory"""
    if not s3_client:
        logger.info("⏭️ S3 not configured, skipping sync from S3")
        return False
        
    try:
        logger.info("🔄 Syncing models from S3 to local cache...")
        
        # List all objects in the models/ prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=S3_BUCKET, Prefix='models/')
        
        downloaded_files = 0
        for page in page_iterator:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    local_path = os.path.join("/workspace", s3_key)
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download file if it doesn't exist locally or is newer
                    if not os.path.exists(local_path):
                        logger.info(f"📥 Downloading {s3_key}")
                        s3_client.download_file(S3_BUCKET, s3_key, local_path)
                        downloaded_files += 1
        
        logger.info(f"✅ Synced {downloaded_files} files from S3")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error syncing from S3: {e}")
        return False

def sync_models_to_s3():
    """Sync models from local cache to S3"""
    if not s3_client:
        logger.info("⏭️ S3 not configured, skipping sync to S3")
        return False
        
    try:
        logger.info("🔄 Syncing models from local cache to S3...")
        
        if not os.path.exists(MODEL_CACHE_DIR):
            logger.info("📁 No local models to sync")
            return True
        
        uploaded_files = 0
        for root, dirs, files in os.walk(MODEL_CACHE_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                # Convert local path to S3 key
                rel_path = os.path.relpath(local_path, "/workspace")
                s3_key = rel_path.replace("\\", "/")  # Ensure forward slashes
                
                try:
                    # Check if file exists in S3
                    s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                    logger.debug(f"⏭️ Skipping {s3_key} (already exists)")
                except:
                    # File doesn't exist in S3, upload it
                    logger.info(f"📤 Uploading {s3_key}")
                    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
                    uploaded_files += 1
        
        logger.info(f"✅ Uploaded {uploaded_files} files to S3")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error syncing to S3: {e}")
        return False

def debug_storage_info():
    """Debug storage configuration and return info"""
    try:
        import os
        import subprocess
        
        debug_info = {
            "status": "success",
            "s3_config": {
                "endpoint": S3_ENDPOINT,
                "bucket": S3_BUCKET,
                "region": S3_REGION
            },
            "local_cache": {
                "cache_dir": MODEL_CACHE_DIR,
                "exists": os.path.exists(MODEL_CACHE_DIR),
                "contents": []
            },
            "disk_usage": {},
            "environment": {
                "HF_HOME": os.environ.get("HF_HOME"),
                "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
                "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE")
            }
        }
        
        # Check local cache contents
        if os.path.exists(MODEL_CACHE_DIR):
            try:
                for item in os.listdir(MODEL_CACHE_DIR):
                    item_path = os.path.join(MODEL_CACHE_DIR, item)
                    if os.path.isdir(item_path):
                        debug_info["local_cache"]["contents"].append({
                            "name": item,
                            "type": "directory",
                            "size": get_dir_size(item_path)
                        })
                    else:
                        debug_info["local_cache"]["contents"].append({
                            "name": item,
                            "type": "file",
                            "size": os.path.getsize(item_path)
                        })
            except Exception as e:
                debug_info["local_cache"]["error"] = str(e)
        
        # Check disk usage
        try:
            result = subprocess.run(['df', '-h'], capture_output=True, text=True)
            debug_info["disk_usage"]["df_output"] = result.stdout
        except Exception as e:
            debug_info["disk_usage"]["error"] = str(e)
        
        # Test S3 connection
        if s3_client:
            try:
                response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix='models/', MaxKeys=5)
                debug_info["s3_test"] = {
                    "status": "success",
                    "objects_found": len(response.get('Contents', [])),
                    "sample_objects": [obj['Key'] for obj in response.get('Contents', [])[:5]]
                }
            except Exception as e:
                debug_info["s3_test"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            debug_info["s3_test"] = {
                "status": "disabled",
                "error": "S3 client not configured"
            }
        
        return debug_info
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def get_dir_size(path):
    """Get directory size in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception:
        pass
    return total_size

class FluxKontextHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.kontext_pipeline = None
        self.fill_pipeline = None
        self.depth_controlnet = None
        self.canny_controlnet = None
        self.control_pipeline = None
        self.depth_detector = None
        self.canny_detector = None
        self.hf_authenticated = False
        
    def authenticate_hf(self):
        """Authenticate with HuggingFace"""
        if not self.hf_authenticated:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                try:
                    logger.info("🔐 Authenticating with HuggingFace...")
                    login(token=hf_token)
                    self.hf_authenticated = True
                    logger.info("✅ HuggingFace authentication successful")
                except Exception as e:
                    logger.error(f"❌ HuggingFace authentication failed: {e}")
                    raise
            else:
                logger.warning("⚠️ No HuggingFace token found")
        
        return self.hf_authenticated
        
    def load_kontext_pipeline(self):
        """Load the FluxKontextPipeline for instruction-based editing"""
        if self.kontext_pipeline is None:
            try:
                # Authenticate with HF first
                self.authenticate_hf()
                
                logger.info("Loading FluxKontextPipeline...")
                logger.info(f"🗂️ Model cache location: {MODEL_CACHE_DIR}")
                
                pipeline_kwargs = {
                    "torch_dtype": self.dtype,
                    "use_safetensors": True,
                    "cache_dir": MODEL_CACHE_DIR
                }
                
                self.kontext_pipeline = FluxKontextPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-Kontext-dev",
                    **pipeline_kwargs
                )
                self.kontext_pipeline.to(self.device)
                
                # Enable memory efficient attention
                if hasattr(self.kontext_pipeline, 'enable_model_cpu_offload'):
                    self.kontext_pipeline.enable_model_cpu_offload()
                if hasattr(self.kontext_pipeline, 'enable_xformers_memory_efficient_attention'):
                    self.kontext_pipeline.enable_xformers_memory_efficient_attention()
                    
                logger.info("FluxKontextPipeline loaded successfully")
                
                # Sync models to S3 after loading
                sync_models_to_s3()
                
            except Exception as e:
                logger.error(f"Error loading FluxKontextPipeline: {e}")
                raise
        return self.kontext_pipeline
        
    def load_fill_pipeline(self):
        """Load the FluxFillPipeline for traditional inpainting"""
        if self.fill_pipeline is None:
            try:
                # Authenticate with HF first
                self.authenticate_hf()
                
                logger.info("Loading FluxFillPipeline...")
                logger.info(f"🗂️ Model cache location: {MODEL_CACHE_DIR}")
                
                pipeline_kwargs = {
                    "torch_dtype": self.dtype,
                    "use_safetensors": True,
                    "cache_dir": MODEL_CACHE_DIR
                }
                
                self.fill_pipeline = FluxFillPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-Fill-dev",
                    **pipeline_kwargs
                )
                self.fill_pipeline.to(self.device)
                
                # Enable memory efficient attention
                if hasattr(self.fill_pipeline, 'enable_model_cpu_offload'):
                    self.fill_pipeline.enable_model_cpu_offload()
                if hasattr(self.fill_pipeline, 'enable_xformers_memory_efficient_attention'):
                    self.fill_pipeline.enable_xformers_memory_efficient_attention()
                    
                logger.info("FluxFillPipeline loaded successfully")
                
                # Sync models to S3 after loading
                sync_models_to_s3()
                
            except Exception as e:
                logger.error(f"Error loading FluxFillPipeline: {e}")
                raise
        return self.fill_pipeline
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    
    def _prepare_mask(self, image: Image.Image, mask_data: Any) -> Image.Image:
        """Prepare mask for inpainting"""
        if isinstance(mask_data, str):
            # Base64 encoded mask
            mask = self._base64_to_image(mask_data)
        elif isinstance(mask_data, dict) and "coordinates" in mask_data:
            # Create mask from coordinates
            width, height = image.size
            mask = Image.new("L", (width, height), 0)
            # Implementation for coordinate-based masking would go here
        else:
            raise ValueError("Invalid mask format")
        
        return mask.convert("L")
    
    def _generate_depth_map(self, image: Image.Image) -> Image.Image:
        """Generate depth map from image"""
        depth = self.depth_estimator(image)
        depth_image = depth["depth"]
        
        # Convert to PIL Image
        depth_array = np.array(depth_image)
        depth_normalized = ((depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
        return Image.fromarray(depth_normalized)
    
    def _generate_canny_edges(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Generate Canny edge detection"""
        image_array = np.array(image)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return Image.fromarray(edges)
    
    def fill_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill/inpaint image using FLUX.1 Fill"""
        try:
            # Load the Fill pipeline
            pipeline = self.load_fill_pipeline()
            
            # Extract parameters
            image_b64 = params.get("image")
            mask_data = params.get("mask")
            prompt = params.get("prompt", "")
            num_inference_steps = params.get("num_inference_steps", 30)
            guidance_scale = params.get("guidance_scale", 30.0)
            strength = params.get("strength", 0.8)
            seed = params.get("seed")
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            
            # Validate required parameters
            if not image_b64:
                raise ValueError("Image is required")
            
            if not prompt:
                raise ValueError("Prompt is required")
            
            # Prepare inputs
            image = self._base64_to_image(image_b64)
            
            # Handle mask - if no mask provided, create a full white mask for instruction-based editing
            if mask_data:
                mask = self._prepare_mask(image, mask_data)
            else:
                # Create a full white mask for instruction-based editing (like Replicate)
                mask = Image.new("L", image.size, 255)  # Full white mask
            
            # Resize if needed
            if width != image.width or height != image.height:
                image = image.resize((width, height))
                mask = mask.resize((width, height))
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate
            result = pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                height=height,
                width=width
            )
            
            # Convert result to base64
            output_image = result.images[0]
            output_b64 = self._image_to_base64(output_image)
            
            return {
                "status": "success",
                "image": output_b64,
                "parameters": {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "seed": seed,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fill_image: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def instruction_edit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """True FLUX.1 Kontext instruction-based editing without masks"""
        try:
            # Load the Kontext pipeline
            pipeline = self.load_kontext_pipeline()
            
            # Extract parameters
            image_b64 = params.get("image")
            instruction = params.get("instruction") or params.get("prompt", "")
            num_inference_steps = params.get("num_inference_steps", 30)
            guidance_scale = params.get("guidance_scale", 3.5)
            seed = params.get("seed")
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            
            # Validate required parameters
            if not image_b64:
                raise ValueError("Image is required")
            
            if not instruction:
                raise ValueError("Instruction is required")
            
            # Prepare inputs
            image = self._base64_to_image(image_b64)
            
            # Resize if needed
            if width != image.width or height != image.height:
                image = image.resize((width, height))
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate using instruction-based editing
            result = pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            # Convert result to base64
            output_image = result.images[0]
            output_b64 = self._image_to_base64(output_image)
            
            return {
                "status": "success",
                "image": output_b64,
                "parameters": {
                    "instruction": instruction,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            logger.error(f"Error in instruction_edit: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def depth_controlled_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with depth control"""
        try:
            # Extract parameters
            image_b64 = params.get("image")
            prompt = params.get("prompt", "")
            num_inference_steps = params.get("num_inference_steps", 30)
            guidance_scale = params.get("guidance_scale", 3.5)
            controlnet_conditioning_scale = params.get("controlnet_conditioning_scale", 0.6)
            seed = params.get("seed")
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            
            if not image_b64:
                raise ValueError("Image is required")
            
            # Prepare inputs
            image = self._base64_to_image(image_b64)
            depth_map = self._generate_depth_map(image)
            
            # Resize if needed
            if width != image.width or height != image.height:
                image = image.resize((width, height))
                depth_map = depth_map.resize((width, height))
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate
            result = self.control_pipeline(
                prompt=prompt,
                control_image=[depth_map],
                controlnet_conditioning_scale=[controlnet_conditioning_scale],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            # Convert result to base64
            output_image = result.images[0]
            output_b64 = self._image_to_base64(output_image)
            
            return {
                "status": "success",
                "image": output_b64,
                "depth_map": self._image_to_base64(depth_map),
                "parameters": {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                    "seed": seed,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            logger.error(f"Error in depth_controlled_generation: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def canny_controlled_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with Canny edge control"""
        try:
            # Extract parameters
            image_b64 = params.get("image")
            prompt = params.get("prompt", "")
            num_inference_steps = params.get("num_inference_steps", 30)
            guidance_scale = params.get("guidance_scale", 3.5)
            controlnet_conditioning_scale = params.get("controlnet_conditioning_scale", 0.6)
            low_threshold = params.get("low_threshold", 100)
            high_threshold = params.get("high_threshold", 200)
            seed = params.get("seed")
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            
            if not image_b64:
                raise ValueError("Image is required")
            
            # Prepare inputs
            image = self._base64_to_image(image_b64)
            canny_edges = self._generate_canny_edges(image, low_threshold, high_threshold)
            
            # Resize if needed
            if width != image.width or height != image.height:
                image = image.resize((width, height))
                canny_edges = canny_edges.resize((width, height))
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate
            result = self.control_pipeline(
                prompt=prompt,
                control_image=[canny_edges],
                controlnet_conditioning_scale=[controlnet_conditioning_scale],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            # Convert result to base64
            output_image = result.images[0]
            output_b64 = self._image_to_base64(output_image)
            
            return {
                "status": "success",
                "image": output_b64,
                "canny_edges": self._image_to_base64(canny_edges),
                "parameters": {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "seed": seed,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            logger.error(f"Error in canny_controlled_generation: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def multi_controlnet_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with multiple ControlNet conditions"""
        try:
            # Extract parameters
            image_b64 = params.get("image")
            prompt = params.get("prompt", "")
            num_inference_steps = params.get("num_inference_steps", 30)
            guidance_scale = params.get("guidance_scale", 3.5)
            depth_conditioning_scale = params.get("depth_conditioning_scale", 0.6)
            canny_conditioning_scale = params.get("canny_conditioning_scale", 0.6)
            low_threshold = params.get("low_threshold", 100)
            high_threshold = params.get("high_threshold", 200)
            seed = params.get("seed")
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            
            if not image_b64:
                raise ValueError("Image is required")
            
            # Prepare inputs
            image = self._base64_to_image(image_b64)
            depth_map = self._generate_depth_map(image)
            canny_edges = self._generate_canny_edges(image, low_threshold, high_threshold)
            
            # Resize if needed
            if width != image.width or height != image.height:
                image = image.resize((width, height))
                depth_map = depth_map.resize((width, height))
                canny_edges = canny_edges.resize((width, height))
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate with multiple ControlNets
            result = self.control_pipeline(
                prompt=prompt,
                control_image=[depth_map, canny_edges],
                controlnet_conditioning_scale=[depth_conditioning_scale, canny_conditioning_scale],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            
            # Convert result to base64
            output_image = result.images[0]
            output_b64 = self._image_to_base64(output_image)
            
            return {
                "status": "success",
                "image": output_b64,
                "depth_map": self._image_to_base64(depth_map),
                "canny_edges": self._image_to_base64(canny_edges),
                "parameters": {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "depth_conditioning_scale": depth_conditioning_scale,
                    "canny_conditioning_scale": canny_conditioning_scale,
                    "low_threshold": low_threshold,
                    "high_threshold": high_threshold,
                    "seed": seed,
                    "width": width,
                    "height": height
                }
            }
            
        except Exception as e:
            logger.error(f"Error in multi_controlnet_generation: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "status": "success",
            "models": {
                "fill_pipeline": "black-forest-labs/FLUX.1-Fill-dev",
                "kontext_pipeline": "black-forest-labs/FLUX.1-Kontext-dev",
                "controlnet_depth": "black-forest-labs/FLUX.1-Depth-dev",
                "controlnet_canny": "black-forest-labs/FLUX.1-Canny-dev",
                "depth_estimator": "Intel/dpt-hybrid-midas"
            },
            "device": self.device,
            "dtype": str(self.dtype),
            "cuda_available": torch.cuda.is_available(),
            "endpoints": [
                "fill_image",
                "instruction_edit",
                "depth_controlled_generation", 
                "canny_controlled_generation",
                "multi_controlnet_generation",
                "get_model_info"
            ]
        }

# Global handler instance
handler = None

def initialize_handler():
    """Initialize the global handler"""
    global handler
    if handler is None:
        # Sync models from S3 to local cache before initializing handler
        sync_models_from_s3()
        handler = FluxKontextHandler()
    return handler

def runpod_handler(job):
    """Main RunPod handler function"""
    try:
        # Set HF_TOKEN from job env if provided
        job_env = job.get("env", {})
        if "HF_TOKEN" in job_env:
            os.environ["HF_TOKEN"] = job_env["HF_TOKEN"]
            logger.info("HF_TOKEN set from job environment")
        
        # Get input parameters
        job_input = job.get("input", {})
        
        # Also check for hf_token in job input and set as environment variable
        if "hf_token" in job_input:
            os.environ["HF_TOKEN"] = job_input["hf_token"]
            logger.info("HF_TOKEN set from job input")
        
        # Initialize handler if not already done
        global handler
        if handler is None:
            handler = initialize_handler()
        endpoint = job_input.get("endpoint", "")
        
        # Route to appropriate endpoint
        if endpoint == "fill_image":
            result = handler.fill_image(job_input)
        elif endpoint == "instruction_edit":
            result = handler.instruction_edit(job_input)
        elif endpoint == "depth_controlled_generation":
            result = handler.depth_controlled_generation(job_input)
        elif endpoint == "canny_controlled_generation":
            result = handler.canny_controlled_generation(job_input)
        elif endpoint == "multi_controlnet_generation":
            result = handler.multi_controlnet_generation(job_input)
        elif endpoint == "get_model_info":
            result = handler.get_model_info()
        elif endpoint == "debug_storage":
            result = debug_storage_info()
        else:
            result = {
                "status": "error",
                "error": f"Unknown endpoint: {endpoint}",
                "available_endpoints": [
                    "fill_image",
                    "instruction_edit",
                    "depth_controlled_generation",
                    "canny_controlled_generation", 
                    "multi_controlnet_generation",
                    "get_model_info",
                    "debug_storage"
                ]
            }
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in runpod_handler: {str(e)}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Initialize handler at module level
handler = None

if __name__ == "__main__":
    runpod.serverless.start({"handler": runpod_handler}) 