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
# boto3 removed - not using S3 storage
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Network volume cache directory (RunPod serverless network volume)
MODEL_CACHE_DIR = "/runpod-volume/huggingface"

# Set HuggingFace cache directory to use local storage
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR
# Create the cache directory if it doesn't exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
logger.info(f"📁 Using local model cache: {MODEL_CACHE_DIR}")

# FLUX model compatibility fixes
# Disable xformers to avoid attn_output bugs
os.environ["DISABLE_XFORMERS"] = "1"
logger.info("🔧 Disabled xformers to avoid FLUX attention bugs")

# S3 sync functions removed - using network volume storage only

def debug_storage_info():
    """Debug storage configuration and return info"""
    try:
        import os
        import subprocess
        
        debug_info = {
            "status": "success",
            "network_volume": {
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
        
        # Check network volume cache contents
        if os.path.exists(MODEL_CACHE_DIR):
            try:
                for item in os.listdir(MODEL_CACHE_DIR):
                    item_path = os.path.join(MODEL_CACHE_DIR, item)
                    if os.path.isdir(item_path):
                        debug_info["network_volume"]["contents"].append({
                            "name": item,
                            "type": "directory",
                            "size": get_dir_size(item_path)
                        })
                    else:
                        debug_info["network_volume"]["contents"].append({
                            "name": item,
                            "type": "file",
                            "size": os.path.getsize(item_path)
                        })
            except Exception as e:
                debug_info["network_volume"]["error"] = str(e)
        
        # Check disk usage
        try:
            result = subprocess.run(['df', '-h'], capture_output=True, text=True)
            debug_info["disk_usage"]["df_output"] = result.stdout
        except Exception as e:
            debug_info["disk_usage"]["error"] = str(e)
        
        # Check network volume specifically
        try:
            result = subprocess.run(['df', '-h', '/runpod-volume'], capture_output=True, text=True)
            debug_info["network_volume_usage"] = result.stdout
        except Exception as e:
            debug_info["network_volume_usage"] = f"Error: {str(e)}"
        
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

# S3 model setup function removed - using network volume storage only

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
                
                # Enable memory efficient attention (skip xformers due to FLUX compatibility issues)
                if hasattr(self.kontext_pipeline, 'enable_model_cpu_offload'):
                    self.kontext_pipeline.enable_model_cpu_offload()
                # Skip xformers due to attn_output bug in FLUX models
                # if hasattr(self.kontext_pipeline, 'enable_xformers_memory_efficient_attention'):
                #     self.kontext_pipeline.enable_xformers_memory_efficient_attention()
                    
                logger.info("FluxKontextPipeline loaded successfully")
                
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
                
                # Enable memory efficient attention (skip xformers due to FLUX compatibility issues)
                if hasattr(self.fill_pipeline, 'enable_model_cpu_offload'):
                    self.fill_pipeline.enable_model_cpu_offload()
                # Skip xformers due to attn_output bug in FLUX models
                # if hasattr(self.fill_pipeline, 'enable_xformers_memory_efficient_attention'):
                #     self.fill_pipeline.enable_xformers_memory_efficient_attention()
                    
                logger.info("FluxFillPipeline loaded successfully")
                
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
            # Add specific FLUX compatibility settings
            generation_kwargs = {
                "prompt": instruction,
                "image": image,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width
            }
            
            # Ensure we're not using problematic attention processors
            logger.info(f"🎨 Starting instruction_edit generation with {num_inference_steps} steps")
            result = pipeline(**generation_kwargs)
            
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
        logger.info("🚀 Initializing FluxKontextHandler with network volume storage")
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
        
        # S3 credentials are ignored - using network volume storage only
        if "s3_access_key" in job_input and "s3_secret_key" in job_input:
            logger.info("⚠️ S3 credentials provided but ignored - using network volume storage")
        
        # Initialize handler
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

# Auto model setup removed - models will be downloaded to network volume on first use
logger.info("🗂️ Using network volume for model storage: /runpod-volume/huggingface")
logger.info("💡 Models will be downloaded to network volume on first use")

if __name__ == "__main__":
    runpod.serverless.start({"handler": runpod_handler}) 