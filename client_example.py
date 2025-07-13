#!/usr/bin/env python3
"""
FLUX.1 Kontext RunPod Client Example

This example demonstrates how to use the RunPod serverless API
for FLUX.1 Kontext image generation and editing.
"""

import requests
import base64
import json
import io
import os
from PIL import Image
from typing import Optional, Dict, Any

class FluxKontextClient:
    """Client for FLUX.1 Kontext RunPod serverless endpoint"""
    
    def __init__(self, endpoint_url: str, api_key: str):
        """
        Initialize the client
        
        Args:
            endpoint_url: RunPod endpoint URL (e.g., https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync)
            api_key: RunPod API key
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    def base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    
    def call_endpoint(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API call to RunPod endpoint"""
        payload = {
            "input": {
                "endpoint": endpoint,
                **kwargs
            }
        }
        
        print(f"Calling endpoint: {endpoint}")
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=payload,
            timeout=600  # 10 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return self.call_endpoint("get_model_info")
    
    def fill_image(self, image_path: str, prompt: str, mask_path: Optional[str] = None,
                  num_inference_steps: int = 30, guidance_scale: float = 30.0,
                  strength: float = 0.8, seed: Optional[int] = None,
                  width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """
        Fill/inpaint image using FLUX.1 Fill
        
        Args:
            image_path: Path to input image
            prompt: Text description of desired changes
            mask_path: Optional path to mask image (white = edit, black = keep). 
                      If not provided, model will intelligently edit based on prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence strength
            strength: Inpainting strength
            seed: Random seed for reproducibility
            width: Output width
            height: Output height
        """
        image_b64 = self.image_to_base64(image_path)
        
        params = {
            "image": image_b64,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "width": width,
            "height": height
        }
        
        # Add mask if provided
        if mask_path:
            mask_b64 = self.image_to_base64(mask_path)
            params["mask"] = mask_b64
        
        if seed is not None:
            params["seed"] = seed
        
        return self.call_endpoint("fill_image", **params)
    
    def depth_controlled_generation(self, image_path: str, prompt: str,
                                  num_inference_steps: int = 30, guidance_scale: float = 3.5,
                                  controlnet_conditioning_scale: float = 0.6,
                                  seed: Optional[int] = None,
                                  width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """
        Generate image with depth control
        
        Args:
            image_path: Path to input image
            prompt: Text description of desired output
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence strength
            controlnet_conditioning_scale: Depth control strength
            seed: Random seed for reproducibility
            width: Output width
            height: Output height
        """
        image_b64 = self.image_to_base64(image_path)
        
        params = {
            "image": image_b64,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "width": width,
            "height": height
        }
        
        if seed is not None:
            params["seed"] = seed
        
        return self.call_endpoint("depth_controlled_generation", **params)
    
    def canny_controlled_generation(self, image_path: str, prompt: str,
                                   num_inference_steps: int = 30, guidance_scale: float = 3.5,
                                   controlnet_conditioning_scale: float = 0.6,
                                   low_threshold: int = 100, high_threshold: int = 200,
                                   seed: Optional[int] = None,
                                   width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """
        Generate image with Canny edge control
        
        Args:
            image_path: Path to input image
            prompt: Text description of desired output
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence strength
            controlnet_conditioning_scale: Edge control strength
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
            seed: Random seed for reproducibility
            width: Output width
            height: Output height
        """
        image_b64 = self.image_to_base64(image_path)
        
        params = {
            "image": image_b64,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "width": width,
            "height": height
        }
        
        if seed is not None:
            params["seed"] = seed
        
        return self.call_endpoint("canny_controlled_generation", **params)
    
    def multi_controlnet_generation(self, image_path: str, prompt: str,
                                   num_inference_steps: int = 30, guidance_scale: float = 3.5,
                                   depth_conditioning_scale: float = 0.6,
                                   canny_conditioning_scale: float = 0.6,
                                   low_threshold: int = 100, high_threshold: int = 200,
                                   seed: Optional[int] = None,
                                   width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """
        Generate image with multiple ControlNet conditions
        
        Args:
            image_path: Path to input image
            prompt: Text description of desired output
            num_inference_steps: Number of denoising steps
            guidance_scale: Prompt adherence strength
            depth_conditioning_scale: Depth control strength
            canny_conditioning_scale: Edge control strength
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
            seed: Random seed for reproducibility
            width: Output width
            height: Output height
        """
        image_b64 = self.image_to_base64(image_path)
        
        params = {
            "image": image_b64,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "depth_conditioning_scale": depth_conditioning_scale,
            "canny_conditioning_scale": canny_conditioning_scale,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "width": width,
            "height": height
        }
        
        if seed is not None:
            params["seed"] = seed
        
        return self.call_endpoint("multi_controlnet_generation", **params)
    
    def save_result(self, result: Dict[str, Any], output_path: str, 
                   save_additional: bool = True) -> None:
        """
        Save API result to file
        
        Args:
            result: API response result
            output_path: Path to save output image
            save_additional: Whether to save additional images (depth maps, etc.)
        """
        if result.get("status") == "success":
            # Save main image
            output_image = self.base64_to_image(result["image"])
            output_image.save(output_path)
            print(f"Output saved to: {output_path}")
            
            # Save additional images if available
            if save_additional:
                base_name = os.path.splitext(output_path)[0]
                
                if "depth_map" in result:
                    depth_image = self.base64_to_image(result["depth_map"])
                    depth_path = f"{base_name}_depth.png"
                    depth_image.save(depth_path)
                    print(f"Depth map saved to: {depth_path}")
                
                if "canny_edges" in result:
                    canny_image = self.base64_to_image(result["canny_edges"])
                    canny_path = f"{base_name}_canny.png"
                    canny_image.save(canny_path)
                    print(f"Canny edges saved to: {canny_path}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")


def main():
    """Example usage of the FLUX.1 Kontext client"""
    
    # Configuration - Replace with your actual values
    ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
    API_KEY = "YOUR_RUNPOD_API_KEY"
    
    # Initialize client
    client = FluxKontextClient(ENDPOINT_URL, API_KEY)
    
    try:
        # Test model info
        print("Getting model info...")
        info = client.get_model_info()
        print(f"Model info: {json.dumps(info, indent=2)}")
        
        # Example 1a: Fill image with mask (targeted editing)
        if os.path.exists("input.jpg") and os.path.exists("mask.jpg"):
            print("\nExample 1a: Fill image with mask")
            result = client.fill_image(
                image_path="input.jpg",
                prompt="a beautiful sunset over mountains",
                mask_path="mask.jpg",
                num_inference_steps=30,
                seed=42
            )
            client.save_result(result, "output_fill_masked.png")
        
        # Example 1b: Fill image without mask (instruction-based editing like Replicate)
        if os.path.exists("input.jpg"):
            print("\nExample 1b: Fill image without mask (instruction-based)")
            result = client.fill_image(
                image_path="input.jpg",
                prompt="change the sky to a beautiful sunset over mountains",
                num_inference_steps=30,
                seed=42
            )
            client.save_result(result, "output_fill_instruction.png")
        
        # Example 2: Depth controlled generation
        if os.path.exists("input.jpg"):
            print("\nExample 2: Depth controlled generation")
            result = client.depth_controlled_generation(
                image_path="input.jpg",
                prompt="a futuristic cityscape at night",
                controlnet_conditioning_scale=0.7,
                seed=123
            )
            client.save_result(result, "output_depth.png")
        
        # Example 3: Canny controlled generation
        if os.path.exists("input.jpg"):
            print("\nExample 3: Canny controlled generation")
            result = client.canny_controlled_generation(
                image_path="input.jpg",
                prompt="an artistic sketch of a landscape",
                low_threshold=50,
                high_threshold=150,
                seed=456
            )
            client.save_result(result, "output_canny.png")
        
        # Example 4: Multi-ControlNet generation
        if os.path.exists("input.jpg"):
            print("\nExample 4: Multi-ControlNet generation")
            result = client.multi_controlnet_generation(
                image_path="input.jpg",
                prompt="a photorealistic portrait",
                depth_conditioning_scale=0.7,
                canny_conditioning_scale=0.5,
                seed=789
            )
            client.save_result(result, "output_multi.png")
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 