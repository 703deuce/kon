# FLUX.1 Kontext RunPod Serverless

A complete RunPod serverless deployment for FLUX.1 Kontext models, providing powerful image generation and editing capabilities through multiple API endpoints.

## Features

- **Fill/Inpainting**: Advanced image inpainting using FLUX.1 Fill
- **Depth Control**: Structure-preserving image generation with depth maps
- **Canny Edge Control**: Edge-guided image generation
- **Multi-ControlNet**: Combined depth + canny control for precise generation
- **GPU Optimized**: Efficient memory management and CUDA acceleration
- **Base64 Image Support**: Easy integration with web applications

## Quick Start

### 1. Deploy to RunPod

1. **Fork/Clone this repository**
   ```bash
   git clone https://github.com/703deuce/kon.git
   cd kon
   ```

2. **Create RunPod Serverless Endpoint**
   - Go to [RunPod Console](https://runpod.io/console/serverless)
   - Click "New Endpoint"
   - Configure your endpoint:
     - **Source**: GitHub (connect your forked repo)
     - **Branch**: main
     - **Docker Image**: Leave empty (will use Dockerfile)
     - **GPU**: RTX 4090 or A100 (recommended)
     - **Memory**: 24GB+
     - **Disk**: 50GB+

3. **Environment Variables** (Optional)
   ```
   TRANSFORMERS_CACHE=/workspace/models
   DIFFUSERS_CACHE=/workspace/models
   HUGGINGFACE_HUB_CACHE=/workspace/models
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment (first run may take 15-20 minutes for model downloads)

### 2. Test Your Endpoint

```python
import requests
import base64
import json

# Your RunPod endpoint URL
ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "YOUR_RUNPOD_API_KEY"

# Test model info
response = requests.post(
    ENDPOINT_URL,
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"input": {"endpoint": "get_model_info"}}
)
print(response.json())
```

## API Endpoints

### 1. Fill Image (Inpainting)

**Endpoint**: `fill_image`

Fill or edit parts of an image using a mask.

**Required Parameters**:
- `image`: Base64 encoded input image
- `prompt`: Text description of desired changes

**Optional Parameters**:
- `mask`: Base64 encoded mask image (white = edit, black = keep). If not provided, the model will intelligently edit the entire image based on the prompt (like Replicate)
- `num_inference_steps`: Number of denoising steps (default: 30)
- `guidance_scale`: Prompt adherence strength (default: 30.0)
- `strength`: Inpainting strength (default: 0.8)
- `seed`: Random seed for reproducibility
- `width`: Output width (default: 1024)
- `height`: Output height (default: 1024)

**Example with mask** (targeted editing):
```python
payload = {
    "input": {
        "endpoint": "fill_image",
        "image": "base64_encoded_image",
        "mask": "base64_encoded_mask", 
        "prompt": "a beautiful sunset over mountains",
        "num_inference_steps": 30,
        "guidance_scale": 30.0,
        "seed": 42
    }
}
```

**Example without mask** (instruction-based editing like Replicate):
```python
payload = {
    "input": {
        "endpoint": "fill_image",
        "image": "base64_encoded_image",
        "prompt": "change the sky to a beautiful sunset over mountains",
        "num_inference_steps": 30,
        "guidance_scale": 30.0,
        "seed": 42
    }
}
```

### 2. Depth Controlled Generation

**Endpoint**: `depth_controlled_generation`

Generate images while preserving the depth structure of the input.

**Required Parameters**:
- `image`: Base64 encoded input image
- `prompt`: Text description of desired output

**Optional Parameters**:
- `num_inference_steps`: Number of denoising steps (default: 30)
- `guidance_scale`: Prompt adherence strength (default: 3.5)
- `controlnet_conditioning_scale`: Depth control strength (default: 0.6)
- `seed`: Random seed for reproducibility
- `width`: Output width (default: 1024)
- `height`: Output height (default: 1024)

**Example**:
```python
payload = {
    "input": {
        "endpoint": "depth_controlled_generation",
        "image": "base64_encoded_image",
        "prompt": "a futuristic cityscape at night",
        "controlnet_conditioning_scale": 0.7,
        "seed": 123
    }
}
```

### 3. Canny Edge Controlled Generation

**Endpoint**: `canny_controlled_generation`

Generate images guided by edge detection from the input.

**Required Parameters**:
- `image`: Base64 encoded input image
- `prompt`: Text description of desired output

**Optional Parameters**:
- `num_inference_steps`: Number of denoising steps (default: 30)
- `guidance_scale`: Prompt adherence strength (default: 3.5)
- `controlnet_conditioning_scale`: Edge control strength (default: 0.6)
- `low_threshold`: Canny low threshold (default: 100)
- `high_threshold`: Canny high threshold (default: 200)
- `seed`: Random seed for reproducibility
- `width`: Output width (default: 1024)
- `height`: Output height (default: 1024)

**Example**:
```python
payload = {
    "input": {
        "endpoint": "canny_controlled_generation",
        "image": "base64_encoded_image",
        "prompt": "an artistic sketch of a person",
        "low_threshold": 50,
        "high_threshold": 150,
        "seed": 456
    }
}
```

### 4. Multi-ControlNet Generation

**Endpoint**: `multi_controlnet_generation`

Combine depth and edge control for maximum precision.

**Required Parameters**:
- `image`: Base64 encoded input image
- `prompt`: Text description of desired output

**Optional Parameters**:
- `num_inference_steps`: Number of denoising steps (default: 30)
- `guidance_scale`: Prompt adherence strength (default: 3.5)
- `depth_conditioning_scale`: Depth control strength (default: 0.6)
- `canny_conditioning_scale`: Edge control strength (default: 0.6)
- `low_threshold`: Canny low threshold (default: 100)
- `high_threshold`: Canny high threshold (default: 200)
- `seed`: Random seed for reproducibility
- `width`: Output width (default: 1024)
- `height`: Output height (default: 1024)

**Example**:
```python
payload = {
    "input": {
        "endpoint": "multi_controlnet_generation",
        "image": "base64_encoded_image",
        "prompt": "a photorealistic portrait",
        "depth_conditioning_scale": 0.7,
        "canny_conditioning_scale": 0.5,
        "seed": 789
    }
}
```

### 5. Model Information

**Endpoint**: `get_model_info`

Get information about loaded models and available endpoints.

**Parameters**: None

**Example**:
```python
payload = {
    "input": {
        "endpoint": "get_model_info"
    }
}
```

## Complete Usage Example

```python
import requests
import base64
import json
from PIL import Image
import io

class FluxKontextClient:
    def __init__(self, endpoint_url, api_key):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    def base64_to_image(self, base64_str):
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    
    def call_endpoint(self, endpoint, **kwargs):
        """Make API call to RunPod endpoint"""
        payload = {
            "input": {
                "endpoint": endpoint,
                **kwargs
            }
        }
        
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json=payload
        )
        
        return response.json()
    
    def fill_image(self, image_path, mask_path, prompt, **kwargs):
        """Fill/inpaint image"""
        image_b64 = self.image_to_base64(image_path)
        mask_b64 = self.image_to_base64(mask_path)
        
        return self.call_endpoint(
            "fill_image",
            image=image_b64,
            mask=mask_b64,
            prompt=prompt,
            **kwargs
        )
    
    def depth_controlled_generation(self, image_path, prompt, **kwargs):
        """Generate with depth control"""
        image_b64 = self.image_to_base64(image_path)
        
        return self.call_endpoint(
            "depth_controlled_generation",
            image=image_b64,
            prompt=prompt,
            **kwargs
        )
    
    def canny_controlled_generation(self, image_path, prompt, **kwargs):
        """Generate with edge control"""
        image_b64 = self.image_to_base64(image_path)
        
        return self.call_endpoint(
            "canny_controlled_generation",
            image=image_b64,
            prompt=prompt,
            **kwargs
        )
    
    def multi_controlnet_generation(self, image_path, prompt, **kwargs):
        """Generate with multiple controls"""
        image_b64 = self.image_to_base64(image_path)
        
        return self.call_endpoint(
            "multi_controlnet_generation",
            image=image_b64,
            prompt=prompt,
            **kwargs
        )

# Usage
client = FluxKontextClient(
    endpoint_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    api_key="YOUR_RUNPOD_API_KEY"
)

# Fill image example
result = client.fill_image(
    image_path="input.jpg",
    mask_path="mask.jpg", 
    prompt="a beautiful garden",
    num_inference_steps=30,
    seed=42
)

if result["output"]["status"] == "success":
    # Save result
    output_image = client.base64_to_image(result["output"]["image"])
    output_image.save("output.png")
    print("Image generated successfully!")
else:
    print(f"Error: {result['output']['error']}")
```

## Response Format

All endpoints return a JSON response with the following structure:

**Success Response**:
```json
{
    "status": "success",
    "image": "base64_encoded_result_image",
    "parameters": {
        "prompt": "...",
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "seed": 42,
        "width": 1024,
        "height": 1024
    }
}
```

**Error Response**:
```json
{
    "status": "error",
    "error": "Error message",
    "traceback": "Full error traceback"
}
```

## Resource Requirements

### Recommended GPU Configurations

- **RTX 4090**: 24GB VRAM - Excellent performance
- **A100**: 40GB/80GB VRAM - Best performance
- **RTX 3090**: 24GB VRAM - Good performance
- **A6000**: 48GB VRAM - Excellent for large batches

### Memory Usage

- **Base Models**: ~12GB VRAM
- **Working Memory**: ~8GB VRAM
- **Total Required**: 20GB+ VRAM recommended

## Deployment Tips

1. **Model Caching**: Uncomment the pre-download section in Dockerfile to bake models into the image for faster cold starts.

2. **Scaling**: Configure auto-scaling based on your usage patterns in RunPod console.

3. **Monitoring**: Use RunPod's built-in monitoring or implement custom logging.

4. **Cost Optimization**: Use spot instances for non-critical workloads.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce image size or use smaller models
2. **Timeout**: Increase timeout settings for complex generations
3. **Model Loading**: Ensure sufficient disk space for model downloads
4. **Base64 Errors**: Verify image encoding/decoding

### Debug Mode

Set environment variable `DEBUG=1` for verbose logging:

```python
import os
os.environ["DEBUG"] = "1"
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- **Documentation**: This README
- **Issues**: GitHub Issues
- **RunPod Support**: [RunPod Discord](https://discord.gg/runpod)

## Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for FLUX.1 models
- [RunPod](https://runpod.io/) for serverless infrastructure
- [Hugging Face](https://huggingface.co/) for model hosting and libraries 