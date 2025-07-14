# FLUX.1 Kontext RunPod Serverless

Complete RunPod serverless deployment for FLUX.1 Kontext image editing models using local diffusers.

## Features

- **Local Processing**: No external API calls or per-request fees
- **Dual Image Editing Approaches**:
  - `instruction_edit`: True FLUX.1 Kontext instruction-based editing (like Replicate)
  - `fill_image`: Traditional inpainting with optional masks
- **ControlNet Support**: Depth and Canny edge control
- **Multi-ControlNet**: Combined depth and edge control
- **GPU Optimized**: Memory efficient attention and SafeTensors
- **No Authentication**: Runs entirely local after model download

## Quick Start

### 1. RunPod Setup

```bash
# Clone repository
git clone https://github.com/703deuce/kon.git
cd kon

# Deploy to RunPod using these files
```

### 2. Model Download Options

**Option A: Download during Docker build (Recommended)**

```bash
# Build with models pre-downloaded
docker build --build-arg HUGGINGFACE_TOKEN=your_hf_token_here -t flux-kontext .
```

**Option B: Download at runtime**

If you didn't build with models, run the setup script in your container:

```bash
# Set environment variable
export HUGGINGFACE_TOKEN=your_hf_token_here

# Download models
python setup_models.py
```

### 3. Environment Variables

**IMPORTANT**: Set these environment variables in your RunPod deployment configuration (not in code or GitHub):

#### Required for Authentication:
```bash
HF_TOKEN=your_huggingface_token_here
```

#### Optional for Optimization:
```bash
TRANSFORMERS_CACHE=/workspace/models
DIFFUSERS_CACHE=/workspace/models
HUGGINGFACE_HUB_CACHE=/workspace/models
```

#### How to Set Environment Variables in RunPod:

1. **Via RunPod Web Interface**: 
   - Go to your endpoint settings
   - Add environment variables in the "Environment Variables" section
   - Set `HF_TOKEN` with your Hugging Face token

2. **Via RunPod API/Config**:
   ```json
   {
     "env": {
       "HF_TOKEN": "your_huggingface_token_here"
     }
   }
   ```

3. **For Testing Locally**:
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```

**Security Note**: Never commit API keys or tokens to GitHub. Always use environment variables set in your deployment platform.

#### Alternative Secure Methods:

4. **GitHub Secrets for CI/CD**:
   ```yaml
   # .github/workflows/deploy.yml
   env:
     HF_TOKEN: ${{ secrets.HF_TOKEN }}
   ```

5. **Configuration Management**:
   - AWS Secrets Manager
   - HashiCorp Vault  
   - Docker secrets
   - Kubernetes secrets

6. **Local .env File** (not committed):
   ```bash
   # Create .env file locally
   echo "HF_TOKEN=your_token_here" > .env
   # Install python-dotenv: pip install python-dotenv
   ```

## API Endpoints

### Instruction Edit (Recommended)

True FLUX.1 Kontext instruction-based editing without masks:

```python
import requests
import base64

# Load your image
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

payload = {
    "input": {
        "endpoint": "instruction_edit",
        "image": image_b64,
        "instruction": "Remove the shower rod and add a shower door",
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "seed": 42
    }
}

response = requests.post(
    "https://api.runpod.ai/v2/your-endpoint-id/run",
    json=payload,
    headers={"Authorization": "Bearer your-api-key"}
)
```

### Fill Image (Traditional Inpainting)

Traditional inpainting with optional masks:

```python
payload = {
    "input": {
        "endpoint": "fill_image",
        "image": image_b64,
        "prompt": "A beautiful shower door",
        "mask": mask_b64,  # Optional - creates full mask if not provided
        "num_inference_steps": 30,
        "guidance_scale": 30.0,
        "strength": 0.8
    }
}
```

## Model Architecture

- **FLUX.1-Fill-dev**: Mask-based inpainting
- **FLUX.1-Kontext-dev**: Instruction-based editing
- **FLUX.1-Depth-dev**: Depth-controlled generation
- **FLUX.1-Canny-dev**: Edge-controlled generation
- **Intel/dpt-hybrid-midas**: Depth estimation

## Performance Benefits

✅ **No API Fees**: All processing runs locally  
✅ **No Authentication**: No per-request HF tokens needed  
✅ **Faster**: No external API calls  
✅ **Private**: Data never leaves your container  
✅ **Reliable**: No external dependencies  

## Memory Optimization

The handler includes several optimizations:

- SafeTensors for faster loading
- Memory efficient attention
- Automatic garbage collection
- GPU memory management

## Troubleshooting

### Model Loading Issues

If you get model loading errors:

1. **First Time Setup**: Models need to be downloaded once with HF authentication
2. **Cache Location**: Ensure cache directories exist in `/workspace/models`
3. **Memory**: Ensure sufficient GPU memory (24GB+ recommended)

### Performance Issues

- Use `torch.compile()` for faster inference
- Enable `use_safetensors=True` for faster loading
- Reduce `num_inference_steps` for faster generation

## Testing

### 🔐 Security First: Never Commit API Keys

**Create test files locally that are NOT committed to GitHub:**

1. **Set Environment Variables**:
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   export RUNPOD_API_KEY=your_runpod_api_key_here
   export RUNPOD_ENDPOINT=https://api.runpod.ai/v2/your_endpoint_id/run
   ```

2. **Create `test_local.py` (in .gitignore)**:
   ```python
   import requests
   import base64
   import os
   
   # Get credentials from environment (NEVER hardcode!)
   hf_token = os.environ.get("HF_TOKEN")
   api_key = os.environ.get("RUNPOD_API_KEY") 
   endpoint = os.environ.get("RUNPOD_ENDPOINT")
   
   # Test instruction editing
   with open("test_image.jpg", "rb") as f:
       image_b64 = base64.b64encode(f.read()).decode()
   
   payload = {
       "input": {
           "endpoint": "instruction_edit",
           "image": image_b64,
           "prompt": "remove the shower rod and add a shower door",
           "guidance_scale": 2.5,
           "num_inference_steps": 30
       }
   }
   
   response = requests.post(endpoint, json=payload, headers={
       "Authorization": f"Bearer {api_key}"
   })
   print(response.json())
   ```

3. **For RunPod Deployment**:
   - Set `HF_TOKEN` in RunPod environment variables
   - No need to pass token in API requests

## Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run handler locally
python handler.py
```

### Configuration

Edit `runpod_config.py` for:
- Model paths
- Default parameters
- Resource limits
- Validation rules

## License

- **Code**: MIT License
- **Models**: Black Forest Labs Non-Commercial License
- **Usage**: Non-commercial use only

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and questions:
- GitHub Issues: Create an issue in this repository
- RunPod Discord: Join the RunPod community
- Documentation: Check RunPod serverless docs 