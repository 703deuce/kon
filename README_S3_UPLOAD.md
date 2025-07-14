# FLUX Model S3 Uploader

This standalone script downloads FLUX models directly and uploads them to your S3 bucket. **No RunPod endpoint required!**

## Benefits

✅ **No disk space limits** - Run locally or on any machine with sufficient space  
✅ **Direct upload** - Bypasses RunPod API completely  
✅ **One-time setup** - Once uploaded, all RunPod endpoints can use cached models  
✅ **Fast and efficient** - Direct S3 transfer without intermediate steps  

## Quick Setup

### 1. Set Environment Variables

```bash
# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token_here"
$env:S3_BUCKET="your_s3_bucket_name"
$env:S3_ACCESS_KEY="your_s3_access_key"
$env:S3_SECRET_KEY="your_s3_secret_key"
$env:S3_ENDPOINT="https://s3api-eu-ro-1.runpod.io"
$env:S3_REGION="EU-RO-1"

# Linux/macOS
export HF_TOKEN="your_huggingface_token_here"
export S3_BUCKET="your_s3_bucket_name"
export S3_ACCESS_KEY="your_s3_access_key"
export S3_SECRET_KEY="your_s3_secret_key"
export S3_ENDPOINT="https://s3api-eu-ro-1.runpod.io"
export S3_REGION="EU-RO-1"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Uploader

```bash
python upload_models_to_s3.py
```

## What It Does

1. **Downloads FLUX models** from HuggingFace:
   - `black-forest-labs/FLUX.1-Kontext-dev` (instruction editing)
   - `black-forest-labs/FLUX.1-Fill-dev` (inpainting)

2. **Uploads to S3** in the correct directory structure

3. **Cleans up** temporary files automatically

## Requirements

- **20GB+ free disk space** (temporary, cleaned up after upload)
- **Valid HuggingFace token** with FLUX model access
- **S3 credentials** for your RunPod bucket
- **Internet connection**

## After Upload

Once the models are in S3, your RunPod endpoints will:

✅ **Sync models from S3** on startup (fast)  
✅ **Use local cache** for model loading  
✅ **No more disk space errors**  
✅ **Fast startup times** after first sync  

## Troubleshooting

### Authentication Error
- Ensure your HF token has access to FLUX models
- Accept the model licenses on HuggingFace

### S3 Connection Error
- Check your S3 credentials and endpoint
- Verify bucket name and permissions

### Disk Space Error
- Free up at least 20GB of disk space
- The script will clean up automatically after upload

## Advanced Usage

### Auto mode (no prompts)
```bash
python upload_models_to_s3.py --auto
```

### Check what's already uploaded
The script automatically skips files that already exist in S3.

## File Structure in S3

After upload, your S3 bucket will contain:
```
models/
├── black-forest-labs--FLUX.1-Kontext-dev/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── black-forest-labs--FLUX.1-Fill-dev/
    ├── config.json
    ├── model.safetensors
    └── ...
```

This matches exactly what the RunPod handler expects for S3 sync! 🎉 