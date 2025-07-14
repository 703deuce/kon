#!/usr/bin/env python3
"""
Standalone script to download FLUX models and upload them directly to S3.
This script can be run locally or on any machine with sufficient disk space.
No RunPod endpoint required - just direct S3 upload.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import boto3
from botocore.client import Config
from huggingface_hub import snapshot_download, login
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3api-eu-ro-1.runpod.io")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_REGION = os.environ.get("S3_REGION", "EU-RO-1")

# Models to download and upload
FLUX_MODELS = [
    "black-forest-labs/FLUX.1-Kontext-dev",
    "black-forest-labs/FLUX.1-Fill-dev",
]

class FluxModelUploader:
    def __init__(self):
        self.s3_client = None
        self.temp_dir = None
        
    def check_requirements(self):
        """Check if all required environment variables are set"""
        missing = []
        if not HF_TOKEN:
            missing.append("HF_TOKEN")
        if not S3_BUCKET:
            missing.append("S3_BUCKET")
        if not S3_ACCESS_KEY:
            missing.append("S3_ACCESS_KEY")
        if not S3_SECRET_KEY:
            missing.append("S3_SECRET_KEY")
            
        if missing:
            logger.error(f"❌ Missing required environment variables: {', '.join(missing)}")
            logger.error("Please set these environment variables before running the script")
            return False
        
        return True
    
    def setup_s3_client(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
                config=Config(signature_version='s3v4'),
                region_name=S3_REGION
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=S3_BUCKET)
            logger.info(f"✅ S3 connection successful: {S3_BUCKET}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to S3: {e}")
            return False
    
    def authenticate_hf(self):
        """Authenticate with HuggingFace"""
        try:
            login(token=HF_TOKEN)
            logger.info("✅ HuggingFace authentication successful")
            return True
        except Exception as e:
            logger.error(f"❌ HuggingFace authentication failed: {e}")
            return False
    
    def check_disk_space(self, required_gb=20):
        """Check available disk space"""
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < required_gb:
            logger.error(f"❌ Insufficient disk space: {free_gb:.1f}GB available, need {required_gb}GB")
            return False
        
        logger.info(f"✅ Sufficient disk space: {free_gb:.1f}GB available")
        return True
    
    def download_model(self, model_name):
        """Download a FLUX model from HuggingFace"""
        logger.info(f"📥 Downloading {model_name}...")
        
        try:
            # Create temp directory for this model
            model_dir = os.path.join(self.temp_dir, "models", model_name.replace("/", "--"))
            
            # Download model
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                token=HF_TOKEN,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"✅ Downloaded {model_name} to {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            return None
    
    def upload_directory_to_s3(self, local_dir, s3_prefix):
        """Upload a directory recursively to S3"""
        logger.info(f"📤 Uploading {local_dir} to S3...")
        
        uploaded_files = 0
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Calculate relative path for S3 key
                rel_path = os.path.relpath(local_path, self.temp_dir)
                s3_key = rel_path.replace("\\", "/")  # Ensure forward slashes for S3
                
                try:
                    # Check if file already exists in S3
                    try:
                        self.s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                        logger.debug(f"⏭️  Skipping {s3_key} (already exists)")
                        continue
                    except:
                        pass  # File doesn't exist, upload it
                    
                    # Upload file
                    logger.info(f"📤 Uploading {s3_key}")
                    self.s3_client.upload_file(local_path, S3_BUCKET, s3_key)
                    uploaded_files += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to upload {s3_key}: {e}")
                    return False
        
        logger.info(f"✅ Uploaded {uploaded_files} files to S3")
        return True
    
    def upload_models(self):
        """Main function to download and upload all FLUX models"""
        logger.info("🚀 Starting FLUX model upload to S3...")
        
        # Check requirements
        if not self.check_requirements():
            return False
        
        # Check disk space
        if not self.check_disk_space():
            return False
        
        # Setup S3
        if not self.setup_s3_client():
            return False
        
        # Authenticate HF
        if not self.authenticate_hf():
            return False
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="flux_models_")
        logger.info(f"📁 Using temporary directory: {self.temp_dir}")
        
        try:
            successful_uploads = []
            failed_uploads = []
            
            for model_name in FLUX_MODELS:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {model_name}")
                logger.info(f"{'='*60}")
                
                # Download model
                model_dir = self.download_model(model_name)
                if not model_dir:
                    failed_uploads.append(model_name)
                    continue
                
                # Upload to S3
                if self.upload_directory_to_s3(model_dir, f"models/{model_name}"):
                    successful_uploads.append(model_name)
                    logger.info(f"✅ {model_name} uploaded successfully")
                else:
                    failed_uploads.append(model_name)
                    logger.error(f"❌ {model_name} upload failed")
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info(f"UPLOAD SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"✅ Successful: {len(successful_uploads)}")
            logger.info(f"❌ Failed: {len(failed_uploads)}")
            
            if successful_uploads:
                logger.info(f"✅ Successfully uploaded models:")
                for model in successful_uploads:
                    logger.info(f"   - {model}")
            
            if failed_uploads:
                logger.error(f"❌ Failed to upload models:")
                for model in failed_uploads:
                    logger.error(f"   - {model}")
            
            return len(failed_uploads) == 0
            
        finally:
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                logger.info(f"🧹 Cleaning up temporary directory: {self.temp_dir}")
                shutil.rmtree(self.temp_dir)

def main():
    print("🎯 FLUX Model S3 Uploader")
    print("=" * 50)
    print("This script will:")
    print("1. Download FLUX models from HuggingFace")
    print("2. Upload them directly to your S3 bucket")
    print("3. Enable fast loading for RunPod endpoints")
    print("")
    print("📋 Required environment variables:")
    print("   - HF_TOKEN: HuggingFace token with FLUX access")
    print("   - S3_BUCKET: S3 bucket name")
    print("   - S3_ACCESS_KEY: S3 access key")
    print("   - S3_SECRET_KEY: S3 secret key")
    print("   - S3_ENDPOINT: S3 endpoint URL (optional)")
    print("   - S3_REGION: S3 region (optional)")
    print("")
    print("⚠️  Requirements:")
    print("   - 20GB+ free disk space")
    print("   - Internet connection")
    print("   - Valid HuggingFace token with FLUX model access")
    print("")
    
    # Check if running with required args
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        confirm = "y"
    else:
        confirm = input("Continue with upload? (y/N): ")
    
    if confirm.lower() == 'y':
        uploader = FluxModelUploader()
        success = uploader.upload_models()
        
        if success:
            print("\n🎉 All models uploaded successfully!")
            print("✨ Your RunPod endpoint can now use S3-cached models")
            print("🚀 No more disk space issues!")
        else:
            print("\n❌ Some uploads failed!")
            print("🔍 Check the logs above for details")
            sys.exit(1)
    else:
        print("Upload cancelled")

if __name__ == "__main__":
    main() 