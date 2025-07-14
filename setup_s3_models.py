import requests
import json
import os

# RunPod API configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_URL = os.environ.get("RUNPOD_ENDPOINT_URL")
HF_TOKEN = os.environ.get("HF_TOKEN")

# S3 Credentials
S3_CREDENTIALS = {
    "s3_endpoint": os.environ.get("S3_ENDPOINT", "https://s3api-eu-ro-1.runpod.io"),
    "s3_bucket": os.environ.get("S3_BUCKET"),
    "s3_access_key": os.environ.get("S3_ACCESS_KEY"),
    "s3_secret_key": os.environ.get("S3_SECRET_KEY"),
    "s3_region": os.environ.get("S3_REGION", "EU-RO-1")
}

def setup_s3_models():
    """
    Setup S3 with FLUX models by running a special setup job
    This needs to be run on a larger RunPod instance with sufficient disk space
    """
    
    # Check if all required environment variables are set
    if not all([RUNPOD_API_KEY, ENDPOINT_URL, HF_TOKEN, S3_CREDENTIALS["s3_bucket"], 
                S3_CREDENTIALS["s3_access_key"], S3_CREDENTIALS["s3_secret_key"]]):
        print("❌ Missing required environment variables!")
        print("Please set: RUNPOD_API_KEY, RUNPOD_ENDPOINT_URL, HF_TOKEN, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY")
        print("See CONFIG.md for setup instructions")
        return False
    
    # Prepare the payload for model setup
    payload = {
        "input": {
            "endpoint": "setup_models",  # Special endpoint for model setup
            "hf_token": HF_TOKEN,
            **S3_CREDENTIALS
        }
    }
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🚀 Setting up FLUX models in S3...")
    print("⚠️  This requires a RunPod instance with at least 20GB disk space")
    print("🔧 Models will be downloaded and uploaded to S3 for future use")
    
    try:
        # Send the request
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Setup job submitted!")
            
            # If there's a job ID, check the status
            if "id" in result:
                job_id = result["id"]
                print(f"🔍 Job ID: {job_id}")
                
                # Check job status
                status_url = f"{ENDPOINT_URL.replace('/run', '')}/status/{job_id}"
                
                import time
                print("⏳ Checking setup job status...")
                print("⏰ This may take 10-15 minutes to download and upload models...")
                
                max_attempts = 180  # 15 minutes max
                attempt = 0
                
                while attempt < max_attempts:
                    status_response = requests.get(status_url, headers=headers)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get('status', 'unknown')
                        print(f"📊 Status: {status}")
                        
                        if status == "COMPLETED":
                            print("✅ Model setup completed successfully!")
                            output = status_data.get("output", {})
                            print(f"📋 Setup result: {json.dumps(output, indent=2)}")
                            
                            print("\n🎉 S3 is now populated with FLUX models!")
                            print("✨ Future runs will use cached models from S3")
                            print("🚀 You can now run image generation without disk space issues")
                            
                            return True
                        elif status == "FAILED":
                            print("❌ Model setup failed!")
                            error = status_data.get('error', 'Unknown error')
                            print(f"🔍 Error: {error}")
                            
                            if "No space left on device" in error:
                                print("\n💡 SOLUTION: Use a larger RunPod template!")
                                print("🔧 You need at least 20GB disk space for model setup")
                                print("📦 Try a different RunPod template with more storage")
                            
                            return False
                        else:
                            if attempt % 6 == 0:  # Print every 30 seconds
                                print(f"⏳ Still {status}... ({attempt//6 + 1}/30 minutes)")
                            time.sleep(5)
                            attempt += 1
                    else:
                        print(f"❌ Status check failed: {status_response.status_code}")
                        return False
                
                if attempt >= max_attempts:
                    print("⏰ Timeout waiting for setup completion")
                    print("🔍 Check RunPod dashboard for job status")
                    return False
        
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            print(f"🔍 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"💥 Error setting up models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 FLUX Model S3 Setup")
    print("=" * 50)
    print("This script will:")
    print("1. Download FLUX models from HuggingFace")
    print("2. Upload them to your S3 bucket")
    print("3. Enable fast loading for future runs")
    print("")
    print("⚠️  REQUIREMENTS:")
    print("- RunPod instance with 20GB+ disk space")
    print("- Valid HF token for FLUX model access")
    print("- S3 credentials configured")
    print("- All environment variables set (see CONFIG.md)")
    print("")
    
    confirm = input("Continue with setup? (y/N): ")
    if confirm.lower() == 'y':
        success = setup_s3_models()
        if success:
            print("🎉 Setup completed successfully!")
        else:
            print("❌ Setup failed!")
    else:
        print("Setup cancelled") 