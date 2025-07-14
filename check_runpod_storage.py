#!/usr/bin/env python3
"""
RunPod Network Volume Storage Inspector
Run this script on your RunPod serverless container to check what's stored on the network volume.
Network volume is mounted at /runpod-volume in serverless endpoints.
"""

import os
import subprocess
import json
import time
from pathlib import Path
import shutil

def get_dir_size(path):
    """Get directory size in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception:
        pass
    return total

def format_bytes(bytes_value):
    """Format bytes to human readable format"""
    if bytes_value == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while bytes_value >= 1024 and i < len(units) - 1:
        bytes_value /= 1024
        i += 1
    
    return f"{bytes_value:.2f} {units[i]}"

def check_network_volume_usage():
    """Check network volume disk usage specifically"""
    print("💾 NETWORK VOLUME DISK USAGE (/runpod-volume)")
    print("=" * 50)
    
    try:
        result = subprocess.run(['df', '-h', '/runpod-volume'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Could not get network volume usage")
    except Exception as e:
        print(f"❌ Error getting network volume usage: {e}")

def check_disk_usage():
    """Check overall disk usage on RunPod container"""
    print("🖥️  CONTAINER DISK USAGE (all filesystems)")
    print("=" * 50)
    
    try:
        result = subprocess.run(['df', '-h'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Could not get disk usage with df command")
    except Exception as e:
        print(f"❌ Error getting disk usage: {e}")

def check_network_volume():
    """Check for network volume mounts"""
    print("🔗 NETWORK VOLUME MOUNT INFO")
    print("=" * 30)
    
    try:
        result = subprocess.run(['mount'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            network_mounts = []
            for line in lines:
                if '/runpod-volume' in line or any(keyword in line.lower() for keyword in ['nfs', 'network', 'volume']):
                    network_mounts.append(line)
            
            if network_mounts:
                for mount in network_mounts:
                    print(mount)
            else:
                print("No network volume mounts found")
                print("\nAll mounts:")
                for line in lines[:15]:  # Show first 15 mounts
                    if line.strip():
                        print(line)
        else:
            print("❌ Could not get mount information")
    except Exception as e:
        print(f"❌ Error getting mount info: {e}")

def check_directory_contents(path, max_depth=3, max_files=20):
    """Check contents of a directory recursively"""
    if not os.path.exists(path):
        return f"❌ Directory does not exist: {path}"
    
    contents = []
    try:
        for root, dirs, files in os.walk(path):
            # Calculate depth
            level = root.replace(path, '').count(os.sep)
            if level >= max_depth:
                dirs.clear()  # Don't go deeper
                continue
            
            indent = "  " * level
            folder_name = os.path.basename(root) if root != path else os.path.basename(path) or "root"
            
            # Get directory size for smaller directories only
            dir_size = 0
            if level < 2:  # Only calculate size for first 2 levels
                dir_size = get_dir_size(root)
            
            size_str = f" ({format_bytes(dir_size)})" if dir_size > 0 else ""
            
            contents.append(f"{indent}📁 {folder_name}/{size_str}")
            
            # Show files in current directory
            sub_indent = "  " * (level + 1)
            files_to_show = files[:max_files]
            for file in files_to_show:
                try:
                    file_path = os.path.join(root, file)
                    file_size = format_bytes(os.path.getsize(file_path))
                    contents.append(f"{sub_indent}📄 {file} ({file_size})")
                except Exception:
                    contents.append(f"{sub_indent}📄 {file} (size unknown)")
            
            if len(files) > max_files:
                contents.append(f"{sub_indent}... and {len(files) - max_files} more files")
    
    except Exception as e:
        contents.append(f"❌ Error reading directory: {e}")
    
    return "\n".join(contents)

def main():
    """Main function to check RunPod network storage"""
    print("🚀 RUNPOD SERVERLESS NETWORK VOLUME INSPECTOR")
    print("=" * 60)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # Check network volume usage first (most important)
    check_network_volume_usage()
    print()
    
    # Check network volume mounts
    check_network_volume()
    print()
    
    # Check overall disk usage
    check_disk_usage()
    print()
    
    # Check directories - focus on network volume first
    directories_to_check = [
        ("/runpod-volume", "🔥 Network Volume (MAIN STORAGE)"),
        ("/workspace", "Workspace Directory (if exists)"),
        ("/runpod-volume/huggingface", "Model Cache Directory (Network Volume)"),
        ("/tmp", "Temporary Directory"),
        ("/root", "Root Home Directory"),
        (".", "Current Directory"),
    ]
    
    for dir_path, description in directories_to_check:
        print(f"📂 {description.upper()}")
        print("-" * 40)
        
        if os.path.exists(dir_path):
            try:
                total_size = get_dir_size(dir_path)
                
                print(f"Path: {dir_path}")
                print(f"Total Size: {format_bytes(total_size)}")
                print(f"Permissions: {oct(os.stat(dir_path).st_mode)[-3:]}")
                print(f"Contents:")
                print(check_directory_contents(dir_path))
                
            except Exception as e:
                print(f"❌ Error checking {dir_path}: {e}")
        else:
            print(f"❌ Directory does not exist: {dir_path}")
        
        print()
    
    # Check environment variables
    print("🔧 STORAGE ENVIRONMENT VARIABLES")
    print("-" * 40)
    
    env_vars = [
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE", 
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
        "MODEL_CACHE_DIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "HOME",
        "PWD",
        "RUNPOD_VOLUME_MOUNT_PATH",
        "RUNPOD_VOLUME_PATH"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    print()
    
    # Show summary
    print("📋 SUMMARY")
    print("-" * 15)
    
    if os.path.exists("/runpod-volume"):
        volume_size = get_dir_size("/runpod-volume")
        print(f"✅ Network Volume: {format_bytes(volume_size)}")
        
        # Count items in network volume
        try:
            items = os.listdir("/runpod-volume")
            print(f"📁 Items in network volume: {len(items)}")
            for item in items[:10]:  # Show first 10 items
                print(f"  - {item}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more items")
        except Exception as e:
            print(f"❌ Could not list network volume contents: {e}")
    else:
        print("❌ Network volume not found at /runpod-volume")

if __name__ == "__main__":
    main() 