#!/usr/bin/env python3
"""
Basic MojoX Environment Test
"""

import sys
import os

print("🚀 MojoX Basic Environment Test")
print("=" * 50)

# Test 1: Python Environment
print(f"✅ Python Version: {sys.version}")
print(f"✅ Working Directory: {os.getcwd()}")

# Test 2: Core Dependencies
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU Count: {torch.cuda.device_count()}")
        print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    from PIL import Image
    print(f"✅ PIL/Pillow: {Image.__version__}")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")

# Test 3: Project Structure
print("\n📁 Project Structure:")
required_dirs = ['src', 'src/kernels', 'src/models', 'src/utils', 'config', 'demos']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"✅ {dir_name}/")
    else:
        print(f"❌ {dir_name}/ - Missing")

# Test 4: Configuration
print("\n⚙️ Testing Configuration:")
try:
    sys.path.insert(0, 'src')
    from utils.config import ConfigManager
    config = ConfigManager()
    config.load_config('config/default.yaml')
    print("✅ Configuration loaded successfully")
    print(f"   Target FPS: {config.get('frame_extraction.target_fps', 'N/A')}")
    print(f"   Model Device: {config.get('model.device', 'N/A')}")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")

# Test 5: Mojo availability
print("\n🔥 Testing Mojo:")
if os.system("which mojo > /dev/null 2>&1") == 0:
    print("✅ Mojo is available")
    os.system("mojo --version")
else:
    print("❌ Mojo not found in PATH")

print("\n" + "=" * 50)
print("🎯 Basic test complete!") 