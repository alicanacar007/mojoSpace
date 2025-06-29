#!/usr/bin/env python3
"""
Test script to verify MojoX installation
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test basic imports"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch imported successfully (version: {torch.__version__})")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} device(s)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available, will use CPU")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ opencv imported successfully (version: {cv2.__version__})")
    except ImportError as e:
        print(f"⚠️ opencv import failed: {e} (optional)")
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"⚠️ PIL import failed: {e} (optional)")
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        print("✅ Configuration loaded successfully")
        print(f"   Target FPS: {config.frame_extraction.target_fps}")
        print(f"   Model device: {config.model.device}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading"""
    print("\n🧠 Testing model loading...")
    
    try:
        from models.yolo_graph import YOLOGraphModel
        model = YOLOGraphModel()
        print("✅ Model initialized successfully")
        print(f"   Model info: {model.get_model_info()}")
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        traceback.print_exc()
        return False

def test_demo_mode():
    """Test demo mode functionality"""
    print("\n🎮 Testing demo mode...")
    
    try:
        from app import VideoObjectDetectionPipeline
        from utils.config import ConfigManager
        
        config = ConfigManager().get_config()
        config.demo_mode = True
        
        pipeline = VideoObjectDetectionPipeline(config)
        stats = pipeline.process_demo_mode()
        
        print("✅ Demo mode completed successfully")
        print(f"   Processed {stats['total_frames']} frames")
        print(f"   Average FPS: {stats['average_fps']:.1f}")
        return True
    except Exception as e:
        print(f"❌ Demo mode test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 MojoX Installation Test")
    print("=" * 40)
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test configuration
    results.append(test_config())
    
    # Test model loading
    results.append(test_model_loading())
    
    # Test demo mode (skip if dependencies missing)
    try:
        results.append(test_demo_mode())
    except ImportError:
        print("\n⚠️ Skipping demo test due to missing dependencies")
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("\n✅ MojoX is ready to use!")
        print("\nNext steps:")
        print("   python src/app.py --demo")
        print("   python benchmark/benchmark_mojo.py")
        return 0
    else:
        print(f"⚠️ Some tests failed ({passed}/{total})")
        print("\n🔧 Please check the installation and try again:")
        print("   bash scripts/install.sh")
        return 1

if __name__ == "__main__":
    exit(main()) 