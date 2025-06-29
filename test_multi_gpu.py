#!/usr/bin/env python3
"""
Test Multi-GPU Video Processing
Quick test to verify all 8 H100 GPUs are being used for video processing
"""

import sys
import os
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_gpu_detection():
    """Test GPU detection and setup"""
    print("🔥 Testing GPU Detection")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPUs")
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {name} ({memory:.0f}GB)")
    
    return gpu_count >= 8

def test_multi_gpu_processor():
    """Test the multi-GPU processor initialization"""
    print("\n🚀 Testing Multi-GPU Processor")
    print("=" * 50)
    
    try:
        from multi_gpu_processor import MultiGPUVideoProcessor
        from utils.config import ConfigManager
        
        # Load config
        config_manager = ConfigManager("config/default.yaml")
        config = config_manager.get_config()
        
        # Initialize multi-GPU processor
        print("🔧 Initializing Multi-GPU processor...")
        processor = MultiGPUVideoProcessor(config)
        
        print(f"✅ Successfully initialized with {processor.gpu_count} GPUs")
        print(f"📊 Batch size per GPU: {processor.batch_size_per_gpu}")
        print(f"📊 Total batch size: {processor.total_batch_size}")
        
        # Test GPU memory usage
        print("\n💾 GPU Memory Usage:")
        gpu_memory = processor.get_gpu_memory_usage()
        for gpu_id, memory_info in gpu_memory.items():
            print(f"  GPU {gpu_id}: {memory_info['usage_percent']:.1f}% "
                  f"({memory_info['allocated_gb']:.1f}GB / {memory_info['total_gb']:.0f}GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing multi-GPU processor: {e}")
        return False

def test_demo_video_processing():
    """Test processing with a demo video"""
    print("\n🎬 Testing Demo Video Processing")
    print("=" * 50)
    
    # Check for demo videos
    demos_path = Path("demos")
    if not demos_path.exists():
        print("❌ Demos directory not found")
        return False
    
    demo_videos = list(demos_path.glob("*.mp4"))
    if not demo_videos:
        print("❌ No demo videos found")
        return False
    
    demo_video = demo_videos[0]
    print(f"📹 Using demo video: {demo_video}")
    
    try:
        from multi_gpu_processor import MultiGPUVideoProcessor
        from utils.config import ConfigManager
        
        # Load config
        config_manager = ConfigManager("config/default.yaml")
        config = config_manager.get_config()
        
        # Initialize processor
        processor = MultiGPUVideoProcessor(config)
        
        # Create output path
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"test_multi_gpu_{int(time.time())}.mp4"
        
        print(f"🚀 Starting multi-GPU processing...")
        print(f"   Input: {demo_video}")
        print(f"   Output: {output_path}")
        
        # Progress callback
        def progress_callback(progress_pct, frames_processed, total_detections):
            print(f"   📊 Progress: {progress_pct:.1f}% - {frames_processed} frames, {total_detections} detections")
        
        # Process video
        start_time = time.time()
        stats = processor.process_video_parallel(
            video_path=str(demo_video),
            output_path=str(output_path),
            progress_callback=progress_callback
        )
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processing completed in {processing_time:.2f}s")
        print(f"📊 Stats: {stats['total_frames']} frames at {stats['average_fps']:.1f} FPS")
        print(f"🎯 Total detections: {stats['total_detections']}")
        print(f"⚡ Speedup factor: {stats.get('speedup_factor', 'N/A'):.1f}x")
        print(f"🚀 GPUs used: {stats['gpu_count']}")
        
        if output_path.exists():
            print(f"✅ Output video saved: {output_path}")
            return True
        else:
            print("❌ Output video not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 MojoX Multi-GPU Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: GPU Detection
    if test_gpu_detection():
        tests_passed += 1
        print("✅ Test 1: GPU Detection - PASSED")
    else:
        print("❌ Test 1: GPU Detection - FAILED")
    
    # Test 2: Multi-GPU Processor
    if test_multi_gpu_processor():
        tests_passed += 1
        print("✅ Test 2: Multi-GPU Processor - PASSED")
    else:
        print("❌ Test 2: Multi-GPU Processor - FAILED")
    
    # Test 3: Demo Video Processing
    if test_demo_video_processing():
        tests_passed += 1
        print("✅ Test 3: Demo Video Processing - PASSED")
    else:
        print("❌ Test 3: Demo Video Processing - FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your 8x H100 setup is ready for multi-GPU video processing!")
        print("\n🚀 Next steps:")
        print("   1. Run: ./start_web.sh")
        print("   2. Go to: http://localhost:8501")
        print("   3. Enable 'Use All 8 H100 GPUs' checkbox")
        print("   4. Upload and process videos with 8x GPU acceleration!")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 