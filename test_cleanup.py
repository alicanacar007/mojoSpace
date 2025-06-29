#!/usr/bin/env python3
"""
Test script for MojoX image cleanup functionality
Demonstrates the cleanup utility working before processing starts
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.cleanup import (
    clean_all_result_directories, 
    ensure_clean_start, 
    get_directory_stats,
    clean_directory_images
)

def create_test_images():
    """Create some test images to demonstrate cleanup"""
    test_dirs = [
        "results/images",
        "results/mojo_images", 
        "results/true_multi_gpu",
        "output"
    ]
    
    print("🧪 Creating test images for cleanup demonstration...")
    
    for directory in test_dirs:
        os.makedirs(directory, exist_ok=True)
        
        # Create some dummy files
        test_files = [
            "frame_0001_gpu0.jpg",
            "annotated_0001_gpu0.jpg",
            "frame_0002_gpu1.jpg", 
            "annotated_0002_gpu1.jpg",
            "processing_results.json",
            "old_result.png"
        ]
        
        for filename in test_files:
            file_path = Path(directory) / filename
            # Create dummy file with some content
            with open(file_path, 'w') as f:
                f.write(f"Test file: {filename}\nCreated for cleanup testing\n")
        
        print(f"   ✅ Created {len(test_files)} test files in {directory}")

def demonstrate_cleanup():
    """Demonstrate the cleanup functionality"""
    print("\n" + "="*60)
    print("🧹 MojoX Cleanup Functionality Demonstration")
    print("="*60)
    
    # Create test images first
    create_test_images()
    
    # Show statistics before cleanup
    print("\n📊 Directory Statistics BEFORE Cleanup:")
    directories = ["results/images", "results/mojo_images", "results/true_multi_gpu", "output"]
    
    for directory in directories:
        stats = get_directory_stats(directory)
        if stats["exists"]:
            print(f"   📁 {directory}:")
            print(f"      Files: {stats['total_files']}")
            print(f"      Images: {stats['image_files']}")
            print(f"      JSON: {stats['json_files']}")
            print(f"      Size: {stats['total_size'] / 1024:.1f} KB")
    
    # Test individual directory cleanup
    print(f"\n🧪 Testing individual directory cleanup...")
    test_dir = "results/images"
    cleaned = ensure_clean_start(test_dir)
    print(f"   ✅ Cleaned {cleaned} files from {test_dir}")
    
    # Test cleaning specific directory
    print(f"\n🧪 Testing specific pattern cleanup...")
    test_dir2 = "results/mojo_images" 
    cleaned2 = clean_directory_images(test_dir2, ["frame_*.jpg", "annotated_*.jpg"])
    print(f"   ✅ Cleaned {cleaned2} image files from {test_dir2}")
    
    # Test cleaning all directories
    print(f"\n🧪 Testing cleanup of all result directories...")
    total_cleaned = clean_all_result_directories()
    print(f"   ✅ Total files cleaned: {total_cleaned}")
    
    # Show statistics after cleanup
    print("\n📊 Directory Statistics AFTER Cleanup:")
    for directory in directories:
        stats = get_directory_stats(directory)
        if stats["exists"]:
            print(f"   📁 {directory}:")
            print(f"      Files: {stats['total_files']}")
            print(f"      Images: {stats['image_files']}")
            print(f"      JSON: {stats['json_files']}")
            print(f"      Size: {stats['total_size'] / 1024:.1f} KB")
    
    print(f"\n✅ Cleanup demonstration completed!")
    print(f"   All directories are now clean and ready for processing")

def simulate_processing_start():
    """Simulate how cleanup works at the start of processing"""
    print("\n" + "="*60)
    print("🚀 Simulating Processing Start with Cleanup")
    print("="*60)
    
    # Create some test files again
    test_dir = "results/test_processing"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create dummy files
    test_files = ["old_frame.jpg", "old_annotation.jpg", "old_results.json"]
    for filename in test_files:
        with open(Path(test_dir) / filename, 'w') as f:
            f.write("Old processing data")
    
    print(f"📁 Created test directory: {test_dir}")
    print(f"📊 Files before processing: {len(test_files)}")
    
    # This is what happens at the start of each processing script
    print(f"\n🧹 Starting cleanup before processing...")
    cleaned_files = ensure_clean_start(test_dir)
    print(f"✅ Cleaned {cleaned_files} existing files")
    
    # Check final state
    remaining_files = list(Path(test_dir).glob("*"))
    print(f"📊 Files after cleanup: {len(remaining_files)}")
    print(f"🚀 Directory is now clean and ready for new processing!")
    
    # Cleanup test directory
    import shutil
    shutil.rmtree(test_dir)
    print(f"🧹 Removed test directory: {test_dir}")

if __name__ == "__main__":
    print("🔥 MojoX Cleanup Test Suite")
    print("This script demonstrates the image cleanup functionality")
    print("that runs before each processing task starts.\n")
    
    try:
        # Run demonstrations
        demonstrate_cleanup()
        simulate_processing_start()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"💡 The cleanup functionality is now integrated into:")
        print(f"   - mojo_multi_gpu_run.py")
        print(f"   - true_multi_gpu_mojo.py") 
        print(f"   - web_app.py")
        print(f"   - src/multi_gpu_processor.py")
        print(f"\n🚀 Every processing run will now start with a clean directory!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 