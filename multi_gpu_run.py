#!/usr/bin/env python3
"""
Multi-GPU Video Processing Launcher
Easy way to upload and process videos with all 8 H100 GPUs
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="🚀 Multi-GPU Video Processor (8x H100)")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", default="results/images", help="Output directory for frames")
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Video file '{args.input}' not found!")
        print("📁 Available demo videos:")
        demo_dir = Path("demos")
        if demo_dir.exists():
            for video_file in demo_dir.glob("*.mp4"):
                print(f"   - {video_file}")
        else:
            print("   - No demo videos found. Upload a video to get started!")
        return
    
    print(f"🎬 Processing video: {args.input}")
    print(f"💾 Output directory: {args.output}")
    print(f"🔥 Using all 8 NVIDIA H100 80GB GPUs")
    print(f"⚡ Extracting frames every second with distributed processing")
    print()
    
    # Run the multi-GPU processor
    os.system(f"python3 src/multi_gpu_processor.py --input {args.input} --output {args.output}")

if __name__ == "__main__":
    main() 