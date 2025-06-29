#!/usr/bin/env python3
"""
Enhanced Multi-GPU Video Processing with Mojo Integration
Combines all 8 H100 GPUs with high-performance Mojo frame processing
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import time
import json
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.yolo_graph import YOLOGraphModel
from utils.config import ConfigManager
from utils.visualizer import Visualizer
from utils.cleanup import ensure_clean_start
from mojo_processors.mojo_bridge import HybridVideoProcessor

class MojoMultiGPUProcessor:
    """Enhanced multi-GPU processor with Mojo frame processing"""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        # Force all GPUs to be visible
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        
        self.config = ConfigManager(config_path).config
        self.num_gpus = torch.cuda.device_count()
        self.models = {}
        self.visualizers = {}
        
        # Initialize Mojo hybrid processor
        self.mojo_processor = HybridVideoProcessor(use_mojo=True)
        
        print(f"🚀 Initializing Enhanced Multi-GPU Processor with Mojo")
        print(f"   Available GPUs: {self.num_gpus}")
        print(f"   GPU Memory: {self.num_gpus * 85:.0f}GB total")
        print(f"   Mojo Integration: {'✅ Enabled' if self.mojo_processor.mojo_processor.mojo_available else '❌ Fallback to Python'}")
        
        # Initialize models on each GPU
        for gpu_id in range(self.num_gpus):
            self.models[gpu_id] = YOLOGraphModel(
                model_path=self.config.model.model_path,
                device=f'cuda:{gpu_id}'
            )
            self.visualizers[gpu_id] = Visualizer(self.config)
            print(f"   GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    def extract_frames_every_second(self, video_path: str, output_dir: str) -> List[Tuple[int, np.ndarray]]:
        """Extract frames every second from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        frame_count = 0
        second_count = 0
        
        print(f"📹 Extracting frames every second (FPS: {fps:.1f})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame every second (every fps frames)
            if frame_count % int(fps) == 0:
                frames.append((second_count, frame.copy()))
                second_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"   Extracted {len(frames)} frames from {second_count} seconds")
        return frames
    
    def process_frame_batch_with_mojo(self, gpu_id: int, batch_data: List[Tuple[int, np.ndarray]], 
                                      output_dir: str) -> Dict:
        """Process a batch of frames with Mojo enhancement on specific GPU"""
        model = self.models[gpu_id]
        visualizer = self.visualizers[gpu_id]
        
        batch_results = {
            'gpu_id': gpu_id,
            'frames_processed': 0,
            'processing_time': 0,
            'mojo_operations': 0,
            'python_operations': 0
        }
        
        start_time = time.perf_counter()
        
        for second, frame in batch_data:
            try:
                # Apply Mojo frame enhancements
                enhanced_frame, mojo_stats = self.mojo_processor.process_frame_enhanced(
                    frame,
                    enhance_contrast=True,
                    apply_gamma=True,
                    blur_kernel=None  # Skip blur for speed
                )
                
                # Run object detection on enhanced frame
                detections_list = model.inference([enhanced_frame])
                detections = detections_list[0] if detections_list else []
                
                # Create annotated version using the available function
                from utils.visualizer import create_detection_overlay
                annotated_frame = create_detection_overlay(enhanced_frame, detections)
                
                # Save both versions
                frame_name = f"frame_{second:04d}_gpu{gpu_id}.jpg"
                annotated_name = f"annotated_{second:04d}_gpu{gpu_id}.jpg"
                
                cv2.imwrite(os.path.join(output_dir, frame_name), enhanced_frame)
                cv2.imwrite(os.path.join(output_dir, annotated_name), annotated_frame)
                
                batch_results['frames_processed'] += 1
                batch_results['mojo_operations'] += len(mojo_stats['operations_used'])
                
                print(f"   GPU {gpu_id}: Frame {second:04d} - {len(detections)} objects ({mojo_stats['backend']})")
                
            except Exception as e:
                print(f"   GPU {gpu_id}: Error processing frame {second}: {e}")
        
        processing_time = time.perf_counter() - start_time
        batch_results['processing_time'] = processing_time
        
        return batch_results
    
    def process_video_distributed(self, video_path: str, output_dir: str) -> Dict:
        """Process video with distributed multi-GPU + Mojo pipeline"""
        print(f"🎬 Processing video: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean existing images from output directory
        cleaned_files = ensure_clean_start(output_dir)
        if cleaned_files > 0:
            print(f"🧹 Cleaned {cleaned_files} existing files from {output_dir}")
        
        # Extract frames every second
        frames = self.extract_frames_every_second(video_path, output_dir)
        
        if not frames:
            print("❌ No frames extracted!")
            return {}
        
        # Distribute frames across GPUs
        frames_per_gpu = len(frames) // self.num_gpus
        gpu_batches = []
        
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * frames_per_gpu
            if gpu_id == self.num_gpus - 1:  # Last GPU gets remaining frames
                end_idx = len(frames)
            else:
                end_idx = (gpu_id + 1) * frames_per_gpu
            
            gpu_batch = frames[start_idx:end_idx]
            gpu_batches.append((gpu_id, gpu_batch))
            print(f"   GPU {gpu_id}: {len(gpu_batch)} frames assigned")
        
        # Process batches in parallel across all GPUs
        print(f"⚡ Processing {len(frames)} frames across {self.num_gpus} GPUs...")
        start_time = time.perf_counter()
        
        results = []
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {
                executor.submit(
                    self.process_frame_batch_with_mojo, 
                    gpu_id, 
                    batch, 
                    output_dir
                ): gpu_id for gpu_id, batch in gpu_batches
            }
            
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ GPU {gpu_id}: {result['frames_processed']} frames completed")
                except Exception as e:
                    print(f"❌ GPU {gpu_id} error: {e}")
        
        total_time = time.perf_counter() - start_time
        
        # Aggregate results
        summary = {
            'total_frames': len(frames),
            'total_time': total_time,
            'fps': len(frames) / total_time,
            'gpu_results': results,
            'mojo_performance': self.mojo_processor.get_performance_summary()
        }
        
        # Save summary
        with open(os.path.join(output_dir, 'mojo_processing_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="🔥 Multi-GPU Video Processor with Mojo Integration")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", default="results/mojo_images", help="Output directory for frames")
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Video file '{args.input}' not found!")
        return
    
    # Initialize processor
    processor = MojoMultiGPUProcessor()
    
    # Process video
    results = processor.process_video_distributed(args.input, args.output)
    
    if results:
        print("\n🏆 PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"📊 Performance Summary:")
        print(f"   Total Frames: {results['total_frames']}")
        print(f"   Processing Time: {results['total_time']:.2f}s")
        print(f"   Throughput: {results['fps']:.1f} FPS")
        print(f"   GPU Utilization: {len(results['gpu_results'])}/{processor.num_gpus} GPUs")
        
        # Mojo performance summary
        mojo_perf = results['mojo_performance']
        print(f"   Mojo Operations: {mojo_perf['mojo_operations']}")
        print(f"   Python Fallback: {mojo_perf['python_operations']}")
        print(f"   Mojo Available: {'✅' if mojo_perf['mojo_available'] else '❌'}")
        
        print(f"\n📁 Results saved to: {args.output}")
        print(f"   Original frames: {results['total_frames']}")
        print(f"   Annotated frames: {results['total_frames']}")
        print(f"   JSON report: mojo_processing_results.json")
    
    print("\n🚀 Multi-GPU + Mojo processing completed!")

if __name__ == "__main__":
    main() 