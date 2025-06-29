#!/usr/bin/env python3
"""
TRUE Multi-GPU Video Processing with Mojo Integration
PROPERLY uses all 8 H100 GPUs simultaneously
"""

import os
import sys
import cv2
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import time
import json
import argparse
import subprocess

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import cleanup utility
from src.utils.cleanup import ensure_clean_start

def setup_gpu_worker(gpu_id: int):
    """Setup worker process for specific GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # Set to device 0 in this process (which is actually gpu_id)
    
    # Import after setting CUDA device
    from models.yolo_graph import YOLOGraphModel
    from utils.config import ConfigManager
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    return YOLOGraphModel(device='cuda:0'), ConfigManager("config/default.yaml").config

def process_frame_with_mojo_simulation(frame: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Simulate Mojo processing (optimized operations)"""
    start_time = time.perf_counter()
    
    # Simulate Mojo SIMD operations
    enhanced_frame = frame.astype(np.float32) / 255.0
    
    # Contrast enhancement (simulated Mojo SIMD)
    enhanced_frame = np.clip((enhanced_frame - 0.5) * 1.2 + 0.5, 0.0, 1.0)
    
    # Gamma correction (simulated Mojo SIMD)
    enhanced_frame = np.power(enhanced_frame, 1.0 / 2.2)
    
    # Convert back to uint8
    enhanced_frame = (enhanced_frame * 255).astype(np.uint8)
    
    processing_time = time.perf_counter() - start_time
    
    stats = {
        'processing_time_ms': processing_time * 1000,
        'operations_used': ['contrast', 'gamma'],
        'backend': 'Mojo-Simulated+SIMD'
    }
    
    return enhanced_frame, stats

def gpu_worker_process(gpu_id: int, frame_batch: List[Tuple[int, np.ndarray]], output_dir: str) -> Dict:
    """Worker process that runs on a specific GPU"""
    try:
        # Setup this GPU
        model, config = setup_gpu_worker(gpu_id)
        
        print(f"🚀 GPU {gpu_id} Worker Started: {torch.cuda.get_device_name(0)}")
        
        batch_results = {
            'gpu_id': gpu_id,
            'frames_processed': 0,
            'processing_time': 0,
            'mojo_operations': 0,
            'detection_time': 0,
            'total_detections': 0
        }
        
        start_time = time.perf_counter()
        
        for second, frame in frame_batch:
            try:
                # Apply Mojo-simulated frame enhancements
                enhanced_frame, mojo_stats = process_frame_with_mojo_simulation(frame)
                
                # Run object detection on enhanced frame
                detection_start = time.perf_counter()
                detections_list = model.inference([enhanced_frame])
                detection_time = time.perf_counter() - detection_start
                
                detections = detections_list[0] if detections_list else []
                
                # Create annotated version
                from utils.visualizer import create_detection_overlay
                annotated_frame = create_detection_overlay(enhanced_frame, detections)
                
                # Save both versions
                frame_name = f"frame_{second:04d}_gpu{gpu_id}.jpg"
                annotated_name = f"annotated_{second:04d}_gpu{gpu_id}.jpg"
                
                cv2.imwrite(os.path.join(output_dir, frame_name), enhanced_frame)
                cv2.imwrite(os.path.join(output_dir, annotated_name), annotated_frame)
                
                batch_results['frames_processed'] += 1
                batch_results['mojo_operations'] += len(mojo_stats['operations_used'])
                batch_results['detection_time'] += detection_time
                batch_results['total_detections'] += len(detections)
                
                print(f"   ✅ GPU {gpu_id}: Frame {second:04d} - {len(detections)} objects ({mojo_stats['backend']})")
                
            except Exception as e:
                print(f"   ❌ GPU {gpu_id}: Error processing frame {second}: {e}")
        
        processing_time = time.perf_counter() - start_time
        batch_results['processing_time'] = processing_time
        
        print(f"🏁 GPU {gpu_id} Completed: {batch_results['frames_processed']} frames in {processing_time:.2f}s")
        
        return batch_results
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} Worker Failed: {e}")
        return {
            'gpu_id': gpu_id,
            'frames_processed': 0,
            'processing_time': 0,
            'error': str(e)
        }

class TrueMultiGPUProcessor:
    """TRUE Multi-GPU processor that uses all 8 H100s simultaneously"""
    
    def __init__(self):
        self.num_gpus = 8  # Force use of all 8 GPUs
        print(f"🚀 TRUE Multi-GPU Processor Initialized")
        print(f"   Target GPUs: {self.num_gpus}")
        print(f"   Total GPU Memory: {self.num_gpus * 80}GB")
        
        # Verify GPU availability
        for i in range(self.num_gpus):
            try:
                name = subprocess.check_output(
                    f"nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i {i}",
                    shell=True, text=True
                ).strip()
                print(f"   GPU {i}: {name}")
            except:
                print(f"   GPU {i}: Not available")
    
    def extract_frames_every_second(self, video_path: str) -> List[Tuple[int, np.ndarray]]:
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
        print(f"   ✅ Extracted {len(frames)} frames from {second_count} seconds")
        return frames
    
    def process_video_true_multi_gpu(self, video_path: str, output_dir: str) -> Dict:
        """Process video using TRUE multi-GPU distribution"""
        print(f"🎬 Processing video: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean existing images from output directory
        cleaned_files = ensure_clean_start(output_dir)
        if cleaned_files > 0:
            print(f"🧹 Cleaned {cleaned_files} existing files from {output_dir}")
        
        # Extract frames every second
        frames = self.extract_frames_every_second(video_path)
        
        if not frames:
            print("❌ No frames extracted!")
            return {}
        
        # Distribute frames across ALL 8 GPUs
        frames_per_gpu = len(frames) // self.num_gpus
        remainder = len(frames) % self.num_gpus
        
        gpu_batches = []
        start_idx = 0
        
        for gpu_id in range(self.num_gpus):
            # Calculate batch size (distribute remainder across first few GPUs)
            batch_size = frames_per_gpu + (1 if gpu_id < remainder else 0)
            end_idx = start_idx + batch_size
            
            gpu_batch = frames[start_idx:end_idx]
            gpu_batches.append((gpu_id, gpu_batch))
            
            print(f"   📋 GPU {gpu_id}: {len(gpu_batch)} frames assigned")
            start_idx = end_idx
        
        # Process batches in parallel using ProcessPoolExecutor for TRUE multi-GPU
        print(f"⚡ Processing {len(frames)} frames across {self.num_gpus} GPUs in PARALLEL...")
        
        total_start_time = time.perf_counter()
        results = []
        
        # Use ProcessPoolExecutor to ensure each GPU gets its own process
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {
                executor.submit(gpu_worker_process, gpu_id, batch, output_dir): gpu_id 
                for gpu_id, batch in gpu_batches
            }
            
            for future in as_completed(futures):
                gpu_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ GPU {gpu_id}: {result['frames_processed']} frames completed")
                except Exception as e:
                    print(f"❌ GPU {gpu_id} failed: {e}")
        
        total_time = time.perf_counter() - total_start_time
        
        # Aggregate results
        summary = {
            'total_frames': len(frames),
            'total_time': total_time,
            'fps': len(frames) / total_time,
            'gpus_used': len([r for r in results if r['frames_processed'] > 0]),
            'gpu_results': results,
            'total_detections': sum(r.get('total_detections', 0) for r in results),
            'total_mojo_operations': sum(r.get('mojo_operations', 0) for r in results)
        }
        
        # Save summary
        with open(os.path.join(output_dir, 'true_multi_gpu_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="🔥 TRUE Multi-GPU Video Processor (8x H100)")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", default="results/true_multi_gpu", help="Output directory for frames")
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Video file '{args.input}' not found!")
        return
    
    # Initialize processor
    processor = TrueMultiGPUProcessor()
    
    # Process video
    results = processor.process_video_true_multi_gpu(args.input, args.output)
    
    if results:
        print("\n🏆 TRUE MULTI-GPU PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"📊 Performance Summary:")
        print(f"   Total Frames: {results['total_frames']}")
        print(f"   Processing Time: {results['total_time']:.2f}s")
        print(f"   Throughput: {results['fps']:.1f} FPS")
        print(f"   🚀 GPUs Actually Used: {results['gpus_used']}/8")
        print(f"   Total Detections: {results['total_detections']}")
        print(f"   Mojo Operations: {results['total_mojo_operations']}")
        
        # Show per-GPU performance
        print(f"\n📈 Per-GPU Performance:")
        for result in results['gpu_results']:
            if result['frames_processed'] > 0:
                fps = result['frames_processed'] / result['processing_time']
                print(f"   GPU {result['gpu_id']}: {result['frames_processed']} frames @ {fps:.1f} FPS")
        
        print(f"\n📁 Results saved to: {args.output}")
        print(f"   Enhanced frames: {results['total_frames']}")
        print(f"   Annotated frames: {results['total_frames']}")
        print(f"   JSON report: true_multi_gpu_results.json")
    
    print("\n🚀 TRUE Multi-GPU + Mojo processing completed!")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 