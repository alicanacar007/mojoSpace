#!/usr/bin/env python3
"""
Multi-GPU Video Processor for MojoX
Distributes video processing across all available GPUs
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional, Any
import time
from PIL import Image
import json
import queue
import threading
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from models.yolo_graph import YOLOGraphModel
from utils.config import ConfigManager, PipelineConfig
from utils.visualizer import Visualizer
from utils.cleanup import ensure_clean_start

class MultiGPUVideoProcessor:
    """Multi-GPU video processing pipeline utilizing all 8 H100 GPUs"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.gpu_count = torch.cuda.device_count()
        self.batch_size_per_gpu = max(1, config.model.batch_size // self.gpu_count)
        self.total_batch_size = self.batch_size_per_gpu * self.gpu_count
        
        print(f"🔥 Initializing Multi-GPU Pipeline with {self.gpu_count} H100 GPUs")
        print(f"📊 Batch size per GPU: {self.batch_size_per_gpu}")
        print(f"📊 Total batch size: {self.total_batch_size}")
        
        # Initialize models on each GPU
        self.models = {}
        self.frame_queues = {}
        self.result_queues = {}
        
        self._initialize_multi_gpu_models()
        
    def _initialize_multi_gpu_models(self):
        """Initialize YOLO models on each GPU"""
        print("🚀 Loading models on all GPUs...")
        
        for gpu_id in range(self.gpu_count):
            print(f"  📌 Loading model on GPU {gpu_id}...")
            
            # Set device for this GPU
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            
            # Create model config for this GPU
            model_config = self.config.model
            model_config.device = device
            model_config.batch_size = self.batch_size_per_gpu
            
            # Initialize model
            model = YOLOGraphModel(
                model_path=model_config.model_path,
                device=device,
                input_size=model_config.input_size,
                num_classes=model_config.num_classes,
                conf_threshold=model_config.conf_threshold,
                batch_size=self.batch_size_per_gpu
            )
            
            self.models[gpu_id] = model
            self.frame_queues[gpu_id] = queue.Queue(maxsize=100)
            self.result_queues[gpu_id] = queue.Queue(maxsize=100)
            
        print("✅ All GPU models loaded successfully!")
    
    def process_video_parallel(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Process video using all GPUs in parallel"""
        
        print(f"🎬 Processing video: {video_path}")
        print(f"🔥 Using all {self.gpu_count} H100 GPUs in parallel")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video info: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
        
        # Initialize video writer if needed
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Start GPU processing threads
        gpu_threads = []
        for gpu_id in range(self.gpu_count):
            thread = threading.Thread(
                target=self._gpu_worker,
                args=(gpu_id, progress_callback),
                daemon=True
            )
            thread.start()
            gpu_threads.append(thread)
        
        # Process frames in batches
        start_time = time.time()
        processed_frames = 0
        total_detections = 0
        frame_buffer = []
        
        print("🚀 Starting parallel frame processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_buffer.append((processed_frames, frame))
            
            # Process batch when buffer is full
            if len(frame_buffer) >= self.total_batch_size:
                batch_results = self._process_frame_batch(frame_buffer, video_writer)
                total_detections += sum(len(dets) for dets in batch_results)
                processed_frames += len(frame_buffer)
                
                # Update progress
                if progress_callback:
                    progress = min(100, (processed_frames / frame_count) * 100)
                    progress_callback(progress, processed_frames, total_detections)
                
                frame_buffer = []
                
                # Print progress
                if processed_frames % (30 * self.gpu_count) == 0:
                    elapsed = time.time() - start_time
                    fps_current = processed_frames / elapsed if elapsed > 0 else 0
                    print(f"📊 Processed {processed_frames}/{frame_count} frames "
                          f"({fps_current:.1f} FPS) - {total_detections} detections")
        
        # Process remaining frames
        if frame_buffer:
            batch_results = self._process_frame_batch(frame_buffer, video_writer)
            total_detections += sum(len(dets) for dets in batch_results)
            processed_frames += len(frame_buffer)
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        
        # Stop GPU threads
        for gpu_id in range(self.gpu_count):
            self.frame_queues[gpu_id].put(None)  # Signal to stop
        for thread in gpu_threads:
            thread.join(timeout=5.0)
        
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        
        stats = {
            'total_frames': processed_frames,
            'total_processing_time': total_time,
            'total_detections': total_detections,
            'average_fps': avg_fps,
            'gpu_count': self.gpu_count,
            'frames_per_gpu': processed_frames // self.gpu_count,
            'speedup_factor': avg_fps / (fps / self.gpu_count) if fps > 0 else 0
        }
        
        print(f"✅ Processing complete!")
        print(f"📊 Stats: {processed_frames} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")
        print(f"🎯 Total detections: {total_detections}")
        print(f"⚡ Speedup: {stats['speedup_factor']:.1f}x over single GPU")
        
        return stats
    
    def _process_frame_batch(self, frame_buffer: List[Tuple[int, np.ndarray]], video_writer=None) -> List[List]:
        """Process a batch of frames across multiple GPUs"""
        
        # Distribute frames across GPUs
        frames_per_gpu = len(frame_buffer) // self.gpu_count
        gpu_batches = []
        
        for gpu_id in range(self.gpu_count):
            start_idx = gpu_id * frames_per_gpu
            end_idx = start_idx + frames_per_gpu
            if gpu_id == self.gpu_count - 1:  # Last GPU gets remaining frames
                end_idx = len(frame_buffer)
            
            gpu_batch = frame_buffer[start_idx:end_idx]
            gpu_batches.append((gpu_id, gpu_batch))
        
        # Submit batches to GPU workers
        batch_results = []
        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            futures = []
            
            for gpu_id, gpu_batch in gpu_batches:
                if gpu_batch:  # Only submit if batch is not empty
                    future = executor.submit(self._process_gpu_batch, gpu_id, gpu_batch)
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    gpu_detections = future.result(timeout=30.0)
                    batch_results.extend(gpu_detections)
                except Exception as e:
                    print(f"⚠️ GPU batch processing error: {e}")
                    batch_results.extend([[] for _ in range(len(gpu_batch))])
        
        # Write to video if needed
        if video_writer:
            for (frame_idx, frame), detections in zip(frame_buffer, batch_results):
                # Draw detections on frame
                annotated_frame = self._draw_detections(frame, detections)
                video_writer.write(annotated_frame)
        
        return batch_results
    
    def _process_gpu_batch(self, gpu_id: int, gpu_batch: List[Tuple[int, np.ndarray]]) -> List[List]:
        """Process a batch of frames on a specific GPU"""
        
        if not gpu_batch:
            return []
        
        # Set GPU device
        torch.cuda.set_device(gpu_id)
        model = self.models[gpu_id]
        
        # Extract frames
        frames = [frame for _, frame in gpu_batch]
        
        # Preprocess batch
        try:
            input_tensor = model.preprocess_batch(frames)
            
            # Run inference
            with torch.cuda.device(gpu_id):
                detections_batch = model.inference(input_tensor)
            
            return detections_batch
        
        except Exception as e:
            print(f"⚠️ Error processing batch on GPU {gpu_id}: {e}")
            return [[] for _ in frames]
    
    def _gpu_worker(self, gpu_id: int, progress_callback=None):
        """GPU worker thread for continuous processing"""
        torch.cuda.set_device(gpu_id)
        model = self.models[gpu_id]
        
        print(f"🔥 GPU {gpu_id} worker started")
        
        while True:
            try:
                # Get batch from queue
                batch = self.frame_queues[gpu_id].get(timeout=1.0)
                if batch is None:  # Stop signal
                    break
                
                # Process batch
                detections = self._process_gpu_batch(gpu_id, batch)
                
                # Put results
                self.result_queues[gpu_id].put(detections)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ GPU {gpu_id} worker error: {e}")
                continue
        
        print(f"🛑 GPU {gpu_id} worker stopped")
    
    def _draw_detections(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """Draw detection boxes on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            if len(detection) >= 6:
                x1, y1, x2, y2, score, class_id = detection[:6]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                if hasattr(self.models[0], 'get_class_name'):
                    class_name = self.models[0].get_class_name(int(class_id))
                    label = f"{class_name}: {score:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame
    
    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get memory usage for all GPUs"""
        gpu_memory = {}
        
        for gpu_id in range(self.gpu_count):
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            
            gpu_memory[gpu_id] = {
                'allocated_gb': round(allocated, 2),
                'total_gb': round(total, 1),
                'usage_percent': round((allocated / total) * 100, 1)
            }
        
        return gpu_memory

def main():
    """Main function for multi-GPU video processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-GPU Video Processor")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="results/images", help="Output directory for images")
    parser.add_argument("--config", default="config/default.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Video file not found: {args.input}")
        return
    
    # Initialize multi-GPU processor
    processor = MultiGPUVideoProcessor(args.config)
    
    # Process video
    results = processor.process_video_parallel(args.input, args.output)
    
    print(f"\n🚀 Multi-GPU processing completed!")
    print(f"   Check {args.output}/ for extracted frames and results")

if __name__ == "__main__":
    main() 