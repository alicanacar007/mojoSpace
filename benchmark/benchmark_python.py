#!/usr/bin/env python3
"""
MojoX Benchmark - Python-only baseline performance testing
"""

import argparse
import time
import statistics
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from app import VideoObjectDetectionPipeline

class PythonBenchmark:
    """Benchmark suite for Python-only pipeline"""
    
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Force Python implementations
        self.config.frame_extraction.use_mojo_kernel = False
        self.config.nms.use_mojo_kernel = False
        self.config.model.use_max_graph = False
        
        self.results = {
            'system_info': self._get_system_info(),
            'config': self._get_config_summary(),
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import torch
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
        }
        
        # GPU info
        if torch.cuda.is_available():
            system_info.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1e9
            })
        else:
            system_info['gpu_available'] = False
        
        return system_info
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'frame_extraction': {
                'target_fps': self.config.frame_extraction.target_fps,
                'use_mojo_kernel': self.config.frame_extraction.use_mojo_kernel
            },
            'model': {
                'device': self.config.model.device,
                'input_size': self.config.model.input_size,
                'use_max_graph': self.config.model.use_max_graph
            },
            'nms': {
                'use_mojo_kernel': self.config.nms.use_mojo_kernel,
                'iou_threshold': self.config.nms.iou_threshold
            }
        }
    
    def benchmark_frame_extraction(self, num_frames: int = 100) -> Dict[str, Any]:
        """Benchmark frame extraction performance"""
        print(f"Benchmarking Python frame extraction with {num_frames} frames...")
        
        # Create synthetic video data
        frame_data = np.random.randint(0, 255, (num_frames, 480, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            _ = frame_data[0]
        
        # Benchmark
        times = []
        for i in range(num_frames):
            start = time.perf_counter()
            
            # Python frame extraction simulation
            frame = frame_data[i]
            
            # Resize using numpy (slower than OpenCV but pure Python)
            h, w = frame.shape[:2]
            target_h, target_w = self.config.model.input_size
            
            # Simple resize simulation
            resized = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_ratio = h / target_h
            x_ratio = w / target_w
            
            for y in range(target_h):
                for x in range(target_w):
                    orig_y = int(y * y_ratio)
                    orig_x = int(x * x_ratio)
                    if orig_y < h and orig_x < w:
                        resized[y, x] = frame[orig_y, orig_x]
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'total_time': sum(times),
            'avg_time_per_frame': statistics.mean(times),
            'std_time_per_frame': statistics.stdev(times),
            'min_time': min(times),
            'max_time': max(times),
            'fps': num_frames / sum(times)
        }
    
    def benchmark_inference(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark model inference performance"""
        print(f"Benchmarking Python inference with {num_iterations} iterations...")
        
        # Initialize pipeline
        pipeline = VideoObjectDetectionPipeline(self.config)
        
        # Create synthetic input
        batch_size = self.config.model.batch_size
        input_shape = (batch_size,) + self.config.model.input_size + (3,)
        input_data = [np.random.randint(0, 255, input_shape[1:], dtype=np.uint8) for _ in range(batch_size)]
        
        # Warmup
        for _ in range(10):
            input_tensor = pipeline.model.preprocess_batch(input_data)
            _ = pipeline.model.inference(input_tensor)
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            
            input_tensor = pipeline.model.preprocess_batch(input_data)
            detections = pipeline.model.inference(input_tensor)
            
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'total_time': sum(times),
            'avg_time_per_batch': statistics.mean(times),
            'std_time_per_batch': statistics.stdev(times),
            'min_time': min(times),
            'max_time': max(times),
            'fps': (num_iterations * batch_size) / sum(times)
        }
    
    def benchmark_nms_python(self, detections: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """Pure Python NMS implementation"""
        if len(detections) == 0:
            return detections
        
        # Extract coordinates and scores
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]
        
        return detections[keep]
    
    def benchmark_nms(self, num_detections_list: List[int] = [10, 50, 100, 500]) -> Dict[str, Any]:
        """Benchmark NMS performance"""
        print("Benchmarking Python NMS performance...")
        
        nms_results = {}
        
        for num_detections in num_detections_list:
            print(f"  Testing with {num_detections} detections...")
            
            # Create synthetic detections [x1, y1, x2, y2, conf, class_id]
            detections = np.random.rand(num_detections, 6)
            detections[:, :4] *= 640  # Scale bounding boxes
            detections[:, 4] = np.random.uniform(0.1, 1.0, num_detections)  # Confidence
            detections[:, 5] = np.random.randint(0, 80, num_detections)  # Class ID
            
            # Warmup
            for _ in range(5):
                _ = self.benchmark_nms_python(detections)
            
            # Benchmark
            times = []
            for _ in range(20):
                start = time.perf_counter()
                
                # Apply Python NMS
                filtered = self.benchmark_nms_python(detections, self.config.nms.iou_threshold)
                
                end = time.perf_counter()
                times.append(end - start)
            
            nms_results[f'{num_detections}_detections'] = {
                'avg_time': statistics.mean(times),
                'std_time': statistics.stdev(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return nms_results
    
    def benchmark_end_to_end(self, video_path: str = None) -> Dict[str, Any]:
        """Benchmark end-to-end pipeline performance"""
        print("Benchmarking end-to-end Python pipeline...")
        
        pipeline = VideoObjectDetectionPipeline(self.config)
        
        if video_path and os.path.exists(video_path):
            print(f"Processing video: {video_path}")
            stats = pipeline.process_video(video_path)
        else:
            print("Running demo mode...")
            stats = pipeline.process_demo_mode()
        
        return {
            'total_frames': stats.get('total_frames', 0),
            'total_time': stats.get('total_processing_time', 0),
            'avg_fps': stats.get('average_fps', 0),
            'avg_inference_time': stats.get('average_inference_time', 0),
            'avg_nms_time': stats.get('average_nms_time', 0)
        }
    
    def run_all_benchmarks(self, video_path: str = None) -> Dict[str, Any]:
        """Run all benchmark suites"""
        print("=" * 60)
        print("MOJOX PYTHON-ONLY BASELINE BENCHMARK SUITE")
        print("=" * 60)
        
        # Frame extraction benchmark
        self.results['benchmarks']['frame_extraction'] = self.benchmark_frame_extraction()
        
        # Inference benchmark
        self.results['benchmarks']['inference'] = self.benchmark_inference()
        
        # NMS benchmark
        self.results['benchmarks']['nms'] = self.benchmark_nms()
        
        # End-to-end benchmark
        self.results['benchmarks']['end_to_end'] = self.benchmark_end_to_end(video_path)
        
        return self.results
    
    def print_results(self):
        """Print benchmark results"""
        print("\n" + "=" * 60)
        print("PYTHON BASELINE RESULTS")
        print("=" * 60)
        
        print(f"\nSystem Info:")
        for key, value in self.results['system_info'].items():
            print(f"  {key}: {value}")
        
        print(f"\nFrame Extraction:")
        fe_results = self.results['benchmarks']['frame_extraction']
        print(f"  Average FPS: {fe_results['fps']:.1f}")
        print(f"  Avg time per frame: {fe_results['avg_time_per_frame']*1000:.2f}ms")
        
        print(f"\nInference:")
        inf_results = self.results['benchmarks']['inference']
        print(f"  Average FPS: {inf_results['fps']:.1f}")
        print(f"  Avg time per batch: {inf_results['avg_time_per_batch']*1000:.2f}ms")
        
        print(f"\nNMS:")
        nms_results = self.results['benchmarks']['nms']
        for key, value in nms_results.items():
            print(f"  {key}: {value['avg_time']*1000:.2f}ms")
        
        print(f"\nEnd-to-End:")
        e2e_results = self.results['benchmarks']['end_to_end']
        print(f"  Average FPS: {e2e_results['avg_fps']:.1f}")
        print(f"  Total frames: {e2e_results['total_frames']}")
        print(f"  Total time: {e2e_results['total_time']:.2f}s")
        
        print("=" * 60)
    
    def save_results(self, output_path: str):
        """Save benchmark results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="MojoX Python-only baseline benchmark")
    parser.add_argument("--video", type=str, help="Path to test video file")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="benchmark_results_python.json", 
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = PythonBenchmark(args.config)
        
        # Run benchmarks
        results = benchmark.run_all_benchmarks(args.video)
        
        # Print results
        benchmark.print_results()
        
        # Save results
        benchmark.save_results(args.output)
        
        return 0
    
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 