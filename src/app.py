#!/usr/bin/env python3
"""
Real-Time Video Object Detection with Mojo Kernels & MAX Graph
Main application entry point and pipeline orchestration
"""

import argparse
import time
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import ConfigManager, PipelineConfig
from utils.video_loader import VideoLoader
from utils.visualizer import (
    VideoVisualizer, FrameAnnotator, Detection, 
    convert_detections_format, VisualizationStyle
)
from models.yolo_graph import YOLOGraphModel

class VideoObjectDetectionPipeline:
    """Main pipeline for real-time video object detection"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.video_loader: Optional[VideoLoader] = None
        self.model: Optional[YOLOGraphModel] = None
        self.frame_annotator: Optional[FrameAnnotator] = None
        
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        self.total_detections = 0
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        print("Initializing Video Object Detection Pipeline...")
        
        # Initialize YOLO model
        print("Loading YOLO model...")
        model_config = self.config.model
        self.model = YOLOGraphModel(
            model_path=model_config.model_path,
            device=model_config.device,
            input_size=model_config.input_size,
            num_classes=model_config.num_classes,
            conf_threshold=model_config.conf_threshold,
            batch_size=model_config.batch_size
        )
        
        # Initialize frame annotator
        style = VisualizationStyle(
            box_thickness=self.config.visualization.box_thickness,
            font_scale=self.config.visualization.font_scale,
            font_thickness=self.config.visualization.font_thickness,
            box_alpha=self.config.visualization.box_alpha,
            text_alpha=self.config.visualization.text_alpha,
            show_confidence=True,
            confidence_threshold=model_config.conf_threshold
        )
        
        self.frame_annotator = FrameAnnotator(
            style=style,
            class_names={i: self.model.get_class_name(i) for i in range(model_config.num_classes)}
        )
        
        print("Pipeline initialization complete!")
        print(f"Model info: {self.model.get_model_info()}")
    
    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> dict:
        """Process video file and return statistics"""
        print(f"Processing video: {input_path}")
        
        # Initialize video loader
        self.video_loader = VideoLoader(
            input_path,
            target_fps=self.config.frame_extraction.target_fps,
            resize_dims=self.config.model.input_size,
            use_mojo_kernel=self.config.frame_extraction.use_mojo_kernel
        )
        
        video_info = self.video_loader.get_video_info()
        print(f"Video Info: {video_info.width}x{video_info.height} @ {video_info.fps:.2f}fps")
        
        # Initialize video writer if output path provided
        video_writer = None
        if output_path and self.config.enable_visualization:
            video_writer = VideoVisualizer(
                output_path,
                fps=self.config.visualization.fps,
                quality=self.config.visualization.quality,
                frame_annotator=self.frame_annotator
            )
        
        # Process frames
        processing_stats = self._process_frames(
            video_writer, start_time, duration
        )
        
        # Finalize output
        if video_writer:
            video_writer.finalize()
        
        # Clean up
        self.video_loader.close()
        
        return processing_stats
    
    def _process_frames(
        self,
        video_writer: Optional[VideoVisualizer],
        start_time: float,
        duration: Optional[float]
    ) -> dict:
        """Process video frames through the detection pipeline"""
        end_time = duration + start_time if duration else None
        frame_count = 0
        total_inference_time = 0.0
        total_nms_time = 0.0
        
        print("Starting frame processing...")
        
        # Extract frames at target FPS
        for timestamp, frame in self.video_loader.extract_frames_at_fps(
            start_time=start_time,
            end_time=end_time
        ):
            frame_start = time.time()
            
            # Preprocess frame
            frames_batch = [frame]
            input_tensor = self.model.preprocess_batch(frames_batch)
            
            # Run inference
            inference_start = time.time()
            detections_batch = self.model.inference(input_tensor)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Process detections for current frame
            frame_detections = detections_batch[0] if detections_batch else np.empty((0, 6))
            
            # Apply NMS if enabled
            nms_start = time.time()
            if self.config.enable_nms and len(frame_detections) > 0:
                # TODO: Integrate Mojo NMS kernel here
                frame_detections = self._apply_nms_fallback(frame_detections)
            nms_time = time.time() - nms_start
            total_nms_time += nms_time
            
            # Convert to Detection objects
            detection_objects = convert_detections_format(
                frame_detections,
                class_names={i: self.model.get_class_name(i) for i in range(self.model.num_classes)}
            )
            
            # Add to video output
            if video_writer:
                video_writer.add_frame(frame, detection_objects)
            
            # Update statistics
            frame_count += 1
            self.total_detections += len(detection_objects)
            frame_processing_time = time.time() - frame_start
            
            # Print progress
            if frame_count % 30 == 0:  # Every second at 30 FPS
                fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
                print(f"Frame {frame_count}: {len(detection_objects)} detections, "
                      f"{fps:.1f} FPS, {inference_time*1000:.1f}ms inference")
        
        total_time = total_inference_time + total_nms_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        stats = {
            'total_frames': frame_count,
            'total_processing_time': total_time,
            'total_inference_time': total_inference_time,
            'total_nms_time': total_nms_time,
            'total_detections': self.total_detections,
            'average_fps': avg_fps,
            'average_inference_time': total_inference_time / frame_count if frame_count > 0 else 0,
            'average_nms_time': total_nms_time / frame_count if frame_count > 0 else 0
        }
        
        return stats
    
    def _apply_nms_fallback(self, detections: np.ndarray) -> np.ndarray:
        """Fallback NMS implementation (to be replaced with Mojo kernel)"""
        if len(detections) == 0:
            return detections
        
        # Simple confidence-based filtering for now
        conf_threshold = self.config.nms.score_threshold
        filtered = detections[detections[:, 4] >= conf_threshold]
        
        # Sort by confidence and take top N
        if len(filtered) > self.config.nms.max_detections:
            sorted_indices = np.argsort(filtered[:, 4])[::-1]
            filtered = filtered[sorted_indices[:self.config.nms.max_detections]]
        
        return filtered
    
    def process_demo_mode(self) -> dict:
        """Run in demo mode with sample data"""
        print("Running in demo mode...")
        
        # Create synthetic frames
        demo_frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            demo_frames.append(frame)
        
        # Process demo frames
        start_time = time.time()
        
        for i, frame in enumerate(demo_frames):
            # Preprocess
            input_tensor = self.model.preprocess_batch([frame])
            
            # Inference
            detections_batch = self.model.inference(input_tensor)
            frame_detections = detections_batch[0] if detections_batch else np.empty((0, 6))
            
            print(f"Demo frame {i+1}: {len(frame_detections)} detections")
        
        processing_time = time.time() - start_time
        
        return {
            'total_frames': len(demo_frames),
            'total_processing_time': processing_time,
            'average_fps': len(demo_frames) / processing_time,
            'mode': 'demo'
        }
    
    def print_statistics(self, stats: dict):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Total processing time: {stats['total_processing_time']:.2f}s")
        
        if 'total_inference_time' in stats:
            print(f"Total inference time: {stats['total_inference_time']:.2f}s")
            print(f"Total NMS time: {stats['total_nms_time']:.2f}s")
            print(f"Average inference time: {stats['average_inference_time']*1000:.1f}ms")
            print(f"Average NMS time: {stats['average_nms_time']*1000:.1f}ms")
        
        print(f"Average FPS: {stats['average_fps']:.1f}")
        
        if 'total_detections' in stats:
            print(f"Total detections: {stats.get('total_detections', 0)}")
        
        print("="*60)

def create_arg_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Real-Time Video Object Detection with Mojo Kernels & MAX Graph"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input video file path"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output video file path"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic data"
    )
    
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration to process in seconds"
    )
    
    parser.add_argument(
        "--fps",
        type=float,
        help="Target output FPS"
    )
    
    parser.add_argument(
        "--conf-threshold",
        type=float,
        help="Confidence threshold for detections"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser

def main():
    """Main application entry point"""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager(args.config) if args.config else ConfigManager()
        
        # Override config with command line arguments
        config_updates = {}
        if args.fps:
            config_updates['visualization.fps'] = args.fps
        if args.conf_threshold:
            config_updates['model.conf_threshold'] = args.conf_threshold
        if args.device:
            config_updates['model.device'] = args.device
        if args.verbose:
            config_updates['verbose'] = True
        if args.demo:
            config_updates['demo_mode'] = True
        
        if config_updates:
            config_manager.update_config(**config_updates)
        
        config = config_manager.get_config()
        
        if config.verbose:
            config_manager.print_config()
        
        # Initialize pipeline
        pipeline = VideoObjectDetectionPipeline(config)
        
        # Process video or run demo
        if args.demo or config.demo_mode:
            stats = pipeline.process_demo_mode()
        elif args.input:
            stats = pipeline.process_video(
                args.input,
                args.output,
                args.start_time,
                args.duration
            )
        else:
            print("Error: Please provide --input video file or use --demo mode")
            parser.print_help()
            return 1
        
        # Print results
        pipeline.print_statistics(stats)
        
        return 0
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 