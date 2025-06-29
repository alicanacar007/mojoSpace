"""
Visualizer - Frame annotation and video output generation
Handles bounding box drawing, class labels, and video creation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import colorsys
import random
from pathlib import Path
import tempfile
import os
from dataclasses import dataclass

@dataclass
class Detection:
    """Single object detection"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""

@dataclass
class VisualizationStyle:
    """Visualization styling configuration"""
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    box_alpha: float = 0.3
    text_alpha: float = 1.0
    text_background: bool = True
    show_confidence: bool = True
    confidence_threshold: float = 0.0

class ColorManager:
    """Manages colors for different classes and visualization modes"""
    
    def __init__(self, color_mode: str = "coco"):
        self.color_mode = color_mode
        self.class_colors: Dict[int, Tuple[int, int, int]] = {}
        self._initialize_colors()
    
    def _initialize_colors(self):
        """Initialize color schemes"""
        if self.color_mode == "coco":
            # Standard COCO colors (BGR format)
            self.base_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (128, 0, 128),  # Purple
                (255, 165, 0),  # Orange
                (255, 192, 203), # Pink
                (0, 128, 128),  # Teal
                (128, 128, 0),  # Olive
                (0, 0, 128),    # Navy
                (128, 0, 0),    # Maroon
                (0, 128, 0),    # Dark Green
                (128, 128, 128), # Gray
            ]
        elif self.color_mode == "random":
            # Generate random colors
            self.base_colors = []
            for _ in range(100):
                self.base_colors.append((
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ))
        else:  # hsv or custom
            # Generate HSV-based colors for better distribution
            self.base_colors = []
            num_colors = 80  # For COCO classes
            for i in range(num_colors):
                hue = i / num_colors
                saturation = 0.7 + (i % 3) * 0.1
                value = 0.8 + (i % 2) * 0.2
                
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                self.base_colors.append(bgr)
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for specific class ID"""
        if class_id not in self.class_colors:
            color_idx = class_id % len(self.base_colors)
            self.class_colors[class_id] = self.base_colors[color_idx]
        
        return self.class_colors[class_id]
    
    def get_contrast_color(self, background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get contrasting color for text"""
        # Calculate brightness
        brightness = (background_color[0] * 0.299 + 
                     background_color[1] * 0.587 + 
                     background_color[2] * 0.114)
        
        # Return black or white based on brightness
        return (0, 0, 0) if brightness > 127 else (255, 255, 255)

class FrameAnnotator:
    """Annotates frames with detection results"""
    
    def __init__(
        self,
        style: Optional[VisualizationStyle] = None,
        color_manager: Optional[ColorManager] = None,
        class_names: Optional[Dict[int, str]] = None
    ):
        self.style = style or VisualizationStyle()
        self.color_manager = color_manager or ColorManager()
        self.class_names = class_names or {}
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated_frame = frame.copy()
        
        # Filter detections by confidence threshold
        filtered_detections = [
            det for det in detections 
            if det.confidence >= self.style.confidence_threshold
        ]
        
        for detection in filtered_detections:
            self._draw_detection(annotated_frame, detection)
        
        return annotated_frame
    
    def _draw_detection(self, frame: np.ndarray, detection: Detection):
        """Draw single detection on frame"""
        # Get coordinates
        x1, y1, x2, y2 = int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2)
        
        # Get color for this class
        color = self.color_manager.get_color(detection.class_id)
        
        # Draw bounding box
        if self.style.box_alpha < 1.0:
            # Semi-transparent box
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.style.box_thickness)
            cv2.addWeighted(overlay, self.style.box_alpha, frame, 1 - self.style.box_alpha, 0, frame)
        else:
            # Solid box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.style.box_thickness)
        
        # Prepare label text
        class_name = detection.class_name or self.class_names.get(detection.class_id, f"Class_{detection.class_id}")
        
        if self.style.show_confidence:
            label = f"{class_name}: {detection.confidence:.2f}"
        else:
            label = class_name
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, self.style.font_scale, self.style.font_thickness
        )
        
        # Calculate text position
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
        # Draw text background if enabled
        if self.style.text_background:
            background_x1 = text_x - 2
            background_y1 = text_y - text_height - 2
            background_x2 = text_x + text_width + 2
            background_y2 = text_y + baseline + 2
            
            if self.style.text_alpha < 1.0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (background_x1, background_y1), (background_x2, background_y2), color, -1)
                cv2.addWeighted(overlay, self.style.text_alpha, frame, 1 - self.style.text_alpha, 0, frame)
            else:
                cv2.rectangle(frame, (background_x1, background_y1), (background_x2, background_y2), color, -1)
        
        # Draw text
        text_color = self.color_manager.get_contrast_color(color)
        cv2.putText(
            frame, label, (text_x, text_y),
            font, self.style.font_scale, text_color, self.style.font_thickness
        )
    
    def create_detection_summary(self, detections: List[Detection]) -> str:
        """Create text summary of detections"""
        if not detections:
            return "No detections"
        
        class_counts = {}
        for det in detections:
            class_name = det.class_name or self.class_names.get(det.class_id, f"Class_{det.class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary_parts = []
        for class_name, count in class_counts.items():
            summary_parts.append(f"{class_name}: {count}")
        
        return ", ".join(summary_parts)

class VideoVisualizer:
    """Creates annotated videos from detection results"""
    
    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        quality: str = "high",
        frame_annotator: Optional[FrameAnnotator] = None
    ):
        self.output_path = output_path
        self.fps = fps
        self.quality = quality
        self.frame_annotator = frame_annotator or FrameAnnotator()
        
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.temp_frame_dir: Optional[str] = None
        self.frame_count = 0
    
    def initialize_writer(self, frame_shape: Tuple[int, int]):
        """Initialize video writer with frame dimensions"""
        height, width = frame_shape[:2]
        
        # Choose codec based on output format
        if self.output_path.lower().endswith('.mp4'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif self.output_path.lower().endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (width, height)
        )
        
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {self.output_path}")
    
    def add_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None
    ):
        """Add annotated frame to video"""
        if detections:
            annotated_frame = self.frame_annotator.annotate_frame(frame, detections)
        else:
            annotated_frame = frame
        
        # Initialize writer if not done yet
        if self.video_writer is None:
            self.initialize_writer(annotated_frame.shape)
        
        # Write frame
        self.video_writer.write(annotated_frame)
        self.frame_count += 1
    
    def add_frame_batch(
        self,
        frames: List[np.ndarray],
        detections_batch: Optional[List[List[Detection]]] = None
    ):
        """Add batch of frames to video"""
        for i, frame in enumerate(frames):
            detections = detections_batch[i] if detections_batch and i < len(detections_batch) else None
            self.add_frame(frame, detections)
    
    def finalize(self):
        """Finalize video creation"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        print(f"Video created: {self.output_path} ({self.frame_count} frames)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

class ImageSequenceVisualizer:
    """Creates sequence of annotated images"""
    
    def __init__(
        self,
        output_dir: str,
        filename_pattern: str = "frame_{:06d}.jpg",
        frame_annotator: Optional[FrameAnnotator] = None
    ):
        self.output_dir = Path(output_dir)
        self.filename_pattern = filename_pattern
        self.frame_annotator = frame_annotator or FrameAnnotator()
        self.frame_count = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None
    ) -> str:
        """Add annotated frame as image file"""
        if detections:
            annotated_frame = self.frame_annotator.annotate_frame(frame, detections)
        else:
            annotated_frame = frame
        
        # Generate filename
        filename = self.filename_pattern.format(self.frame_count)
        filepath = self.output_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), annotated_frame)
        self.frame_count += 1
        
        return str(filepath)
    
    def get_saved_frames(self) -> List[str]:
        """Get list of saved frame files"""
        return sorted([
            str(f) for f in self.output_dir.glob("*.jpg")
        ] + [
            str(f) for f in self.output_dir.glob("*.png")
        ])

# Utility functions
def convert_detections_format(
    detections_array: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> List[Detection]:
    """Convert numpy detection array to Detection objects"""
    detections = []
    
    for det in detections_array:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, class_id = det[:6]
            class_name = class_names.get(int(class_id), "") if class_names else ""
            
            detections.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=float(conf), class_id=int(class_id), class_name=class_name
            ))
    
    return detections

def create_demo_video(
    input_frames: List[np.ndarray],
    output_path: str,
    detections_batch: Optional[List[List[Detection]]] = None,
    fps: float = 30.0,
    style: Optional[VisualizationStyle] = None
):
    """Create demo video from frames and detections"""
    frame_annotator = FrameAnnotator(style=style)
    
    with VideoVisualizer(output_path, fps=fps, frame_annotator=frame_annotator) as visualizer:
        for i, frame in enumerate(input_frames):
            detections = detections_batch[i] if detections_batch and i < len(detections_batch) else None
            visualizer.add_frame(frame, detections)

def create_detection_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    style: Optional[VisualizationStyle] = None
) -> np.ndarray:
    """Simple function to overlay detections on frame"""
    annotator = FrameAnnotator(style=style)
    return annotator.annotate_frame(frame, detections)

class Visualizer:
    """Main Visualizer class for multi-GPU processing compatibility"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize style from config
        if hasattr(self.config, 'get'):
            style_config = self.config.get('style', {})
        elif hasattr(self.config, '__dict__'):
            style_config = getattr(self.config, 'style', {})
        else:
            style_config = {}
        self.style = VisualizationStyle(
            box_thickness=style_config.get('box_thickness', 2) if hasattr(style_config, 'get') else getattr(style_config, 'box_thickness', 2),
            font_scale=style_config.get('font_scale', 0.6) if hasattr(style_config, 'get') else getattr(style_config, 'font_scale', 0.6),
            font_thickness=style_config.get('font_thickness', 2) if hasattr(style_config, 'get') else getattr(style_config, 'font_thickness', 2),
            box_alpha=style_config.get('box_alpha', 0.3) if hasattr(style_config, 'get') else getattr(style_config, 'box_alpha', 0.3),
            show_confidence=style_config.get('show_confidence', True) if hasattr(style_config, 'get') else getattr(style_config, 'show_confidence', True),
            confidence_threshold=style_config.get('confidence_threshold', 0.0) if hasattr(style_config, 'get') else getattr(style_config, 'confidence_threshold', 0.0)
        )
        
        # Initialize color manager
        if hasattr(self.config, 'get'):
            color_mode = self.config.get('color_mode', 'coco')
        else:
            color_mode = getattr(self.config, 'color_mode', 'coco')
        self.color_manager = ColorManager(color_mode=color_mode)
        
        # Initialize annotator
        self.annotator = FrameAnnotator(
            style=self.style,
            color_manager=self.color_manager
        )
    
    def draw_detections(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Draw detections on frame (compatible with multi-GPU processor)"""
        if len(detections) == 0:
            return frame
        
        # Convert numpy array to Detection objects
        detection_objects = convert_detections_format(detections)
        
        # Annotate frame
        return self.annotator.annotate_frame(frame, detection_objects)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Demo with sample detections
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create sample detections
        detections = [
            Detection(50, 50, 150, 150, 0.9, 0, "person"),
            Detection(200, 100, 300, 200, 0.8, 2, "car"),
            Detection(400, 150, 500, 250, 0.7, 1, "bicycle")
        ]
        
        # Annotate frame
        annotated = create_detection_overlay(frame, detections)
        
        # Save result
        cv2.imwrite("demo_annotated.jpg", annotated)
        print("Demo annotation saved as demo_annotated.jpg")
    else:
        print("Usage: python visualizer.py demo") 