"""
Video Loader - Handles video loading and frame extraction with FFmpeg fallback
Provides both Mojo kernel integration and Python-based video processing
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator, Dict, Any
import ffmpeg
from pathlib import Path
import tempfile
import subprocess
from dataclasses import dataclass

@dataclass
class VideoInfo:
    """Video metadata information"""
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    format: str
    bitrate: Optional[int] = None

class VideoLoader:
    """High-performance video loader with multiple backends"""
    
    def __init__(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        resize_dims: Optional[Tuple[int, int]] = None,
        use_mojo_kernel: bool = True,
        buffer_size: int = 32
    ):
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.resize_dims = resize_dims
        self.use_mojo_kernel = use_mojo_kernel
        self.buffer_size = buffer_size
        
        self.video_info: Optional[VideoInfo] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_buffer: List[np.ndarray] = []
        self._current_frame_idx = 0
        
        self._initialize_video()
    
    def _initialize_video(self):
        """Initialize video capture and get metadata"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Get video information
        self.video_info = self._get_video_info()
        
        # Initialize OpenCV capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        print(f"Video loaded: {self.video_info.width}x{self.video_info.height} @ {self.video_info.fps:.2f}fps")
        print(f"Total frames: {self.video_info.total_frames}, Duration: {self.video_info.duration:.2f}s")
    
    def _get_video_info(self) -> VideoInfo:
        """Extract video metadata using FFmpeg probe"""
        try:
            probe = ffmpeg.probe(str(self.video_path))
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise RuntimeError("No video stream found in file")
            
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Parse FPS
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)
            
            # Get total frames and duration
            total_frames = int(video_stream.get('nb_frames', 0))
            duration = float(video_stream.get('duration', 0))
            
            # If total_frames is not available, estimate from duration and fps
            if total_frames == 0 and duration > 0:
                total_frames = int(duration * fps)
            
            codec = video_stream.get('codec_name', 'unknown')
            format_info = probe.get('format', {})
            format_name = format_info.get('format_name', 'unknown')
            bitrate = format_info.get('bit_rate')
            bitrate = int(bitrate) if bitrate else None
            
            return VideoInfo(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec,
                format=format_name,
                bitrate=bitrate
            )
        
        except Exception as e:
            print(f"Failed to get video info with FFmpeg: {e}")
            return self._get_video_info_opencv()
    
    def _get_video_info_opencv(self) -> VideoInfo:
        """Fallback video info extraction using OpenCV"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise RuntimeError("Cannot open video with OpenCV")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec='unknown',
            format='unknown'
        )
    
    def extract_frames_batch(self, frame_indices: List[int]) -> List[np.ndarray]:
        """Extract specific frames by index"""
        frames = []
        
        for idx in frame_indices:
            frame = self._get_frame_at_index(idx)
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def extract_frames_at_fps(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        target_fps: Optional[float] = None
    ) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Extract frames at specified FPS within time range"""
        target_fps = target_fps or self.target_fps or self.video_info.fps
        end_time = end_time or self.video_info.duration
        
        # Calculate frame indices based on target FPS
        source_fps = self.video_info.fps
        frame_interval = source_fps / target_fps
        
        current_time = start_time
        frame_idx = int(start_time * source_fps)
        
        while current_time <= end_time and frame_idx < self.video_info.total_frames:
            frame = self._get_frame_at_index(frame_idx)
            if frame is not None:
                yield current_time, frame
            
            # Move to next frame based on target FPS
            current_time += 1.0 / target_fps
            frame_idx = int(current_time * source_fps)
    
    def extract_all_frames(self) -> Generator[np.ndarray, None, None]:
        """Extract all frames from video"""
        if not self.cap or not self.cap.isOpened():
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.resize_dims:
                frame = cv2.resize(frame, self.resize_dims)
            
            yield frame
    
    def _get_frame_at_index(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get specific frame by index"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        if frame_idx >= self.video_info.total_frames:
            return None
        
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        if self.resize_dims:
            frame = cv2.resize(frame, self.resize_dims)
        
        return frame
    
    def extract_frames_ffmpeg(
        self,
        output_pattern: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
        fps: Optional[float] = None
    ) -> List[str]:
        """Extract frames using FFmpeg (high performance option)"""
        fps = fps or self.target_fps or self.video_info.fps
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, output_pattern)
        
        try:
            input_args = {'ss': start_time}
            if duration:
                input_args['t'] = duration
            
            output_args = {
                'vf': f'fps={fps}',
                'q:v': 2,  # High quality
                'format': 'image2'
            }
            
            if self.resize_dims:
                scale_filter = f'scale={self.resize_dims[0]}:{self.resize_dims[1]}'
                output_args['vf'] = f'{output_args["vf"]},{scale_filter}'
            
            (
                ffmpeg
                .input(str(self.video_path), **input_args)
                .output(output_path, **output_args)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Get list of generated frame files
            frame_files = sorted([
                os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            return frame_files
        
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return []
    
    def create_video_summary(self, num_samples: int = 10) -> List[Tuple[float, np.ndarray]]:
        """Create a video summary with evenly spaced sample frames"""
        if num_samples <= 0:
            return []
        
        duration = self.video_info.duration
        interval = duration / num_samples
        
        summary_frames = []
        for i in range(num_samples):
            timestamp = i * interval
            frame_idx = int(timestamp * self.video_info.fps)
            frame = self._get_frame_at_index(frame_idx)
            
            if frame is not None:
                summary_frames.append((timestamp, frame))
        
        return summary_frames
    
    def get_video_info(self) -> VideoInfo:
        """Get video metadata"""
        return self.video_info
    
    def get_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp"""
        frame_idx = int(timestamp * self.video_info.fps)
        return self._get_frame_at_index(frame_idx)
    
    def close(self):
        """Release video resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class VideoProcessor:
    """High-level video processing utilities"""
    
    @staticmethod
    def convert_video_format(
        input_path: str,
        output_path: str,
        target_fps: Optional[float] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        quality: str = 'high'
    ):
        """Convert video to different format/settings"""
        input_stream = ffmpeg.input(input_path)
        
        # Build filter chain
        filters = []
        if target_fps:
            filters.append(f'fps={target_fps}')
        if target_resolution:
            filters.append(f'scale={target_resolution[0]}:{target_resolution[1]}')
        
        # Quality settings
        codec_args = {}
        if quality == 'high':
            codec_args.update({'crf': 18, 'preset': 'slow'})
        elif quality == 'medium':
            codec_args.update({'crf': 23, 'preset': 'medium'})
        else:  # low
            codec_args.update({'crf': 28, 'preset': 'fast'})
        
        # Apply filters if any
        if filters:
            input_stream = input_stream.video.filter('scale', *filters)
        
        # Output with codec settings
        output = ffmpeg.output(
            input_stream,
            output_path,
            vcodec='libx264',
            **codec_args
        )
        
        ffmpeg.run(output, overwrite_output=True)
    
    @staticmethod
    def extract_audio(input_path: str, output_path: str):
        """Extract audio from video"""
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='copy')
            .overwrite_output()
            .run()
        )
    
    @staticmethod
    def create_video_from_frames(
        frame_paths: List[str],
        output_path: str,
        fps: float = 30.0,
        quality: str = 'high'
    ):
        """Create video from sequence of frames"""
        if not frame_paths:
            raise ValueError("No frame paths provided")
        
        # Determine input pattern
        input_pattern = frame_paths[0].replace('000', '%03d')
        
        codec_args = {}
        if quality == 'high':
            codec_args.update({'crf': 18, 'preset': 'slow'})
        elif quality == 'medium':
            codec_args.update({'crf': 23, 'preset': 'medium'})
        else:  # low
            codec_args.update({'crf': 28, 'preset': 'fast'})
        
        (
            ffmpeg
            .input(input_pattern, framerate=fps)
            .output(
                output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',
                **codec_args
            )
            .overwrite_output()
            .run()
        )

# Utility functions
def load_video(video_path: str, **kwargs) -> VideoLoader:
    """Convenience function to load video"""
    return VideoLoader(video_path, **kwargs)

def get_video_info(video_path: str) -> VideoInfo:
    """Get video information without loading the full video"""
    with VideoLoader(video_path) as loader:
        return loader.get_video_info()

def extract_frames_simple(
    video_path: str,
    num_frames: int = 10,
    resize_dims: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """Simple frame extraction for quick testing"""
    with VideoLoader(video_path, resize_dims=resize_dims) as loader:
        video_info = loader.get_video_info()
        duration = video_info.duration
        
        frames = []
        for i in range(num_frames):
            timestamp = (i / (num_frames - 1)) * duration if num_frames > 1 else 0
            frame = loader.get_frame_at_time(timestamp)
            if frame is not None:
                frames.append(frame)
        
        return frames

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        with VideoLoader(video_path) as loader:
            info = loader.get_video_info()
            print(f"Video Info: {info}")
            
            # Extract sample frames
            frames = extract_frames_simple(video_path, num_frames=5)
            print(f"Extracted {len(frames)} frames")
    else:
        print("Usage: python video_loader.py <video_path>") 