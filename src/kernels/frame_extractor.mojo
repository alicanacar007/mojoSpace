"""
Frame Extractor Kernel - High-performance video frame extraction using Mojo
Extracts frames from video streams at configurable FPS and converts to tensors
"""

from tensor import Tensor, TensorSpec
from algorithm import vectorize, parallelize
from memory import memset_zero, memcpy
from sys import simdwidthof
from runtime.llcl import Runtime

struct FrameExtractorConfig:
    var target_fps: Float32
    var max_width: Int
    var max_height: Int
    var channels: Int
    var batch_size: Int
    
    fn __init__(inout self, target_fps: Float32 = 30.0, max_width: Int = 1920, max_height: Int = 1080, channels: Int = 3, batch_size: Int = 8):
        self.target_fps = target_fps
        self.max_width = max_width
        self.max_height = max_height
        self.channels = channels
        self.batch_size = batch_size

struct FrameExtractor:
    var config: FrameExtractorConfig
    var runtime: Runtime
    
    fn __init__(inout self, config: FrameExtractorConfig):
        self.config = config
        self.runtime = Runtime()
    
    fn extract_frame_batch(self, video_buffer: Tensor[DType.uint8], frame_indices: Tensor[DType.int32]) -> Tensor[DType.float32]:
        """
        Extract a batch of frames from video buffer and convert to normalized tensors
        Args:
            video_buffer: Raw video data buffer
            frame_indices: Indices of frames to extract
        Returns:
            Normalized tensor batch [N, C, H, W] in range [0, 1]
        """
        let batch_size = frame_indices.shape()[0]
        let output_shape = TensorSpec(
            DType.float32,
            batch_size, self.config.channels, self.config.max_height, self.config.max_width
        )
        var output_tensor = Tensor[DType.float32](output_shape)
        
        # Parallelize frame extraction across batch
        @parameter
        fn extract_single_frame(batch_idx: Int):
            let frame_idx = int(frame_indices[batch_idx])
            self._extract_and_normalize_frame(video_buffer, frame_idx, output_tensor, batch_idx)
        
        parallelize[extract_single_frame](batch_size)
        return output_tensor
    
    fn _extract_and_normalize_frame(self, video_buffer: Tensor[DType.uint8], frame_idx: Int, inout output: Tensor[DType.float32], batch_idx: Int):
        """Extract single frame and normalize pixel values"""
        let frame_size = self.config.max_width * self.config.max_height * self.config.channels
        let frame_offset = frame_idx * frame_size
        
        # Vectorized normalization: convert uint8 [0,255] to float32 [0,1]
        @parameter
        fn normalize_pixels(i: Int):
            let simd_width = simdwidthof[DType.uint8]()
            for j in range(0, simd_width):
                if i + j < frame_size:
                    let pixel_val = video_buffer[frame_offset + i + j]
                    let normalized = Float32(pixel_val) / 255.0
                    output[batch_idx * frame_size + i + j] = normalized
        
        vectorize[normalize_pixels, simdwidthof[DType.uint8]()](frame_size)
    
    fn calculate_frame_timestamps(self, total_frames: Int, video_fps: Float32) -> Tensor[DType.float32]:
        """Calculate frame timestamps based on target FPS"""
        let frame_interval = 1.0 / self.config.target_fps
        let source_interval = 1.0 / video_fps
        let frame_ratio = video_fps / self.config.target_fps
        
        let num_output_frames = int(total_frames / frame_ratio)
        let timestamps_spec = TensorSpec(DType.float32, num_output_frames)
        var timestamps = Tensor[DType.float32](timestamps_spec)
        
        for i in range(num_output_frames):
            timestamps[i] = Float32(i) * frame_interval
        
        return timestamps
    
    fn get_frame_indices_from_timestamps(self, timestamps: Tensor[DType.float32], video_fps: Float32) -> Tensor[DType.int32]:
        """Convert timestamps to frame indices"""
        let num_frames = timestamps.shape()[0]
        let indices_spec = TensorSpec(DType.int32, num_frames)
        var indices = Tensor[DType.int32](indices_spec)
        
        for i in range(num_frames):
            let frame_idx = int(timestamps[i] * video_fps)
            indices[i] = frame_idx
        
        return indices

# Export the kernel for Python interop
alias frame_extractor_kernel = FrameExtractor 