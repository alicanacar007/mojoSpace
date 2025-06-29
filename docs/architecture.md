# MojoX Architecture Documentation

## Overview

MojoX is a high-performance real-time video object detection system that combines the power of Mojo kernels for computational acceleration with MAX Graph for GPU-optimized neural network inference. The system is designed for maximum throughput and minimum latency in video processing workloads.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MojoX Pipeline Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐     │
│  │   Video     │    │   Frame      │    │   YOLO Model   │     │
│  │   Input     │───▶│  Extraction  │───▶│   (MAX Graph)  │     │
│  │             │    │ (Mojo Kernel)│    │                │     │
│  └─────────────┘    └──────────────┘    └────────────────┘     │
│                                                   │             │
│                                                   ▼             │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐     │
│  │  Annotated  │◀───│ Visualization│◀───│      NMS       │     │
│  │   Output    │    │   Pipeline   │    │ (Mojo Kernel)  │     │
│  │             │    │              │    │                │     │
│  └─────────────┘    └──────────────┘    └────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Frame Extraction Kernel (Mojo)

**Location**: `src/kernels/frame_extractor.mojo`

**Purpose**: High-performance video frame extraction and preprocessing

**Key Features**:
- Parallel frame extraction using Mojo's SIMD capabilities
- Configurable FPS sampling
- Memory-efficient tensor conversion
- Vectorized pixel normalization

**Performance Optimizations**:
- SIMD width optimization for pixel operations
- Parallel batch processing
- Zero-copy memory operations where possible

**Workflow**:
```mojo
struct FrameExtractor:
    fn extract_frame_batch(self, video_buffer: Tensor[DType.uint8], 
                          frame_indices: Tensor[DType.int32]) -> Tensor[DType.float32]:
        # Parallelize across frames
        parallelize[extract_single_frame](batch_size)
        return normalized_tensor
```

### 2. MAX Graph Model Integration

**Location**: `src/models/yolo_graph.py`

**Purpose**: GPU-accelerated YOLO inference using MAX Graph API

**Features**:
- Automatic fallback to PyTorch if MAX Graph unavailable
- Batch processing support
- Model optimization for target hardware
- Dynamic input sizing

**Workflow**:
1. Load pretrained YOLOv10 model
2. Convert to MAX Graph format (ONNX → MAX Graph)
3. Apply hardware-specific optimizations
4. Compile for target device (CUDA/CPU)

### 3. NMS Kernel (Mojo)

**Location**: `src/kernels/nms.mojo`

**Purpose**: Efficient Non-Maximum Suppression for detection post-processing

**Algorithm**:
1. Filter detections by confidence threshold
2. Sort by confidence scores (descending)
3. Apply IoU-based suppression
4. Return top-K detections

**Optimizations**:
- Vectorized IoU calculations
- Parallel processing for large detection sets
- Memory-efficient sorting algorithms

### 4. Configuration Management

**Location**: `src/utils/config.py`

**Purpose**: Centralized configuration with validation and environment setup

**Configuration Sections**:
- Frame Extraction Settings
- Model Parameters
- NMS Configuration
- Visualization Options
- System Resources
- Benchmarking Parameters

## Data Flow

### 1. Input Processing

```
Video File → VideoLoader → Frame Extraction
     ↓
Frame Buffer (Raw RGB) → Mojo Kernel → Normalized Tensors
     ↓
Batch Formation → Preprocessing → Model Input
```

### 2. Inference Pipeline

```
Input Tensors → MAX Graph Session → GPU Inference
     ↓
Raw Detections → Post-processing → Confidence Filtering
     ↓
Detection Array [N, 6] → NMS Kernel → Final Detections
```

### 3. Output Generation

```
Detections + Original Frame → Visualization Pipeline
     ↓
Bounding Box Rendering → Annotation → Output Frame
     ↓
Frame Sequence → Video Encoder → Final Video
```

## Performance Characteristics

### Computational Complexity

- **Frame Extraction**: O(W×H×C) per frame, parallelized
- **YOLO Inference**: O(model_params) - GPU accelerated
- **NMS**: O(N²) worst case, O(N log N) average with optimizations
- **Visualization**: O(D×frame_size) where D = number of detections

### Memory Usage

- **Video Buffer**: Configurable circular buffer
- **Model Memory**: ~100-500MB depending on YOLO variant
- **Frame Processing**: 2-4x frame size for processing pipeline
- **Detection Storage**: Minimal overhead (6 floats per detection)

### Throughput Targets

- **1080p @ 30 FPS**: Real-time processing target
- **4K @ 24 FPS**: High-resolution target
- **Batch Processing**: 2-5x real-time for offline processing

## Scalability Design

### Horizontal Scaling

- **Multi-GPU Support**: Batch distribution across devices
- **Multi-Process**: Parallel video stream processing
- **Cloud Deployment**: Container-based scaling

### Vertical Scaling

- **Memory Optimization**: Efficient tensor management
- **Compute Optimization**: Mojo kernel acceleration
- **I/O Optimization**: Asynchronous video loading

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4-core Intel/AMD processor
- **RAM**: 8GB system memory
- **GPU**: NVIDIA GTX 1060 or equivalent (optional)
- **Storage**: 10GB free space

### Recommended Configuration
- **CPU**: 8-core Intel/AMD processor with AVX2
- **RAM**: 16GB+ system memory
- **GPU**: NVIDIA RTX 3070 or better with 8GB+ VRAM
- **Storage**: NVMe SSD for video I/O

### Optimal Performance
- **CPU**: 16-core high-frequency processor
- **RAM**: 32GB+ DDR4/DDR5
- **GPU**: NVIDIA RTX 4080/4090 or H100
- **Storage**: High-speed NVMe RAID

## Integration Points

### Mojo Kernel Integration

The system integrates Mojo kernels through Python bindings:

```python
# Frame extraction integration
from kernels.frame_extractor import FrameExtractor
extractor = FrameExtractor(config)
frames = extractor.extract_frame_batch(video_buffer, indices)

# NMS integration  
from kernels.nms import NMSKernel
nms = NMSKernel(nms_config)
filtered_detections = nms.apply_nms(raw_detections)
```

### MAX Graph Integration

Model loading and optimization:

```python
# Load and optimize model
graph = max_graph.load_onnx(model_path)
optimized_graph = max_graph.optimize_graph(
    graph, target_device="cuda", optimization_level=3
)
compiled_model = session.compile(optimized_graph)
```

## Error Handling and Fallbacks

### Graceful Degradation

1. **MAX Graph Unavailable** → PyTorch fallback
2. **GPU Memory Insufficient** → Batch size reduction
3. **Mojo Kernel Issues** → Python implementation
4. **CUDA Unavailable** → CPU processing mode

### Recovery Mechanisms

- Automatic retry on transient failures
- Memory cleanup on OOM conditions  
- Model reloading on corruption detection
- Configuration validation and correction

## Monitoring and Observability

### Performance Metrics

- **Throughput**: Frames per second processed
- **Latency**: End-to-end processing time per frame
- **Resource Usage**: CPU, GPU, memory utilization
- **Detection Quality**: Precision, recall, mAP scores

### Logging and Debugging

- **Structured Logging**: JSON format with context
- **Performance Profiling**: Built-in timing and memory tracking
- **Error Tracking**: Detailed stack traces and context
- **Debug Visualization**: Intermediate result inspection

## Future Enhancements

### Planned Features

1. **Multi-Stream Processing**: Concurrent video streams
2. **Real-Time Streaming**: RTMP/WebRTC input support
3. **Custom Model Support**: User-provided model integration
4. **Advanced Analytics**: Object tracking and behavior analysis
5. **Edge Deployment**: Optimized builds for edge devices

### Research Directions

1. **Kernel Fusion**: Combined preprocessing and inference kernels
2. **Dynamic Batching**: Adaptive batch sizing based on load
3. **Model Compression**: Quantization and pruning integration
4. **Hardware Specialization**: Custom accelerator support 