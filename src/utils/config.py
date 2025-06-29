"""
Configuration Management - Centralized configuration for the video object detection pipeline
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import yaml
from pathlib import Path

@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction"""
    target_fps: float = 30.0
    max_width: int = 1920
    max_height: int = 1080
    channels: int = 3
    batch_size: int = 8
    use_mojo_kernel: bool = True

@dataclass
class ModelConfig:
    """Configuration for YOLO model"""
    model_path: Optional[str] = None
    model_type: str = "yolov10n"  # yolov10n, yolov10s, yolov10m, yolov10l, yolov10x
    input_size: Tuple[int, int] = (640, 640)
    num_classes: int = 80
    conf_threshold: float = 0.25
    batch_size: int = 1
    device: str = "cuda"
    use_max_graph: bool = True

@dataclass
class NMSConfig:
    """Configuration for Non-Maximum Suppression"""
    iou_threshold: float = 0.5
    score_threshold: float = 0.5
    max_detections: int = 100
    use_mojo_kernel: bool = True
    parallel_threshold: int = 50

@dataclass
class VisualizationConfig:
    """Configuration for visualization and annotation"""
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    box_alpha: float = 0.3
    text_alpha: float = 1.0
    color_map: str = "coco"  # coco, random, class_based
    save_format: str = "mp4"  # mp4, avi, images
    fps: float = 30.0
    quality: str = "high"  # low, medium, high

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    measure_memory: bool = True
    measure_power: bool = False
    output_format: str = "json"  # json, csv, txt
    detailed_profiling: bool = False

@dataclass
class SystemConfig:
    """System-level configuration"""
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    max_memory_usage: str = "8GB"
    enable_cuda: bool = True
    enable_tensorrt: bool = False
    enable_mixed_precision: bool = True

@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    nms: NMSConfig = field(default_factory=NMSConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # I/O Configuration
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    demo_mode: bool = False
    verbose: bool = True
    
    # Pipeline control
    enable_frame_extraction: bool = True
    enable_object_detection: bool = True
    enable_nms: bool = True
    enable_visualization: bool = True
    enable_benchmarking: bool = False

class ConfigManager:
    """Configuration manager with file I/O and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = PipelineConfig()
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
        
        self._validate_config()
        self._setup_environment()
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self._update_config_from_dict(config_dict)
            print(f"Configuration loaded from {config_path}")
        
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
    
    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file"""
        try:
            config_dict = self._config_to_dict()
            
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            print(f"Configuration saved to {config_path}")
        
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def _update_config_from_dict(self, config_dict: dict):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self.config, section) and isinstance(values, dict):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
            elif hasattr(self.config, section):
                setattr(self.config, section, values)
    
    def _config_to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        config_dict = {}
        
        for field_name, field_value in self.config.__dict__.items():
            if hasattr(field_value, '__dict__'):
                config_dict[field_name] = field_value.__dict__.copy()
            else:
                config_dict[field_name] = field_value
        
        return config_dict
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate frame extraction
        if self.config.frame_extraction.target_fps <= 0:
            errors.append("Frame extraction target_fps must be positive")
        
        if self.config.frame_extraction.batch_size <= 0:
            errors.append("Frame extraction batch_size must be positive")
        
        # Validate model config
        if self.config.model.conf_threshold < 0 or self.config.model.conf_threshold > 1:
            errors.append("Model confidence threshold must be between 0 and 1")
        
        if self.config.model.batch_size <= 0:
            errors.append("Model batch_size must be positive")
        
        # Validate NMS config
        if self.config.nms.iou_threshold < 0 or self.config.nms.iou_threshold > 1:
            errors.append("NMS IoU threshold must be between 0 and 1")
        
        if self.config.nms.score_threshold < 0 or self.config.nms.score_threshold > 1:
            errors.append("NMS score threshold must be between 0 and 1")
        
        # Validate visualization config
        if self.config.visualization.fps <= 0:
            errors.append("Visualization FPS must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    def _setup_environment(self):
        """Setup environment based on configuration"""
        # Set CUDA device
        if self.config.system.enable_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Set number of threads
        os.environ['OMP_NUM_THREADS'] = str(self.config.system.num_workers)
        
        # Set memory settings
        if 'GB' in self.config.system.max_memory_usage:
            memory_gb = float(self.config.system.max_memory_usage.replace('GB', ''))
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{int(memory_gb * 1024)}'
    
    def get_config(self) -> PipelineConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested configuration (e.g., "model.conf_threshold")
                section, param = key.split('.', 1)
                if hasattr(self.config, section):
                    section_config = getattr(self.config, section)
                    if hasattr(section_config, param):
                        setattr(section_config, param, value)
            else:
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self._validate_config()
    
    def get_model_config_dict(self) -> dict:
        """Get model configuration as dictionary for easy passing"""
        return {
            'model_path': self.config.model.model_path,
            'device': self.config.model.device,
            'input_size': self.config.model.input_size,
            'num_classes': self.config.model.num_classes,
            'conf_threshold': self.config.model.conf_threshold,
            'batch_size': self.config.model.batch_size
        }
    
    def get_nms_config_dict(self) -> dict:
        """Get NMS configuration as dictionary"""
        return {
            'iou_threshold': self.config.nms.iou_threshold,
            'score_threshold': self.config.nms.score_threshold,
            'max_detections': self.config.nms.max_detections,
            'parallel_threshold': self.config.nms.parallel_threshold
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print("PIPELINE CONFIGURATION")
        print("=" * 50)
        
        config_dict = self._config_to_dict()
        for section, values in config_dict.items():
            print(f"\n[{section.upper()}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {section}: {values}")
        
        print("=" * 50)

# Default configuration instance
default_config_manager = ConfigManager()

# Utility functions
def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return default_config_manager.get_config()

def load_config(config_path: str) -> ConfigManager:
    """Load configuration from file"""
    return ConfigManager(config_path)

def create_sample_config(output_path: str):
    """Create a sample configuration file"""
    config_manager = ConfigManager()
    config_manager.save_to_file(output_path)
    print(f"Sample configuration created at {output_path}")

if __name__ == "__main__":
    # Create sample configuration
    create_sample_config("config/sample_config.yaml") 