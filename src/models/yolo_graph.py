"""
YOLO Graph - MAX Graph API wrapper for YOLOv10 inference
Provides GPU-accelerated YOLO inference using MAX Graph
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import torch
import torchvision.transforms as transforms
from pathlib import Path

try:
    import max_graph
    from modular_sdk import MAX_GRAPH_AVAILABLE
except ImportError:
    MAX_GRAPH_AVAILABLE = False
    print("MAX Graph SDK not available, falling back to PyTorch")

class YOLOGraphModel:
    """MAX Graph wrapper for YOLOv10 inference"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        input_size: Tuple[int, int] = (640, 640),
        num_classes: int = 80,
        conf_threshold: float = 0.25,
        batch_size: int = 1
    ):
        self.model_path = model_path
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        
        self.model = None
        self.graph = None
        self.session = None
        self.class_names = self._load_class_names()
        
        self.preprocessing = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize MAX Graph model or fallback to PyTorch"""
        if MAX_GRAPH_AVAILABLE and self.model_path:
            try:
                self._initialize_max_graph()
            except Exception as e:
                print(f"Failed to initialize MAX Graph: {e}")
                print("Falling back to PyTorch implementation")
                self._initialize_pytorch_fallback()
        else:
            self._initialize_pytorch_fallback()
    
    def _initialize_max_graph(self):
        """Initialize MAX Graph model"""
        print("Initializing MAX Graph YOLOv10 model...")
        
        # Create MAX Graph session
        self.session = max_graph.Session()
        
        # Load model graph
        if self.model_path.endswith('.onnx'):
            self.graph = max_graph.load_onnx(self.model_path)
        else:
            # Assume PyTorch model, convert to ONNX first
            self.graph = self._convert_pytorch_to_max_graph()
        
        # Optimize graph for inference
        self.graph = max_graph.optimize_graph(
            self.graph,
            target_device=self.device,
            optimization_level=3
        )
        
        # Compile model
        self.model = self.session.compile(self.graph)
        print("MAX Graph model initialized successfully")
    
    def _initialize_pytorch_fallback(self):
        """Initialize PyTorch YOLOv10 model as fallback"""
        print("Initializing PyTorch YOLOv10 fallback...")
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov10n.pt')  # Use nano model for speed
            
            # Move model to GPU if available
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to(self.device)
                print(f"Model moved to {self.device}")
            
            print("PyTorch YOLO model loaded successfully")
        except ImportError:
            print("Ultralytics not available, using dummy model")
            self.model = None
    
    def _convert_pytorch_to_max_graph(self):
        """Convert PyTorch model to MAX Graph format"""
        # This is a placeholder for actual conversion logic
        # In practice, you would use the MAX Graph conversion utilities
        raise NotImplementedError("PyTorch to MAX Graph conversion not implemented")
    
    def _load_class_names(self) -> List[str]:
        """Load COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def preprocess_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess batch of frames for inference"""
        batch_tensors = []
        
        for frame in frames:
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame_rgb = frame[:, :, ::-1]  # BGR to RGB
            else:
                frame_rgb = frame
            
            # Convert to PIL Image for transforms
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply preprocessing
            tensor = self.preprocessing(pil_image)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors)
        
        if self.device == "cuda" and torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        
        return batch_tensor
    
    def inference(self, input_tensor: torch.Tensor) -> List[np.ndarray]:
        """Run inference on preprocessed tensor"""
        if self.model is None:
            # Return dummy detections for testing
            return self._generate_dummy_detections(input_tensor.shape[0])
        
        if MAX_GRAPH_AVAILABLE and hasattr(self, 'session'):
            return self._inference_max_graph(input_tensor)
        else:
            return self._inference_pytorch(input_tensor)
    
    def _inference_max_graph(self, input_tensor: torch.Tensor) -> List[np.ndarray]:
        """Run inference using MAX Graph"""
        # Convert PyTorch tensor to MAX Graph tensor
        input_array = input_tensor.cpu().numpy()
        
        # Run inference
        outputs = self.model.run(input_array)
        
        # Process outputs (YOLOv10 format)
        detections_list = []
        for batch_idx in range(input_tensor.shape[0]):
            batch_output = outputs[batch_idx]
            detections = self._process_yolo_output(batch_output)
            detections_list.append(detections)
        
        return detections_list
    
    def _inference_pytorch(self, input_tensor: torch.Tensor) -> List[np.ndarray]:
        """Run inference using PyTorch fallback"""
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                # Ultralytics YOLO
                results = self.model.predict(input_tensor, verbose=False)
                detections_list = []
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        # Combine into detection format [x1, y1, x2, y2, score, class_id]
                        detections = np.column_stack([boxes, scores, classes])
                    else:
                        detections = np.empty((0, 6))
                    
                    detections_list.append(detections)
                
                return detections_list
            else:
                # Generic PyTorch model
                outputs = self.model(input_tensor)
                return self._process_pytorch_outputs(outputs)
    
    def _process_yolo_output(self, output: np.ndarray) -> np.ndarray:
        """Process YOLO model output to detection format"""
        # YOLOv10 output format: [batch, detections, (x, y, w, h, conf, class_probs...)]
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        detections = []
        for detection in output:
            if len(detection) < 5:
                continue
            
            x, y, w, h, conf = detection[:5]
            
            # Convert center format to corner format
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            # Get class probabilities
            class_probs = detection[5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # Final confidence
            final_conf = conf * class_conf
            
            if final_conf >= self.conf_threshold:
                detections.append([x1, y1, x2, y2, final_conf, class_id])
        
        return np.array(detections) if detections else np.empty((0, 6))
    
    def _generate_dummy_detections(self, batch_size: int) -> List[np.ndarray]:
        """Generate dummy detections for testing"""
        detections_list = []
        
        for _ in range(batch_size):
            # Generate random detections
            num_detections = np.random.randint(0, 5)
            if num_detections > 0:
                detections = np.random.rand(num_detections, 6)
                # Scale to reasonable values
                detections[:, :4] *= self.input_size[0]  # Bounding boxes
                detections[:, 4] = np.random.uniform(0.5, 1.0, num_detections)  # Confidence
                detections[:, 5] = np.random.randint(0, self.num_classes, num_detections)  # Class
            else:
                detections = np.empty((0, 6))
            
            detections_list.append(detections)
        
        return detections_list
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"Unknown_{class_id}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_type": "MAX Graph" if MAX_GRAPH_AVAILABLE and hasattr(self, 'session') else "PyTorch",
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "batch_size": self.batch_size,
            "device": self.device,
            "conf_threshold": self.conf_threshold
        } 