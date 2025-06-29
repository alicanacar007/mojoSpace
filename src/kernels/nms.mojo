"""
Non-Maximum Suppression (NMS) Kernel - GPU-accelerated NMS for object detection
Efficiently removes overlapping bounding boxes based on IoU threshold
"""

from tensor import Tensor, TensorSpec
from algorithm import vectorize, parallelize
from math import sqrt, max, min
from memory import memset_zero
from sys import simdwidthof
from runtime.llcl import Runtime

struct NMSConfig:
    var iou_threshold: Float32
    var score_threshold: Float32
    var max_detections: Int
    var parallel_threshold: Int
    
    fn __init__(inout self, iou_threshold: Float32 = 0.5, score_threshold: Float32 = 0.5, max_detections: Int = 100, parallel_threshold: Int = 50):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_threshold = parallel_threshold

struct BoundingBox:
    var x1: Float32
    var y1: Float32
    var x2: Float32
    var y2: Float32
    var score: Float32
    var class_id: Int32
    
    fn __init__(inout self, x1: Float32, y1: Float32, x2: Float32, y2: Float32, score: Float32, class_id: Int32):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.class_id = class_id
    
    fn area(self) -> Float32:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

struct NMSKernel:
    var config: NMSConfig
    var runtime: Runtime
    
    fn __init__(inout self, config: NMSConfig):
        self.config = config
        self.runtime = Runtime()
    
    fn apply_nms(self, detections: Tensor[DType.float32]) -> Tensor[DType.float32]:
        """
        Apply Non-Maximum Suppression to detection results
        Args:
            detections: Tensor of shape [N, 6] containing [x1, y1, x2, y2, score, class_id]
        Returns:
            Filtered detections after NMS
        """
        let num_detections = detections.shape()[0]
        
        # Pre-filter by score threshold
        var filtered_indices = self._filter_by_score(detections)
        let num_filtered = len(filtered_indices)
        
        if num_filtered == 0:
            return Tensor[DType.float32](TensorSpec(DType.float32, 0, 6))
        
        # Sort by confidence scores (descending)
        var sorted_indices = self._sort_by_confidence(detections, filtered_indices)
        
        # Apply NMS algorithm
        var keep_mask = Tensor[DType.bool](TensorSpec(DType.bool, num_filtered))
        memset_zero(keep_mask.data(), num_filtered)
        
        var num_kept = 0
        for i in range(num_filtered):
            if num_kept >= self.config.max_detections:
                break
                
            let current_idx = sorted_indices[i]
            if not keep_mask[i]:
                continue
                
            keep_mask[i] = True
            num_kept += 1
            
            # Suppress overlapping boxes
            for j in range(i + 1, num_filtered):
                if keep_mask[j]:
                    let other_idx = sorted_indices[j]
                    let iou = self._calculate_iou(detections, current_idx, other_idx)
                    
                    if iou > self.config.iou_threshold:
                        keep_mask[j] = False
        
        # Collect final results
        return self._collect_final_detections(detections, sorted_indices, keep_mask, num_kept)
    
    fn _filter_by_score(self, detections: Tensor[DType.float32]) -> List[Int]:
        """Filter detections by minimum score threshold"""
        var filtered_indices = List[Int]()
        let num_detections = detections.shape()[0]
        
        for i in range(num_detections):
            let score = detections[i * 6 + 4]  # Score is at index 4
            if score >= self.config.score_threshold:
                filtered_indices.append(i)
        
        return filtered_indices
    
    fn _sort_by_confidence(self, detections: Tensor[DType.float32], indices: List[Int]) -> List[Int]:
        """Sort detection indices by confidence score (descending)"""
        var sorted_indices = List[Int]()
        for i in range(len(indices)):
            sorted_indices.append(indices[i])
        
        # Simple insertion sort (can be optimized with quicksort for large datasets)
        for i in range(1, len(sorted_indices)):
            let key_idx = sorted_indices[i]
            let key_score = detections[key_idx * 6 + 4]
            var j = i - 1
            
            while j >= 0 and detections[sorted_indices[j] * 6 + 4] < key_score:
                sorted_indices[j + 1] = sorted_indices[j]
                j -= 1
            
            sorted_indices[j + 1] = key_idx
        
        return sorted_indices
    
    fn _calculate_iou(self, detections: Tensor[DType.float32], idx1: Int, idx2: Int) -> Float32:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract box coordinates
        let x1_1 = detections[idx1 * 6 + 0]
        let y1_1 = detections[idx1 * 6 + 1]
        let x2_1 = detections[idx1 * 6 + 2]
        let y2_1 = detections[idx1 * 6 + 3]
        
        let x1_2 = detections[idx2 * 6 + 0]
        let y1_2 = detections[idx2 * 6 + 1]
        let x2_2 = detections[idx2 * 6 + 2]
        let y2_2 = detections[idx2 * 6 + 3]
        
        # Calculate intersection
        let x1_i = max(x1_1, x1_2)
        let y1_i = max(y1_1, y1_2)
        let x2_i = min(x2_1, x2_2)
        let y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        let intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        let area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        let area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        let union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area
    
    fn _collect_final_detections(self, detections: Tensor[DType.float32], sorted_indices: List[Int], keep_mask: Tensor[DType.bool], num_kept: Int) -> Tensor[DType.float32]:
        """Collect final detections after NMS"""
        let output_spec = TensorSpec(DType.float32, num_kept, 6)
        var output = Tensor[DType.float32](output_spec)
        
        var output_idx = 0
        for i in range(len(sorted_indices)):
            if keep_mask[i] and output_idx < num_kept:
                let src_idx = sorted_indices[i]
                for j in range(6):
                    output[output_idx * 6 + j] = detections[src_idx * 6 + j]
                output_idx += 1
        
        return output

# Optimized parallel NMS for large batch sizes
fn parallel_nms_batch(detections_batch: Tensor[DType.float32], config: NMSConfig) -> List[Tensor[DType.float32]]:
    """Apply NMS in parallel across batch dimension"""
    let batch_size = detections_batch.shape()[0]
    var results = List[Tensor[DType.float32]]()
    
    @parameter
    fn process_single_batch(batch_idx: Int):
        let nms_kernel = NMSKernel(config)
        # Extract single batch detection
        let single_batch = detections_batch[batch_idx]  # This would need proper tensor slicing
        let result = nms_kernel.apply_nms(single_batch)
        results.append(result)
    
    parallelize[process_single_batch](batch_size)
    return results

# Export the kernel for Python interop
alias nms_kernel = NMSKernel 