"""
Main detection pipeline that combines multiple detection techniques.
"""

import numpy as np
from typing import Union, List, Dict, Any, Tuple
from .detectors import CNNDetector, TemporalDetector, LandmarkDetector, FrequencyDetector
from .config import DetectionConfig


class DeepfakeDetectionPipeline:
    """Main pipeline for deepfake detection using multiple techniques."""
    
    def __init__(self, config: DetectionConfig = None):
        """
        Initialize the detection pipeline.
        
        Args:
            config: Detection configuration
        """
        self.config = config or DetectionConfig()
        self.detectors = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self) -> None:
        """Initialize all detection modules."""
        # CNN Detector
        self.detectors["cnn"] = CNNDetector(
            model_path=self.config.cnn_model_path,
            threshold=self.config.cnn_threshold,
            image_size=self.config.cnn_image_size
        )
        
        # Temporal Detector
        self.detectors["temporal"] = TemporalDetector(
            threshold=self.config.temporal_threshold,
            frame_count=self.config.temporal_frame_count
        )
        
        # Landmark Detector
        self.detectors["landmark"] = LandmarkDetector(
            threshold=self.config.landmark_threshold,
            max_distance=self.config.landmark_max_distance
        )
        
        # Frequency Detector
        self.detectors["frequency"] = FrequencyDetector(
            threshold=self.config.frequency_threshold,
            frequency_bands=self.config.frequency_bands
        )
    
    def detect(self, 
               input_data: Union[np.ndarray, str], 
               methods: List[str] = None) -> Dict[str, Any]:
        """
        Detect deepfakes using specified methods.
        
        Args:
            input_data: Input image/video array or path
            methods: List of detection methods to use. If None, uses all methods.
            
        Returns:
            Dictionary containing detection results
        """
        if methods is None:
            methods = ["cnn", "temporal", "landmark", "frequency"]
        
        results = {}
        confidences = {}
        
        for method in methods:
            if method not in self.detectors:
                continue
                
            try:
                detector = self.detectors[method]
                confidence, metadata = detector.predict(input_data)
                
                results[method] = {
                    "confidence": confidence,
                    "is_deepfake": confidence > detector.threshold,
                    "metadata": metadata
                }
                confidences[method] = confidence
                
            except Exception as e:
                results[method] = {
                    "error": str(e),
                    "confidence": 0.0,
                    "is_deepfake": False,
                    "metadata": {}
                }
                confidences[method] = 0.0
        
        # Calculate ensemble result
        ensemble_result = self._calculate_ensemble_result(confidences)
        
        return {
            "individual_results": results,
            "ensemble_confidence": ensemble_result["confidence"],
            "ensemble_prediction": ensemble_result["is_deepfake"],
            "ensemble_weights": self.config.ensemble_weights,
            "overall_confidence_threshold": self.config.confidence_threshold
        }
    
    def _calculate_ensemble_result(self, confidences: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate ensemble result from individual detector confidences.
        
        Args:
            confidences: Dictionary of method names to confidence scores
            
        Returns:
            Dictionary with ensemble confidence and prediction
        """
        if not confidences:
            return {"confidence": 0.0, "is_deepfake": False}
        
        # Weighted ensemble
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, confidence in confidences.items():
            if method in self.config.ensemble_weights:
                weight = self.config.ensemble_weights[method]
                weighted_sum += confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            # Fallback to simple average
            ensemble_confidence = np.mean(list(confidences.values()))
        else:
            ensemble_confidence = weighted_sum / total_weight
        
        is_deepfake = ensemble_confidence > self.config.confidence_threshold
        
        return {
            "confidence": ensemble_confidence,
            "is_deepfake": is_deepfake
        }
    
    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect deepfakes in a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Detection results
        """
        # Use image-appropriate methods
        methods = ["cnn", "landmark", "frequency"]
        return self.detect(image_path, methods)
    
    def detect_video(self, video_path: str) -> Dict[str, Any]:
        """
        Detect deepfakes in a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Detection results
        """
        # Use all methods for video analysis
        methods = ["cnn", "temporal", "landmark", "frequency"]
        return self.detect(video_path, methods)
    
    def get_detector_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all loaded detectors.
        
        Returns:
            Dictionary with detector information
        """
        info = {}
        
        for name, detector in self.detectors.items():
            info[name] = {
                "type": type(detector).__name__,
                "threshold": detector.threshold,
                "is_loaded": detector.is_loaded,
                "weight": self.config.ensemble_weights.get(name, 0.0)
            }
        
        return info