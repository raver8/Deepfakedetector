"""
Deepfake Detection System

A comprehensive toolkit for detecting deepfake videos and images using various techniques.
"""

__version__ = "1.0.0"
__author__ = "Deepfake Detection Team"

from .detectors import CNNDetector, TemporalDetector, LandmarkDetector, FrequencyDetector
from .pipeline import DeepfakeDetectionPipeline
from .config import DetectionConfig

__all__ = [
    "CNNDetector",
    "TemporalDetector", 
    "LandmarkDetector",
    "FrequencyDetector",
    "DeepfakeDetectionPipeline",
    "DetectionConfig"
]