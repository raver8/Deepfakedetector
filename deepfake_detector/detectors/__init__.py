"""
Detector modules initialization.
"""

from .base import BaseDetector
from .cnn_detector import CNNDetector
from .temporal_detector import TemporalDetector
from .landmark_detector import LandmarkDetector
from .frequency_detector import FrequencyDetector

__all__ = [
    "BaseDetector",
    "CNNDetector",
    "TemporalDetector",
    "LandmarkDetector", 
    "FrequencyDetector"
]