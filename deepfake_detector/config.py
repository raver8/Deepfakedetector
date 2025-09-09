"""
Configuration settings for deepfake detection.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DetectionConfig:
    """Configuration class for deepfake detection parameters."""
    
    # CNN Detection parameters
    cnn_model_path: str = "models/cnn_detector.pth"
    cnn_threshold: float = 0.5
    cnn_image_size: int = 224
    
    # Temporal Detection parameters
    temporal_frame_count: int = 30
    temporal_threshold: float = 0.3
    
    # Landmark Detection parameters
    landmark_threshold: float = 0.4
    landmark_max_distance: float = 50.0
    
    # Frequency Detection parameters
    frequency_threshold: float = 0.6
    frequency_bands: List[int] = None
    
    # General parameters
    ensemble_weights: Dict[str, float] = None
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        """Initialize default values for complex fields."""
        if self.frequency_bands is None:
            self.frequency_bands = [0, 10, 50, 100]
            
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "cnn": 0.4,
                "temporal": 0.2,
                "landmark": 0.2,
                "frequency": 0.2
            }