"""
Base detector class and common utilities.
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Any
import numpy as np
import cv2


class BaseDetector(ABC):
    """Abstract base class for all deepfake detectors."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the detector.
        
        Args:
            threshold: Detection threshold for classification
        """
        self.threshold = threshold
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass
    
    @abstractmethod
    def predict(self, input_data: Union[np.ndarray, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict if input is a deepfake.
        
        Args:
            input_data: Input image/video data or path
            
        Returns:
            Tuple of (confidence_score, metadata)
        """
        pass
    
    def classify(self, input_data: Union[np.ndarray, str]) -> bool:
        """
        Classify input as deepfake or real.
        
        Args:
            input_data: Input image/video data or path
            
        Returns:
            True if deepfake detected, False otherwise
        """
        confidence, _ = self.predict(input_data)
        return confidence > self.threshold


def preprocess_image(image: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Preprocess image for detection.
    
    Args:
        image: Input image array
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image array
    """
    # Resize image
    resized = cv2.resize(image, (target_size, target_size))
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert BGR to RGB if needed
    if len(normalized.shape) == 3 and normalized.shape[2] == 3:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
    
    return normalized


def extract_faces(image: np.ndarray) -> list:
    """
    Extract face regions from image.
    
    Args:
        image: Input image array
        
    Returns:
        List of face region coordinates [(x, y, w, h), ...]
    """
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces.tolist() if len(faces) > 0 else []