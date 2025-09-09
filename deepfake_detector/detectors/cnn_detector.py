"""
CNN-based deepfake detector.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Union, Tuple, Dict, Any
import numpy as np
import cv2
from .base import BaseDetector, preprocess_image, extract_faces


class CNNDetectorModel(nn.Module):
    """Simple CNN model for deepfake detection."""
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize CNN model.
        
        Args:
            num_classes: Number of output classes (2 for real/fake)
        """
        super(CNNDetectorModel, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNDetector(BaseDetector):
    """CNN-based deepfake detector."""
    
    def __init__(self, model_path: str = None, threshold: float = 0.5, image_size: int = 224):
        """
        Initialize CNN detector.
        
        Args:
            model_path: Path to pretrained model (optional)
            threshold: Detection threshold
            image_size: Input image size
        """
        super().__init__(threshold)
        self.model_path = model_path
        self.image_size = image_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self) -> None:
        """Load the CNN model."""
        self.model = CNNDetectorModel(num_classes=2)
        
        if self.model_path and torch.cuda.is_available():
            try:
                # Try to load pretrained weights if available
                self.model.load_state_dict(torch.load(self.model_path))
                print(f"Loaded pretrained model from {self.model_path}")
            except FileNotFoundError:
                print(f"Model file {self.model_path} not found, using random weights")
        else:
            print("Using model with random weights (for demonstration)")
        
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
    
    def predict(self, input_data: Union[np.ndarray, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict if input is a deepfake using CNN.
        
        Args:
            input_data: Input image array or path to image
            
        Returns:
            Tuple of (confidence_score, metadata)
        """
        if not self.is_loaded:
            self.load_model()
        
        # Load image if path is provided
        if isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                raise ValueError(f"Could not load image from {input_data}")
        else:
            image = input_data.copy()
        
        # Extract faces from image
        faces = extract_faces(image)
        
        if not faces:
            # If no faces detected, analyze entire image
            processed_image = preprocess_image(image, self.image_size)
            confidence = self._predict_single_image(processed_image)
            
            metadata = {
                "faces_detected": 0,
                "face_regions": [],
                "method": "full_image"
            }
        else:
            # Analyze each detected face
            face_confidences = []
            face_regions = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = image[y:y+h, x:x+w]
                processed_face = preprocess_image(face_region, self.image_size)
                
                face_confidence = self._predict_single_image(processed_face)
                face_confidences.append(face_confidence)
                face_regions.append((x, y, w, h))
            
            # Take maximum confidence across all faces
            confidence = max(face_confidences)
            
            metadata = {
                "faces_detected": len(faces),
                "face_regions": face_regions,
                "face_confidences": face_confidences,
                "method": "face_analysis"
            }
        
        return confidence, metadata
    
    def _predict_single_image(self, image: np.ndarray) -> float:
        """
        Predict single preprocessed image.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Confidence score for deepfake detection
        """
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            # Get probability for fake class (index 1)
            confidence = outputs[0][1].item()
        
        return confidence