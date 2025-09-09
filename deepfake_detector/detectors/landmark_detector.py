"""
Face landmark-based deepfake detector.
"""

import cv2
import numpy as np
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None
    
from typing import Union, Tuple, Dict, Any, List
from .base import BaseDetector


class LandmarkDetector(BaseDetector):
    """Detector that analyzes facial landmarks for deepfake detection."""
    
    def __init__(self, threshold: float = 0.4, max_distance: float = 50.0):
        """
        Initialize landmark detector.
        
        Args:
            threshold: Detection threshold
            max_distance: Maximum allowed landmark deviation
        """
        super().__init__(threshold)
        self.max_distance = max_distance
        self.face_detector = None
        self.landmark_predictor = None
        
    def load_model(self) -> None:
        """Load face detection and landmark prediction models."""
        try:
            if not DLIB_AVAILABLE:
                print("Warning: dlib not available, using OpenCV face detection only")
                self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.landmark_predictor = None
                self.is_loaded = True
                return
                
            # Initialize dlib face detector
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Try to load landmark predictor (requires shape_predictor_68_face_landmarks.dat)
            try:
                self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            except RuntimeError:
                # Fallback to simpler landmark detection if dlib model not available
                print("Warning: dlib landmark model not found, using simplified landmark detection")
                self.landmark_predictor = None
                
            self.is_loaded = True
            
        except Exception as e:
            print(f"Error loading landmark detector: {e}")
            self.is_loaded = False
    
    def predict(self, input_data: Union[np.ndarray, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict if input contains deepfake based on landmark analysis.
        
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
        
        # Convert to grayscale for landmark detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        if DLIB_AVAILABLE and hasattr(self.face_detector, '__call__'):
            # Using dlib detector
            faces = self.face_detector(gray)
        else:
            # Using OpenCV detector 
            faces_cv = self.face_detector.detectMultiScale(gray, 1.1, 4)
            # Convert to dlib rectangle format
            faces = []
            for (x, y, w, h) in faces_cv:
                # Create a simple rectangle object
                class Rectangle:
                    def __init__(self, x, y, w, h):
                        self._left = x
                        self._top = y
                        self._right = x + w
                        self._bottom = y + h
                    
                    def left(self): return self._left
                    def top(self): return self._top
                    def right(self): return self._right
                    def bottom(self): return self._bottom
                    def width(self): return self._right - self._left
                    def height(self): return self._bottom - self._top
                
                faces.append(Rectangle(x, y, w, h))
        
        if len(faces) == 0:
            return 0.0, {"error": "No faces detected for landmark analysis"}
        
        landmark_scores = []
        face_landmarks = []
        asymmetry_scores = []
        
        for face in faces:
            if self.landmark_predictor is not None:
                # Use dlib landmark predictor
                landmarks = self.landmark_predictor(gray, face)
                landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            else:
                # Use simplified landmark detection
                landmark_points = self._simple_landmark_detection(gray, face)
            
            if landmark_points:
                # Analyze landmark consistency
                consistency_score = self._analyze_landmark_consistency(landmark_points)
                asymmetry_score = self._analyze_facial_asymmetry(landmark_points)
                
                landmark_scores.append(consistency_score)
                asymmetry_scores.append(asymmetry_score)
                face_landmarks.append(landmark_points)
        
        if not landmark_scores:
            return 0.0, {"error": "Could not extract landmarks"}
        
        # Combine scores
        mean_consistency = np.mean(landmark_scores)
        mean_asymmetry = np.mean(asymmetry_scores)
        confidence = (mean_consistency + mean_asymmetry) / 2
        
        metadata = {
            "faces_analyzed": len(faces),
            "landmark_consistency_scores": landmark_scores,
            "asymmetry_scores": asymmetry_scores,
            "face_landmarks": face_landmarks,
            "mean_consistency": mean_consistency,
            "mean_asymmetry": mean_asymmetry,
            "method": "landmark_analysis"
        }
        
        return confidence, metadata
    
    def _simple_landmark_detection(self, gray_image: np.ndarray, face_rect) -> List[Tuple[int, int]]:
        """
        Simple landmark detection fallback when dlib model is not available.
        
        Args:
            gray_image: Grayscale image
            face_rect: Face rectangle from dlib
            
        Returns:
            List of landmark points
        """
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        
        # Define approximate landmark positions relative to face rectangle
        landmarks = [
            # Face outline (simplified)
            (x, y + h//4), (x, y + h//2), (x, y + 3*h//4),  # Left side
            (x + w//4, y + h), (x + w//2, y + h), (x + 3*w//4, y + h),  # Bottom
            (x + w, y + 3*h//4), (x + w, y + h//2), (x + w, y + h//4),  # Right side
            (x + 3*w//4, y), (x + w//2, y), (x + w//4, y),  # Top
            
            # Eyes (approximate)
            (x + w//4, y + h//3), (x + 3*w//4, y + h//3),  # Eye centers
            
            # Nose (approximate)
            (x + w//2, y + h//2),
            
            # Mouth (approximate)
            (x + w//2, y + 2*h//3),
        ]
        
        return landmarks
    
    def _analyze_landmark_consistency(self, landmarks: List[Tuple[int, int]]) -> float:
        """
        Analyze the consistency of facial landmarks.
        
        Args:
            landmarks: List of landmark points
            
        Returns:
            Inconsistency score (higher = more inconsistent)
        """
        if len(landmarks) < 4:
            return 0.0
        
        # Calculate expected symmetry for bilateral landmarks
        inconsistencies = []
        
        # Assuming we have at least some symmetric points
        points = np.array(landmarks)
        center_x = np.mean(points[:, 0])
        
        # Analyze symmetry by comparing left and right sides
        left_points = points[points[:, 0] < center_x]
        right_points = points[points[:, 0] > center_x]
        
        if len(left_points) > 0 and len(right_points) > 0:
            # Mirror left points to right side
            mirrored_left = left_points.copy()
            mirrored_left[:, 0] = 2 * center_x - mirrored_left[:, 0]
            
            # Find closest matches and calculate distances
            for left_point in mirrored_left:
                if len(right_points) > 0:
                    distances = np.sqrt(np.sum((right_points - left_point)**2, axis=1))
                    min_distance = np.min(distances)
                    inconsistencies.append(min_distance)
        
        # Calculate geometric inconsistencies
        if len(points) >= 4:
            # Check if points form reasonable geometric relationships
            distances = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = np.sqrt(np.sum((points[i] - points[j])**2))
                    distances.append(dist)
            
            # High variance in distances might indicate inconsistencies
            if distances:
                distance_variance = np.var(distances)
                mean_distance = np.mean(distances)
                if mean_distance > 0:
                    inconsistencies.append(distance_variance / mean_distance)
        
        if not inconsistencies:
            return 0.0
        
        # Normalize inconsistency score
        mean_inconsistency = np.mean(inconsistencies)
        normalized_score = min(mean_inconsistency / self.max_distance, 1.0)
        
        return normalized_score
    
    def _analyze_facial_asymmetry(self, landmarks: List[Tuple[int, int]]) -> float:
        """
        Analyze facial asymmetry which might indicate deepfakes.
        
        Args:
            landmarks: List of landmark points
            
        Returns:
            Asymmetry score (higher = more asymmetric)
        """
        if len(landmarks) < 6:
            return 0.0
        
        points = np.array(landmarks)
        center_x = np.mean(points[:, 0])
        
        # Separate left and right halves
        left_points = points[points[:, 0] < center_x]
        right_points = points[points[:, 0] > center_x]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return 0.0
        
        # Calculate asymmetry metrics
        left_variance = np.var(left_points, axis=0)
        right_variance = np.var(right_points, axis=0)
        
        # Compare distributions of points on each side
        asymmetry_score = 0.0
        
        # Variance asymmetry
        var_diff = np.abs(left_variance - right_variance)
        asymmetry_score += np.mean(var_diff) / 100.0  # Normalize
        
        # Count asymmetry
        count_diff = abs(len(left_points) - len(right_points))
        asymmetry_score += count_diff / len(landmarks)
        
        return min(asymmetry_score, 1.0)