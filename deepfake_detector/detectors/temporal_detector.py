"""
Temporal inconsistency detector for video deepfakes.
"""

import cv2
import numpy as np
from typing import Union, Tuple, Dict, Any, List
from .base import BaseDetector, extract_faces


class TemporalDetector(BaseDetector):
    """Detector that analyzes temporal inconsistencies in video frames."""
    
    def __init__(self, threshold: float = 0.3, frame_count: int = 30):
        """
        Initialize temporal detector.
        
        Args:
            threshold: Detection threshold
            frame_count: Number of frames to analyze
        """
        super().__init__(threshold)
        self.frame_count = frame_count
        self.is_loaded = True  # No model to load
    
    def load_model(self) -> None:
        """No model to load for temporal analysis."""
        self.is_loaded = True
    
    def predict(self, input_data: Union[np.ndarray, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict if video has temporal inconsistencies.
        
        Args:
            input_data: Path to video file or video array
            
        Returns:
            Tuple of (confidence_score, metadata)
        """
        if isinstance(input_data, str):
            frames = self._extract_frames_from_video(input_data)
        else:
            # Assume input is a list of frames or single frame
            if len(input_data.shape) == 3:
                # Single frame - cannot do temporal analysis
                return 0.0, {"error": "Single frame provided, need video for temporal analysis"}
            frames = input_data
        
        if len(frames) < 2:
            return 0.0, {"error": "Need at least 2 frames for temporal analysis"}
        
        # Analyze temporal inconsistencies
        inconsistency_scores = []
        face_tracking_scores = []
        optical_flow_scores = []
        
        for i in range(len(frames) - 1):
            # Face tracking consistency
            face_score = self._analyze_face_tracking_consistency(frames[i], frames[i + 1])
            face_tracking_scores.append(face_score)
            
            # Optical flow analysis
            flow_score = self._analyze_optical_flow(frames[i], frames[i + 1])
            optical_flow_scores.append(flow_score)
            
            # Combined inconsistency score
            combined_score = (face_score + flow_score) / 2
            inconsistency_scores.append(combined_score)
        
        # Overall confidence is the mean inconsistency score
        confidence = np.mean(inconsistency_scores)
        
        metadata = {
            "frames_analyzed": len(frames),
            "face_tracking_scores": face_tracking_scores,
            "optical_flow_scores": optical_flow_scores,
            "inconsistency_scores": inconsistency_scores,
            "mean_inconsistency": confidence,
            "max_inconsistency": np.max(inconsistency_scores),
            "method": "temporal_analysis"
        }
        
        return confidence, metadata
    
    def _extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame arrays
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // self.frame_count)
        
        frame_idx = 0
        while len(frames) < self.frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            frames.append(frame)
            frame_idx += frame_step
        
        cap.release()
        return frames
    
    def _analyze_face_tracking_consistency(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Analyze consistency of face positions between frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Inconsistency score (higher = more inconsistent)
        """
        faces1 = extract_faces(frame1)
        faces2 = extract_faces(frame2)
        
        if not faces1 or not faces2:
            return 0.0  # No faces to track
        
        if len(faces1) != len(faces2):
            # Different number of faces detected - potential inconsistency
            return 0.7
        
        # Calculate position differences for each face
        position_diffs = []
        for i in range(min(len(faces1), len(faces2))):
            x1, y1, w1, h1 = faces1[i]
            x2, y2, w2, h2 = faces2[i]
            
            # Calculate center position difference
            center1 = (x1 + w1/2, y1 + h1/2)
            center2 = (x2 + w2/2, y2 + h2/2)
            
            position_diff = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            size_diff = abs((w1 * h1) - (w2 * h2)) / max(w1 * h1, w2 * h2)
            
            position_diffs.append(position_diff + size_diff * 10)
        
        # Normalize by image size
        height, width = frame1.shape[:2]
        max_distance = np.sqrt(width**2 + height**2)
        
        mean_diff = np.mean(position_diffs) / max_distance
        return min(mean_diff * 2, 1.0)  # Scale and cap at 1.0
    
    def _analyze_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Analyze optical flow between consecutive frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Inconsistency score based on optical flow
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        if flow[0] is None:
            return 0.0
        
        # Analyze flow patterns
        good_points = flow[1].ravel() == 1
        
        if np.sum(good_points) == 0:
            return 0.5  # No good tracking points
        
        # Calculate flow magnitudes
        flow_vectors = flow[0][good_points]
        if len(flow_vectors) == 0:
            return 0.5
        
        flow_magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
        
        # High variance in flow magnitudes may indicate inconsistencies
        flow_variance = np.var(flow_magnitudes)
        mean_magnitude = np.mean(flow_magnitudes)
        
        # Normalize inconsistency score
        if mean_magnitude > 0:
            inconsistency = flow_variance / (mean_magnitude**2)
        else:
            inconsistency = 0.0
        
        return min(inconsistency * 0.1, 1.0)  # Scale and cap at 1.0