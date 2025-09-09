"""
Frequency domain analysis detector for deepfakes.
"""

import cv2
import numpy as np
from scipy import signal
from typing import Union, Tuple, Dict, Any, List
from .base import BaseDetector, extract_faces


class FrequencyDetector(BaseDetector):
    """Detector that analyzes frequency domain characteristics for deepfake detection."""
    
    def __init__(self, threshold: float = 0.6, frequency_bands: List[int] = None):
        """
        Initialize frequency detector.
        
        Args:
            threshold: Detection threshold
            frequency_bands: Frequency bands to analyze [low, mid, high, very_high]
        """
        super().__init__(threshold)
        self.frequency_bands = frequency_bands or [0, 10, 50, 100]
        self.is_loaded = True  # No model to load
    
    def load_model(self) -> None:
        """No model to load for frequency analysis."""
        self.is_loaded = True
    
    def predict(self, input_data: Union[np.ndarray, str]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict if input contains deepfake based on frequency analysis.
        
        Args:
            input_data: Input image array or path to image
            
        Returns:
            Tuple of (confidence_score, metadata)
        """
        # Load image if path is provided
        if isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                raise ValueError(f"Could not load image from {input_data}")
        else:
            image = input_data.copy()
        
        # Extract faces for focused analysis
        faces = extract_faces(image)
        
        if not faces:
            # Analyze entire image if no faces detected
            frequency_features = self._analyze_frequency_domain(image)
            artifact_score = self._detect_compression_artifacts(image)
            
            metadata = {
                "faces_detected": 0,
                "method": "full_image_frequency",
                "frequency_features": frequency_features,
                "compression_artifacts": artifact_score
            }
        else:
            # Analyze each face region
            face_frequency_features = []
            face_artifact_scores = []
            
            for (x, y, w, h) in faces:
                face_region = image[y:y+h, x:x+w]
                
                freq_features = self._analyze_frequency_domain(face_region)
                artifact_score = self._detect_compression_artifacts(face_region)
                
                face_frequency_features.append(freq_features)
                face_artifact_scores.append(artifact_score)
            
            # Combine results across faces
            frequency_features = self._combine_frequency_features(face_frequency_features)
            artifact_score = np.mean(face_artifact_scores)
            
            metadata = {
                "faces_detected": len(faces),
                "face_regions": faces,
                "method": "face_frequency_analysis",
                "face_frequency_features": face_frequency_features,
                "face_artifact_scores": face_artifact_scores,
                "frequency_features": frequency_features,
                "compression_artifacts": artifact_score
            }
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(frequency_features, artifact_score)
        
        return confidence, metadata
    
    def _analyze_frequency_domain(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze frequency domain characteristics of the image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of frequency domain features
        """
        # Convert to grayscale for frequency analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply 2D FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        # Log transform for better visualization
        log_spectrum = np.log(magnitude_spectrum + 1)
        
        # Analyze different frequency bands
        h, w = log_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        features = {}
        
        # Low frequency (DC and near-DC components)
        low_freq_region = log_spectrum[center_h-10:center_h+10, center_w-10:center_w+10]
        features["low_freq_energy"] = np.mean(low_freq_region)
        
        # Mid frequency
        mid_freq_mask = self._create_frequency_mask(h, w, 10, 50)
        features["mid_freq_energy"] = np.mean(log_spectrum[mid_freq_mask])
        
        # High frequency
        high_freq_mask = self._create_frequency_mask(h, w, 50, 100)
        features["high_freq_energy"] = np.mean(log_spectrum[high_freq_mask])
        
        # Very high frequency (often affected by compression/generation)
        very_high_freq_mask = self._create_frequency_mask(h, w, 100, min(h, w) // 2)
        features["very_high_freq_energy"] = np.mean(log_spectrum[very_high_freq_mask])
        
        # Frequency distribution analysis
        features["freq_variance"] = np.var(log_spectrum)
        features["freq_skewness"] = self._calculate_skewness(log_spectrum)
        features["freq_kurtosis"] = self._calculate_kurtosis(log_spectrum)
        
        # Spectral centroid (weighted mean frequency)
        features["spectral_centroid"] = self._calculate_spectral_centroid(magnitude_spectrum)
        
        return features
    
    def _create_frequency_mask(self, h: int, w: int, inner_radius: int, outer_radius: int) -> np.ndarray:
        """
        Create a ring mask for frequency band analysis.
        
        Args:
            h: Height of the spectrum
            w: Width of the spectrum
            inner_radius: Inner radius of the ring
            outer_radius: Outer radius of the ring
            
        Returns:
            Boolean mask for the frequency band
        """
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Create ring mask
        mask = (distance >= inner_radius) & (distance < outer_radius)
        return mask
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data."""
        flat_data = data.flatten()
        mean = np.mean(flat_data)
        std = np.std(flat_data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((flat_data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        flat_data = data.flatten()
        mean = np.mean(flat_data)
        std = np.std(flat_data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((flat_data - mean) / std) ** 4) - 3
        return kurtosis
    
    def _calculate_spectral_centroid(self, magnitude_spectrum: np.ndarray) -> float:
        """Calculate the spectral centroid."""
        h, w = magnitude_spectrum.shape
        
        # Create frequency grids
        freq_h = np.arange(h)
        freq_w = np.arange(w)
        
        # Calculate weighted mean frequency
        total_energy = np.sum(magnitude_spectrum)
        
        if total_energy == 0:
            return 0.0
        
        centroid_h = np.sum(np.sum(magnitude_spectrum, axis=1) * freq_h) / total_energy
        centroid_w = np.sum(np.sum(magnitude_spectrum, axis=0) * freq_w) / total_energy
        
        # Return normalized centroid
        centroid = np.sqrt(centroid_h**2 + centroid_w**2)
        max_freq = np.sqrt(h**2 + w**2)
        
        return centroid / max_freq
    
    def _detect_compression_artifacts(self, image: np.ndarray) -> float:
        """
        Detect compression artifacts that might indicate deepfakes.
        
        Args:
            image: Input image array
            
        Returns:
            Artifact score (higher = more artifacts)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply discrete cosine transform (DCT) to detect JPEG-like artifacts
        # Divide image into 8x8 blocks like JPEG compression
        h, w = gray.shape
        block_size = 8
        artifact_scores = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray[i:i+block_size, j:j+block_size].astype(np.float32)
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Analyze DCT coefficients
                # High frequency artifacts often appear in specific patterns
                high_freq_energy = np.sum(np.abs(dct_block[4:, 4:]))
                total_energy = np.sum(np.abs(dct_block))
                
                if total_energy > 0:
                    ratio = high_freq_energy / total_energy
                    artifact_scores.append(ratio)
        
        if not artifact_scores:
            return 0.0
        
        # Calculate statistics
        mean_artifact = np.mean(artifact_scores)
        var_artifact = np.var(artifact_scores)
        
        # High variance in artifact distribution might indicate inconsistent compression
        combined_score = mean_artifact + var_artifact * 0.1
        
        return min(combined_score, 1.0)
    
    def _combine_frequency_features(self, face_features: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Combine frequency features across multiple faces.
        
        Args:
            face_features: List of frequency feature dictionaries
            
        Returns:
            Combined frequency features
        """
        if not face_features:
            return {}
        
        combined = {}
        
        # Average features across faces
        for key in face_features[0].keys():
            values = [features[key] for features in face_features]
            combined[key] = np.mean(values)
            combined[f"{key}_variance"] = np.var(values)
        
        return combined
    
    def _calculate_confidence(self, frequency_features: Dict[str, float], artifact_score: float) -> float:
        """
        Calculate overall confidence score from frequency analysis.
        
        Args:
            frequency_features: Frequency domain features
            artifact_score: Compression artifact score
            
        Returns:
            Confidence score for deepfake detection
        """
        if not frequency_features:
            return artifact_score
        
        # Deepfakes often have:
        # 1. Unusual high-frequency content
        # 2. Inconsistent compression artifacts
        # 3. Abnormal frequency distribution
        
        confidence_factors = []
        
        # High frequency anomalies
        if "very_high_freq_energy" in frequency_features:
            high_freq_anomaly = frequency_features["very_high_freq_energy"]
            confidence_factors.append(min(high_freq_anomaly / 10.0, 1.0))
        
        # Frequency distribution anomalies
        if "freq_skewness" in frequency_features:
            skewness_anomaly = abs(frequency_features["freq_skewness"])
            confidence_factors.append(min(skewness_anomaly / 5.0, 1.0))
        
        if "freq_kurtosis" in frequency_features:
            kurtosis_anomaly = abs(frequency_features["freq_kurtosis"])
            confidence_factors.append(min(kurtosis_anomaly / 10.0, 1.0))
        
        # Compression artifacts
        confidence_factors.append(artifact_score)
        
        # Spectral centroid anomaly
        if "spectral_centroid" in frequency_features:
            centroid = frequency_features["spectral_centroid"]
            # Unusual spectral centroid might indicate artificial generation
            centroid_anomaly = abs(centroid - 0.3)  # 0.3 is typical for natural images
            confidence_factors.append(min(centroid_anomaly * 2, 1.0))
        
        if not confidence_factors:
            return 0.0
        
        # Weight different factors
        weighted_confidence = (
            np.mean(confidence_factors) * 0.7 +  # Average of all factors
            artifact_score * 0.3  # Extra weight on compression artifacts
        )
        
        return min(weighted_confidence, 1.0)