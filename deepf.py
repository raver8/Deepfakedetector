import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from scipy import linalg
from scipy.stats import entropy, skew, kurtosis
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedDeepfakeDetector:
    """
    Advanced linear algebra-based deepfake detection system with multiple
    mathematical approaches for robust analysis.
    """
    
    def __init__(self, block_size: int = 16, overlap: float = 0.5):
        """
        Initialize the detector with configurable parameters.
        
        Args:
            block_size: Size of image blocks for local analysis
            overlap: Overlap ratio between blocks (0-1)
        """
        self.block_size = block_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.classifier = None
        self.is_trained = False
        
    def extract_overlapping_blocks(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract overlapping blocks from image for more comprehensive analysis.
        """
        blocks = []
        h, w = image.shape
        step = int(self.block_size * (1 - self.overlap))
        
        for i in range(0, h - self.block_size + 1, step):
            for j in range(0, w - self.block_size + 1, step):
                block = image[i:i+self.block_size, j:j+self.block_size]
                if block.shape == (self.block_size, self.block_size):
                    blocks.append(block)
        return blocks
    
    def enhanced_svd_features(self, block: np.ndarray) -> np.ndarray:
        """
        Extract enhanced SVD-based features with statistical analysis.
        """
        # Compute SVD
        U, s, Vt = np.linalg.svd(block, full_matrices=False)
        
        # Normalize singular values
        s_norm = s / (np.sum(s) + 1e-8)
        
        # Statistical features of singular values
        features = [
            np.mean(s_norm),
            np.std(s_norm),
            entropy(s_norm + 1e-8),  # Shannon entropy
            skew(s_norm),  # Skewness
            kurtosis(s_norm),  # Kurtosis
            np.sum(s_norm[:5]) / np.sum(s_norm),  # Energy concentration in top 5 components
            np.count_nonzero(s_norm > 0.01) / len(s_norm),  # Effective rank ratio
        ]
        
        # Spectral features
        s_diff = np.diff(s_norm)
        features.extend([
            np.mean(s_diff),
            np.std(s_diff),
            np.max(s_diff) - np.min(s_diff)  # Range of differences
        ])
        
        return np.array(features)
    
    def eigenvalue_analysis(self, block: np.ndarray) -> np.ndarray:
        """
        Analyze eigenvalues of covariance matrix for texture analysis.
        """
        # Flatten block and compute covariance matrix
        flat_block = block.flatten().reshape(-1, 1)
        if len(flat_block) > 1:
            cov_matrix = np.cov(flat_block.T)
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])
            
            # Compute eigenvalues
            eigenvals = np.real(np.linalg.eigvals(cov_matrix))
            eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
            
            # Normalize
            eigenvals_norm = eigenvals / (np.sum(eigenvals) + 1e-8)
            
            # Statistical features
            features = [
                np.mean(eigenvals_norm),
                np.std(eigenvals_norm),
                entropy(eigenvals_norm + 1e-8),
                skew(eigenvals_norm),
                kurtosis(eigenvals_norm)
            ]
        else:
            features = [0] * 5
            
        return np.array(features)
    
    def gradient_coherence_analysis(self, block: np.ndarray) -> np.ndarray:
        """
        Analyze gradient coherence for detecting GAN artifacts.
        """
        # Compute gradients
        grad_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Coherence matrix analysis
        J11 = np.sum(grad_x**2)
        J22 = np.sum(grad_y**2)
        J12 = np.sum(grad_x * grad_y)
        
        coherence_matrix = np.array([[J11, J12], [J12, J22]])
        eigenvals = np.real(np.linalg.eigvals(coherence_matrix))
        
        # Coherence and anisotropy measures
        if np.sum(eigenvals) > 0:
            coherence = (eigenvals[0] - eigenvals[1])**2 / (eigenvals[0] + eigenvals[1] + 1e-8)**2
            anisotropy = (eigenvals[0] - eigenvals[1]) / (eigenvals[0] + eigenvals[1] + 1e-8)
        else:
            coherence = anisotropy = 0
        
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            coherence,
            anisotropy,
            entropy(magnitude.flatten() + 1e-8)
        ]
        
        return np.array(features)
    
    def frequency_domain_analysis(self, block: np.ndarray) -> np.ndarray:
        """
        Analyze frequency domain characteristics using DCT.
        """
        # Discrete Cosine Transform
        dct_block = cv2.dct(block.astype(np.float32))
        
        # Divide into frequency bands
        low_freq = dct_block[:4, :4]  # Low frequency components
        mid_freq = dct_block[4:8, 4:8]  # Mid frequency components
        high_freq = dct_block[8:, 8:]  # High frequency components
        
        features = [
            np.mean(np.abs(low_freq)),
            np.std(np.abs(low_freq)),
            np.mean(np.abs(mid_freq)),
            np.std(np.abs(mid_freq)),
            np.mean(np.abs(high_freq)),
            np.std(np.abs(high_freq)),
            np.sum(np.abs(low_freq)) / (np.sum(np.abs(dct_block)) + 1e-8),  # Energy ratio
        ]
        
        return np.array(features)
    
    def extract_comprehensive_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature set combining all analysis methods.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize image
        image = image.astype(np.float64) / 255.0
        
        # Extract blocks
        blocks = self.extract_overlapping_blocks(image)
        
        all_features = []
        
        for block in blocks:
            # Extract features from each method
            svd_feats = self.enhanced_svd_features(block)
            eigen_feats = self.eigenvalue_analysis(block)
            grad_feats = self.gradient_coherence_analysis(block)
            freq_feats = self.frequency_domain_analysis(block)
            
            # Combine all features
            combined_feats = np.concatenate([svd_feats, eigen_feats, grad_feats, freq_feats])
            all_features.append(combined_feats)
        
        # Statistical aggregation of block features
        all_features = np.array(all_features)
        
        # Compute statistics across all blocks
        feature_stats = [
            np.mean(all_features, axis=0),
            np.std(all_features, axis=0),
            np.median(all_features, axis=0),
            np.percentile(all_features, 25, axis=0),
            np.percentile(all_features, 75, axis=0),
        ]
        
        return np.concatenate(feature_stats)
    
    def train(self, real_images: List[np.ndarray], fake_images: List[np.ndarray]):
        """
        Train the classifier using real and fake image datasets.
        """
        print("Extracting features from training data...")
        
        # Extract features
        real_features = [self.extract_comprehensive_features(img) for img in real_images]
        fake_features = [self.extract_comprehensive_features(img) for img in fake_images]
        
        # Prepare training data
        X = np.vstack([real_features, fake_features])
        y = np.hstack([np.zeros(len(real_features)), np.ones(len(fake_features))])
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"Feature dimensionality: {X.shape[1]} -> {X_pca.shape[1]} (after PCA)")
        
        # Train ensemble classifier
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_pca, y)
        self.is_trained = True
        
        # Report training accuracy
        train_pred = self.classifier.predict(X_pca)
        train_acc = accuracy_score(y, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict if an image is a deepfake.
        
        Returns:
            (prediction, confidence) where prediction is 0 for real, 1 for fake
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_comprehensive_features(image)
        
        # Transform features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_pca = self.pca.transform(features_scaled)
        
        # Make prediction
        prediction = self.classifier.predict(features_pca)[0]
        confidence = np.max(self.classifier.predict_proba(features_pca)[0])
        
        return int(prediction), confidence
    
    def analyze_video(self, video_path: str, sample_rate: int = 10) -> Dict:
        """
        Analyze video for deepfake detection with temporal analysis.
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        predictions = []
        confidences = []
        temporal_features = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate == 0:
                pred, conf = self.predict(frame)
                predictions.append(pred)
                confidences.append(conf)
                
                # Extract temporal features (frame-to-frame consistency)
                if len(temporal_features) > 0:
                    current_features = self.extract_comprehensive_features(frame)
                    prev_features = temporal_features[-1]
                    consistency = np.corrcoef(current_features, prev_features)[0, 1]
                    temporal_features.append(current_features)
                else:
                    consistency = 1.0
                    temporal_features.append(self.extract_comprehensive_features(frame))
            
            frame_idx += 1
        
        cap.release()
        
        # Analyze temporal consistency
        avg_prediction = np.mean(predictions)
        avg_confidence = np.mean(confidences)
        prediction_std = np.std(predictions)
        
        # Temporal coherence analysis
        if len(temporal_features) > 1:
            coherence_scores = []
            for i in range(1, len(temporal_features)):
                coherence = np.corrcoef(temporal_features[i-1], temporal_features[i])[0, 1]
                if not np.isnan(coherence):
                    coherence_scores.append(coherence)
            
            temporal_coherence = np.mean(coherence_scores) if coherence_scores else 0
        else:
            temporal_coherence = 1.0
        
        return {
            'overall_prediction': 1 if avg_prediction > 0.5 else 0,
            'confidence': avg_confidence,
            'temporal_consistency': 1 - prediction_std,  # Lower std = higher consistency
            'temporal_coherence': temporal_coherence,
            'frame_predictions': predictions,
            'analysis_summary': {
                'total_frames': frame_count,
                'analyzed_frames': len(predictions),
                'fake_frame_ratio': np.sum(predictions) / len(predictions) if predictions else 0
            }
        }

# Example usage and testing functions
def create_sample_dataset():
    """Create sample dataset for demonstration."""
    # This would normally load your real dataset
    real_images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(50)]
    fake_images = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(50)]
    
    # Add some distinguishing characteristics to fake images for demo
    for img in fake_images:
        # Add subtle noise pattern that might be characteristic of GANs
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img[:] = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return real_images, fake_images

def demo_detection():
    """Demonstrate the enhanced deepfake detection system."""
    print("Enhanced Deepfake Detection System Demo")
    print("="*50)
    
    # Create detector
    detector = EnhancedDeepfakeDetector(block_size=16, overlap=0.3)
    
    # Create sample dataset
    print("Creating sample dataset...")
    real_images, fake_images = create_sample_dataset()
    
    # Train detector
    print("Training detector...")
    detector.train(real_images[:40], fake_images[:40])  # Use first 40 for training
    
    # Test on remaining samples
    print("\nTesting on validation set...")
    test_real = real_images[40:]
    test_fake = fake_images[40:]
    
    correct_predictions = 0
    total_predictions = 0
    
    for img in test_real:
        pred, conf = detector.predict(img)
        if pred == 0:  # Correctly identified as real
            correct_predictions += 1
        total_predictions += 1
        print(f"Real image: Predicted {'Real' if pred == 0 else 'Fake'} (confidence: {conf:.3f})")
    
    for img in test_fake:
        pred, conf = detector.predict(img)
        if pred == 1:  # Correctly identified as fake
            correct_predictions += 1
        total_predictions += 1
        print(f"Fake image: Predicted {'Real' if pred == 0 else 'Fake'} (confidence: {conf:.3f})")
    
    accuracy = correct_predictions / total_predictions
    print(f"\nValidation Accuracy: {accuracy:.3f}")
    
    return detector

if __name__ == "__main__":
    # Run demonstration
    detector = demo_detection()
    print("\nDetector trained and ready for use!")
    print("Use detector.predict(image) for single image analysis")
    print("Use detector.analyze_video(video_path) for video analysis")
