"""
Example usage of the Deepfake Detection System.
"""

import numpy as np
import cv2
from deepfake_detector import DeepfakeDetectionPipeline, DetectionConfig
from deepfake_detector.utils import visualize_detection_result, save_detection_result


def example_image_detection():
    """Example of detecting deepfakes in an image."""
    print("Example: Image Deepfake Detection")
    print("-" * 40)
    
    # Create a sample image (in real usage, you'd load an actual image)
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detection pipeline
    config = DetectionConfig()
    pipeline = DeepfakeDetectionPipeline(config)
    
    # Run detection on image
    result = pipeline.detect_image("sample_image.jpg") if False else pipeline.detect(sample_image, ["cnn", "landmark", "frequency"])
    
    # Display results
    print(f"Ensemble Confidence: {result['ensemble_confidence']:.3f}")
    print(f"Prediction: {'DEEPFAKE' if result['ensemble_prediction'] else 'AUTHENTIC'}")
    
    # Show individual detector results
    for method, method_result in result['individual_results'].items():
        if 'error' not in method_result:
            confidence = method_result['confidence']
            prediction = method_result['is_deepfake']
            print(f"{method.upper()}: {confidence:.3f} ({'FAKE' if prediction else 'REAL'})")
    
    return result


def example_video_detection():
    """Example of detecting deepfakes in a video."""
    print("\nExample: Video Deepfake Detection")
    print("-" * 40)
    
    # Create sample video frames
    frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    
    # Initialize detection pipeline
    config = DetectionConfig()
    pipeline = DeepfakeDetectionPipeline(config)
    
    # Run detection on video frames
    result = pipeline.detect(frames, ["cnn", "temporal", "landmark", "frequency"])
    
    # Display results
    print(f"Ensemble Confidence: {result['ensemble_confidence']:.3f}")
    print(f"Prediction: {'DEEPFAKE' if result['ensemble_prediction'] else 'AUTHENTIC'}")
    
    # Show individual detector results
    for method, method_result in result['individual_results'].items():
        if 'error' not in method_result:
            confidence = method_result['confidence']
            prediction = method_result['is_deepfake']
            print(f"{method.upper()}: {confidence:.3f} ({'FAKE' if prediction else 'REAL'})")
    
    return result


def example_custom_configuration():
    """Example of using custom configuration."""
    print("\nExample: Custom Configuration")
    print("-" * 40)
    
    # Create custom configuration
    config = DetectionConfig()
    config.confidence_threshold = 0.7  # Higher threshold
    config.ensemble_weights = {
        "cnn": 0.5,
        "temporal": 0.1,
        "landmark": 0.2,
        "frequency": 0.2
    }  # Give more weight to CNN
    
    # Initialize pipeline with custom config
    pipeline = DeepfakeDetectionPipeline(config)
    
    # Get detector information
    detector_info = pipeline.get_detector_info()
    print("Detector Configuration:")
    for name, info in detector_info.items():
        print(f"  {name.upper()}: threshold={info['threshold']:.2f}, weight={info['weight']:.2f}")
    
    return pipeline


def example_individual_detectors():
    """Example of using individual detectors."""
    print("\nExample: Individual Detectors")
    print("-" * 40)
    
    from deepfake_detector.detectors import CNNDetector, LandmarkDetector, FrequencyDetector
    
    # Create sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test CNN Detector
    cnn_detector = CNNDetector(threshold=0.5)
    cnn_confidence, cnn_metadata = cnn_detector.predict(sample_image)
    print(f"CNN Detector: {cnn_confidence:.3f} ({cnn_metadata.get('method', 'N/A')})")
    
    # Test Landmark Detector  
    landmark_detector = LandmarkDetector(threshold=0.4)
    landmark_confidence, landmark_metadata = landmark_detector.predict(sample_image)
    print(f"Landmark Detector: {landmark_confidence:.3f}")
    
    # Test Frequency Detector
    frequency_detector = FrequencyDetector(threshold=0.6)
    freq_confidence, freq_metadata = frequency_detector.predict(sample_image)
    print(f"Frequency Detector: {freq_confidence:.3f}")


def main():
    """Run all examples."""
    print("Deepfake Detection System - Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_image_detection()
        example_video_detection() 
        example_custom_configuration()
        example_individual_detectors()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()