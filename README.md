# Deepfake Detection System

A comprehensive toolkit for detecting deepfake videos and images using multiple advanced techniques.

## Overview

This system implements various state-of-the-art techniques for deepfake detection:

1. **CNN-based Detection** - Deep learning model for analyzing visual patterns
2. **Temporal Analysis** - Detecting inconsistencies across video frames
3. **Facial Landmark Analysis** - Analyzing facial geometry and symmetry
4. **Frequency Domain Analysis** - Detecting artifacts in frequency spectrum

## Features

- **Multi-technique Ensemble**: Combines multiple detection methods for improved accuracy
- **Flexible Configuration**: Customizable thresholds and weights for different techniques
- **Video and Image Support**: Works with both static images and video files
- **CLI Interface**: Easy-to-use command-line interface
- **Visualization**: Generate annotated results showing detection confidence
- **Extensible Architecture**: Easy to add new detection techniques

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- dlib >= 19.22.0 (optional, for advanced landmark detection)
- face-recognition >= 1.3.0

## Quick Start

### Command Line Usage

```bash
# Detect deepfakes in an image
python main.py image.jpg --output results.json --visualize annotated.jpg

# Detect deepfakes in a video
python main.py video.mp4 --output results.json --verbose

# Use specific detection methods
python main.py image.jpg --methods cnn frequency --threshold 0.7

# Use custom configuration
python main.py video.mp4 --config custom_config.json
```

### Python API Usage

```python
from deepfake_detector import DeepfakeDetectionPipeline, DetectionConfig

# Initialize with default configuration
pipeline = DeepfakeDetectionPipeline()

# Detect deepfakes in an image
result = pipeline.detect_image("path/to/image.jpg")

# Detect deepfakes in a video
result = pipeline.detect_video("path/to/video.mp4")

# Use custom configuration
config = DetectionConfig()
config.confidence_threshold = 0.7
config.ensemble_weights = {"cnn": 0.5, "temporal": 0.2, "landmark": 0.15, "frequency": 0.15}

pipeline = DeepfakeDetectionPipeline(config)
result = pipeline.detect("input_file.mp4")

print(f"Confidence: {result['ensemble_confidence']:.3f}")
print(f"Is Deepfake: {result['ensemble_prediction']}")
```

## Detection Techniques

### 1. CNN-based Detection

Uses a convolutional neural network to analyze visual patterns and artifacts commonly found in deepfakes.

**Features:**
- Face detection and extraction
- Deep learning-based classification
- Confidence scoring

### 2. Temporal Analysis

Analyzes inconsistencies across video frames that may indicate synthetic content.

**Features:**
- Optical flow analysis
- Face tracking consistency
- Temporal artifact detection

### 3. Landmark Detection

Examines facial landmarks for geometric inconsistencies and asymmetries.

**Features:**
- Facial landmark extraction
- Symmetry analysis
- Geometric consistency checking

### 4. Frequency Domain Analysis

Analyzes the frequency spectrum for compression artifacts and generation signatures.

**Features:**
- FFT-based frequency analysis
- Compression artifact detection
- Spectral anomaly identification

## Configuration

### Default Configuration

```python
config = DetectionConfig(
    # CNN parameters
    cnn_threshold=0.5,
    cnn_image_size=224,
    
    # Temporal parameters
    temporal_threshold=0.3,
    temporal_frame_count=30,
    
    # Landmark parameters
    landmark_threshold=0.4,
    landmark_max_distance=50.0,
    
    # Frequency parameters
    frequency_threshold=0.6,
    frequency_bands=[0, 10, 50, 100],
    
    # Ensemble parameters
    confidence_threshold=0.5,
    ensemble_weights={
        "cnn": 0.4,
        "temporal": 0.2,
        "landmark": 0.2,
        "frequency": 0.2
    }
)
```

### Custom Configuration File

Create a JSON configuration file:

```json
{
    "confidence_threshold": 0.7,
    "ensemble_weights": {
        "cnn": 0.5,
        "temporal": 0.1,
        "landmark": 0.2,
        "frequency": 0.2
    },
    "cnn_threshold": 0.6,
    "temporal_frame_count": 50
}
```

## Examples

See `examples.py` for comprehensive usage examples:

```bash
python examples.py
```

## Output Format

The detection system returns detailed results:

```json
{
    "ensemble_confidence": 0.742,
    "ensemble_prediction": true,
    "individual_results": {
        "cnn": {
            "confidence": 0.823,
            "is_deepfake": true,
            "metadata": {
                "faces_detected": 1,
                "method": "face_analysis"
            }
        },
        "temporal": {
            "confidence": 0.654,
            "is_deepfake": true,
            "metadata": {
                "frames_analyzed": 30,
                "mean_inconsistency": 0.654
            }
        }
    }
}
```

## Supported Formats

**Images:** .jpg, .jpeg, .png, .bmp, .tiff, .webp
**Videos:** .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm

## Performance Considerations

- **GPU Acceleration**: CNN detector will use GPU if available (CUDA)
- **Memory Usage**: Large videos are processed in chunks to manage memory
- **Processing Time**: Video analysis takes longer due to temporal analysis
- **Model Loading**: First run may be slower due to model initialization

## Limitations

- CNN detector uses random weights by default (for demonstration purposes)
- dlib landmark model requires separate download for advanced features
- Video processing time scales with video length and resolution
- Requires sufficient computational resources for real-time processing

## Contributing

This system is designed to be extensible. To add new detection techniques:

1. Inherit from `BaseDetector` class
2. Implement `load_model()` and `predict()` methods
3. Add to the detection pipeline
4. Update configuration and ensemble weights

## License

This project is for educational and research purposes.
