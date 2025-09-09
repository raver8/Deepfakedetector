"""
Utility functions for deepfake detection.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Union


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array
        
    Raises:
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    return image


def load_video_frames(video_path: str, max_frames: int = 100) -> List[np.ndarray]:
    """
    Load frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frame arrays
        
    Raises:
        ValueError: If video cannot be loaded
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // max_frames)
    
    frame_idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        frames.append(frame)
        frame_idx += frame_step
    
    cap.release()
    return frames


def save_detection_result(result: dict, output_path: str) -> None:
    """
    Save detection result to JSON file.
    
    Args:
        result: Detection result dictionary
        output_path: Path to output JSON file
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    converted_result = convert_numpy(result)
    
    with open(output_path, 'w') as f:
        json.dump(converted_result, f, indent=2)


def visualize_detection_result(image: np.ndarray, 
                             result: dict, 
                             output_path: str = None) -> np.ndarray:
    """
    Visualize detection result on image.
    
    Args:
        image: Input image array
        result: Detection result dictionary
        output_path: Optional path to save visualization
        
    Returns:
        Annotated image array
    """
    annotated_image = image.copy()
    
    # Add overall prediction text
    ensemble_confidence = result.get("ensemble_confidence", 0.0)
    ensemble_prediction = result.get("ensemble_prediction", False)
    
    # Choose color based on prediction
    color = (0, 0, 255) if ensemble_prediction else (0, 255, 0)  # Red for fake, green for real
    label = f"{'DEEPFAKE' if ensemble_prediction else 'REAL'} ({ensemble_confidence:.2f})"
    
    # Add text to image
    cv2.putText(annotated_image, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add individual detector results
    y_offset = 60
    individual_results = result.get("individual_results", {})
    
    for method, method_result in individual_results.items():
        if "error" in method_result:
            continue
            
        confidence = method_result.get("confidence", 0.0)
        is_fake = method_result.get("is_deepfake", False)
        
        method_color = (0, 0, 255) if is_fake else (0, 255, 0)
        method_label = f"{method}: {confidence:.2f}"
        
        cv2.putText(annotated_image, method_label, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, method_color, 1)
        y_offset += 25
    
    # Draw face regions if available
    for method, method_result in individual_results.items():
        metadata = method_result.get("metadata", {})
        face_regions = metadata.get("face_regions", [])
        
        for (x, y, w, h) in face_regions:
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, annotated_image)
    
    return annotated_image


def create_model_directory(model_dir: str = "models") -> str:
    """
    Create directory for storing models.
    
    Args:
        model_dir: Model directory path
        
    Returns:
        Absolute path to model directory
    """
    abs_model_dir = os.path.abspath(model_dir)
    os.makedirs(abs_model_dir, exist_ok=True)
    return abs_model_dir


def get_supported_formats() -> dict:
    """
    Get supported file formats.
    
    Returns:
        Dictionary of supported formats
    """
    return {
        "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        "videos": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    }


def is_supported_format(file_path: str) -> Tuple[bool, str]:
    """
    Check if file format is supported.
    
    Args:
        file_path: Path to file
        
    Returns:
        Tuple of (is_supported, file_type)
    """
    supported = get_supported_formats()
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext in supported["images"]:
        return True, "image"
    elif file_ext in supported["videos"]:
        return True, "video"
    else:
        return False, "unknown"