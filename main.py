#!/usr/bin/env python3
"""
Main CLI interface for the Deepfake Detection System.
"""

import argparse
import sys
import os
import json
from deepfake_detector import DeepfakeDetectionPipeline, DetectionConfig
from deepfake_detector.utils import (
    load_image, 
    visualize_detection_result, 
    save_detection_result,
    is_supported_format
)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Deepfake Detection System - Detect deepfakes using multiple techniques"
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input image or video file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output path for results (JSON format)"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        help="Path to save visualization image"
    )
    
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        choices=["cnn", "temporal", "landmark", "frequency"],
        help="Detection methods to use (default: auto-select based on input type)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold for classification (default: 0.5)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    # Check if file format is supported
    is_supported, file_type = is_supported_format(args.input_path)
    if not is_supported:
        print(f"Error: Unsupported file format: {args.input_path}")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = DetectionConfig()
    
    # Override threshold if provided
    if args.threshold != 0.5:
        config.confidence_threshold = args.threshold
    
    # Initialize pipeline
    if args.verbose:
        print("Initializing detection pipeline...")
    
    pipeline = DeepfakeDetectionPipeline(config)
    
    # Select detection methods
    if args.methods:
        methods = args.methods
    else:
        # Auto-select based on file type
        if file_type == "image":
            methods = ["cnn", "landmark", "frequency"]
        else:  # video
            methods = ["cnn", "temporal", "landmark", "frequency"]
    
    if args.verbose:
        print(f"Using detection methods: {', '.join(methods)}")
        print(f"Analyzing {file_type}: {args.input_path}")
    
    # Run detection
    try:
        result = pipeline.detect(args.input_path, methods)
        
        # Display results
        display_results(result, args.verbose)
        
        # Save results if requested
        if args.output:
            save_detection_result(result, args.output)
            if args.verbose:
                print(f"Results saved to: {args.output}")
        
        # Create visualization if requested
        if args.visualize and file_type == "image":
            image = load_image(args.input_path)
            visualize_detection_result(image, result, args.visualize)
            if args.verbose:
                print(f"Visualization saved to: {args.visualize}")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def load_config(config_path: str) -> DetectionConfig:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object with loaded values
        config = DetectionConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return DetectionConfig()


def display_results(result: dict, verbose: bool = False):
    """Display detection results."""
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION RESULTS")
    print("="*50)
    
    # Overall result
    ensemble_confidence = result.get("ensemble_confidence", 0.0)
    ensemble_prediction = result.get("ensemble_prediction", False)
    
    status = "DEEPFAKE DETECTED" if ensemble_prediction else "APPEARS AUTHENTIC"
    print(f"\nOverall Result: {status}")
    print(f"Confidence Score: {ensemble_confidence:.3f}")
    
    # Individual detector results
    if verbose:
        print(f"\nIndividual Detector Results:")
        print("-" * 30)
        
        individual_results = result.get("individual_results", {})
        
        for method, method_result in individual_results.items():
            if "error" in method_result:
                print(f"{method.upper()}: ERROR - {method_result['error']}")
                continue
            
            confidence = method_result.get("confidence", 0.0)
            is_fake = method_result.get("is_deepfake", False)
            status = "FAKE" if is_fake else "REAL"
            
            print(f"{method.upper()}: {status} (confidence: {confidence:.3f})")
            
            # Show additional metadata if available
            metadata = method_result.get("metadata", {})
            if "faces_detected" in metadata:
                print(f"  └─ Faces detected: {metadata['faces_detected']}")
            if "frames_analyzed" in metadata:
                print(f"  └─ Frames analyzed: {metadata['frames_analyzed']}")
    
    print("\n" + "="*50)


def create_sample_config():
    """Create a sample configuration file."""
    config = DetectionConfig()
    
    config_dict = {
        "cnn_threshold": config.cnn_threshold,
        "temporal_threshold": config.temporal_threshold,
        "landmark_threshold": config.landmark_threshold,
        "frequency_threshold": config.frequency_threshold,
        "confidence_threshold": config.confidence_threshold,
        "ensemble_weights": config.ensemble_weights,
        "temporal_frame_count": config.temporal_frame_count,
        "cnn_image_size": config.cnn_image_size
    }
    
    with open("sample_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print("Sample configuration created: sample_config.json")


if __name__ == "__main__":
    main()