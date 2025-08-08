#!/usr/bin/env python3
"""
Basic usage example for the AI Classification System
"""

import sys
import os
import cv2
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.classification_engine import ClassificationEngine

def create_sample_image():
    """Create a simple sample image for testing"""
    # Create a simple colored rectangle image
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(image, (200, 100), (350, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(image, (100, 220), 40, (0, 0, 255), -1)  # Red circle
    
    return image

def main():
    """Demonstrate basic system usage"""
    print("AI Classification System - Basic Usage Example")
    print("=" * 50)
    
    # Initialize the classification engine
    print("Initializing classification engine...")
    engine = ClassificationEngine()
    
    # Create a sample image
    print("Creating sample image...")
    sample_image = create_sample_image()
    
    # Save sample image
    os.makedirs('examples/output', exist_ok=True)
    cv2.imwrite('examples/output/sample_image.jpg', sample_image)
    print("Sample image saved to: examples/output/sample_image.jpg")
    
    # Process the image
    print("\nProcessing image...")
    results = engine.process_image(sample_image, 'examples/output/sample_image.jpg')
    
    # Display results
    print("\n=== PROCESSING RESULTS ===")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Detections: {len(results['detections'])}")
    print(f"Classifications: {len(results['classifications'])}")
    print(f"Unknown objects: {len(results['unknown_objects'])}")
    
    # Show detection details
    if results['detections']:
        print("\nDetected Objects:")
        for i, detection in enumerate(results['detections'], 1):
            bbox = detection['bbox']
            print(f"  {i}. {detection['class_name']} "
                  f"(confidence: {detection['confidence']:.2f}) "
                  f"at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    
    # Show classification results
    if results['classifications']:
        print("\nClassified Objects:")
        for i, classification in enumerate(results['classifications'], 1):
            print(f"  {i}. Class: {classification['class_name']} "
                  f"(confidence: {classification['confidence']:.2f})")
    
    # Show unknown objects
    if results['unknown_objects']:
        print("\nUnknown Objects:")
        for i, unknown in enumerate(results['unknown_objects'], 1):
            print(f"  {i}. Unknown object "
                  f"(best match similarity: {unknown.get('confidence', 0):.2f})")
    
    # Create annotated image
    annotated_image = engine._draw_results(sample_image, results)
    cv2.imwrite('examples/output/annotated_image.jpg', annotated_image)
    print("\nAnnotated image saved to: examples/output/annotated_image.jpg")
    
    # Show system statistics
    print("\n=== SYSTEM STATISTICS ===")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Demonstrate manual class creation
    print("\n=== MANUAL CLASS CREATION ===")
    try:
        # Create a simple test class with the sample image
        class_id = engine.add_manual_class(
            "sample_objects", 
            ['examples/output/sample_image.jpg'],
            "Sample objects for testing"
        )
        print(f"Created test class with ID: {class_id}")
        
        # Process the same image again to see classification
        print("\nProcessing image again after creating class...")
        results2 = engine.process_image(sample_image, 'examples/output/sample_image.jpg')
        
        print(f"New classifications: {len(results2['classifications'])}")
        print(f"New unknown objects: {len(results2['unknown_objects'])}")
        
    except Exception as e:
        print(f"Error creating manual class: {e}")
    
    # Demonstrate clustering (if enough unknown objects)
    print("\n=== CLUSTERING DEMONSTRATION ===")
    try:
        clustering_results = engine.discover_new_classes()
        print(f"Clustering status: {clustering_results['status']}")
        
        if clustering_results['status'] == 'success':
            print(f"Clusters found: {len(clustering_results['cluster_analysis'])}")
        elif clustering_results['status'] == 'insufficient_data':
            print(f"Need more unknown objects (current: {clustering_results['count']})")
        
    except Exception as e:
        print(f"Error during clustering: {e}")
    
    print("\n=== EXAMPLE COMPLETED ===")
    print("Check the examples/output/ directory for generated images.")
    print("Run 'python main.py web' to start the web interface.")

if __name__ == '__main__':
    main()