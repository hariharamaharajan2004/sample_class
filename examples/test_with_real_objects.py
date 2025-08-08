#!/usr/bin/env python3
"""
Test the system with real objects that YOLO can detect
"""

import sys
import os
import cv2
import numpy as np
from urllib.request import urlretrieve

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.classification_engine import ClassificationEngine

def download_test_images():
    """Download test images with objects YOLO can detect"""
    test_images = [
        ("https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=400", "person.jpg"),
        ("https://images.unsplash.com/photo-1552053831-71594a27632d?w=400", "dog.jpg"),
        ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400", "cat.jpg"),
        ("https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400", "car.jpg"),
    ]
    
    os.makedirs('examples/test_images', exist_ok=True)
    downloaded = []
    
    for url, filename in test_images:
        filepath = f'examples/test_images/{filename}'
        if not os.path.exists(filepath):
            try:
                print(f"Downloading {filename}...")
                urlretrieve(url, filepath)
                downloaded.append(filepath)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
        else:
            downloaded.append(filepath)
            print(f"✓ Using existing {filename}")
    
    return downloaded

def test_yolo_detection():
    """Test YOLO detection capabilities"""
    print("="*60)
    print("TESTING YOLO OBJECT DETECTION")
    print("="*60)
    
    # Initialize engine
    engine = ClassificationEngine()
    
    # Download test images
    test_images = download_test_images()
    
    if not test_images:
        print("No test images available. Creating a simple test image...")
        # Create a more realistic test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
        
        # Add some text (which YOLO might detect as objects in some cases)
        cv2.putText(img, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Add some geometric shapes that might be detected
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green square
        cv2.circle(img, (400, 150), 50, (255, 0, 0), -1)  # Blue circle
        
        test_path = 'examples/test_images/synthetic_test.jpg'
        os.makedirs('examples/test_images', exist_ok=True)
        cv2.imwrite(test_path, img)
        test_images = [test_path]
    
    # Test each image
    total_detections = 0
    for image_path in test_images:
        print(f"\nTesting: {os.path.basename(image_path)}")
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ Could not load image")
            continue
        
        print(f"  Image size: {image.shape}")
        
        # Process with the classification engine
        results = engine.process_image(image, image_path)
        
        detections = len(results['detections'])
        classifications = len(results['classifications'])
        unknown_objects = len(results['unknown_objects'])
        
        print(f"  Detections: {detections}")
        print(f"  Classifications: {classifications}")
        print(f"  Unknown objects: {unknown_objects}")
        
        total_detections += detections
        
        # Show detection details
        if results['detections']:
            print("  Detected objects:")
            for i, detection in enumerate(results['detections']):
                print(f"    {i+1}. {detection['class_name']} "
                      f"(confidence: {detection['confidence']:.3f}) "
                      f"bbox: {detection['bbox']}")
        
        # Save annotated image
        if detections > 0:
            annotated = engine._draw_results(image, results)
            output_path = f"examples/test_images/annotated_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated)
            print(f"  ✓ Saved annotated image: {output_path}")
    
    print(f"\nTotal detections across all images: {total_detections}")
    
    if total_detections == 0:
        print("\n" + "="*60)
        print("TROUBLESHOOTING: NO DETECTIONS FOUND")
        print("="*60)
        print("Possible reasons:")
        print("1. Images don't contain objects from YOLO's training classes")
        print("2. Objects are too small or low quality")
        print("3. Detection confidence threshold is too high")
        print("4. Images need better lighting or contrast")
        print("\nYOLO can detect these object classes:")
        
        # Show YOLO classes
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        class_names = model.names
        
        print("COCO Dataset Classes (80 total):")
        for i, name in class_names.items():
            print(f"  {i}: {name}")
        
        print(f"\nTry uploading images containing these objects through the web interface!")
        print(f"Web interface: http://localhost:12000")

def main():
    """Run the test"""
    print("AI Classification System - Real Object Detection Test")
    print("This test uses images with objects that YOLO can detect")
    
    test_yolo_detection()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Use the web interface to upload your own images")
    print("2. Try images with people, animals, vehicles, or common objects")
    print("3. Use the camera interface for real-time detection")
    print("4. Once objects are detected, you can create custom classes")

if __name__ == '__main__':
    main()