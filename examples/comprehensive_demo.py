#!/usr/bin/env python3
"""
Comprehensive demonstration of the AI Classification System
This script demonstrates all major features including:
- Object detection and classification
- Feature extraction and embedding matching
- Clustering and new class discovery
- Human-in-the-loop learning
- Database operations
"""

import sys
import os
import cv2
import numpy as np
import requests
from urllib.request import urlretrieve
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.classification_engine import ClassificationEngine

def download_sample_images():
    """Download some sample images for demonstration"""
    sample_urls = [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/300px-Cat03.jpg", "cat.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/300px-Collage_of_Nine_Dogs.jpg", "dogs.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/2014_Bentley_Continental_GT_Speed_convertible_front_left.jpg/300px-2014_Bentley_Continental_GT_Speed_convertible_front_left.jpg", "car.jpg")
    ]
    
    os.makedirs('examples/sample_images', exist_ok=True)
    
    downloaded_images = []
    for url, filename in sample_urls:
        filepath = f'examples/sample_images/{filename}'
        if not os.path.exists(filepath):
            try:
                print(f"Downloading {filename}...")
                urlretrieve(url, filepath)
                downloaded_images.append(filepath)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            downloaded_images.append(filepath)
    
    return downloaded_images

def create_synthetic_objects():
    """Create synthetic images with different objects"""
    os.makedirs('examples/synthetic_objects', exist_ok=True)
    
    synthetic_images = []
    
    # Create different types of synthetic objects
    objects = [
        ("red_circle", lambda img: cv2.circle(img, (150, 150), 80, (0, 0, 255), -1)),
        ("blue_square", lambda img: cv2.rectangle(img, (70, 70), (230, 230), (255, 0, 0), -1)),
        ("green_triangle", lambda img: cv2.fillPoly(img, [np.array([[150, 50], [50, 250], [250, 250]])], (0, 255, 0))),
        ("yellow_ellipse", lambda img: cv2.ellipse(img, (150, 150), (100, 60), 0, 0, 360, (0, 255, 255), -1)),
    ]
    
    for name, draw_func in objects:
        # Create multiple variations of each object
        for i in range(3):
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            
            # Add some noise/variation
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            
            # Draw the object
            draw_func(img)
            
            # Add some random background elements
            cv2.circle(img, (np.random.randint(50, 250), np.random.randint(50, 250)), 
                      np.random.randint(10, 30), (128, 128, 128), -1)
            
            filepath = f'examples/synthetic_objects/{name}_{i+1}.jpg'
            cv2.imwrite(filepath, img)
            synthetic_images.append(filepath)
    
    return synthetic_images

def demonstrate_object_detection(engine, image_paths):
    """Demonstrate object detection capabilities"""
    print("\n" + "="*60)
    print("OBJECT DETECTION DEMONSTRATION")
    print("="*60)
    
    detection_results = []
    
    for image_path in image_paths:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Could not load image")
            continue
        
        # Process image
        results = engine.process_image(image, image_path)
        
        print(f"  Detections: {len(results['detections'])}")
        print(f"  Classifications: {len(results['classifications'])}")
        print(f"  Unknown objects: {len(results['unknown_objects'])}")
        
        # Show detection details
        for i, detection in enumerate(results['detections']):
            print(f"    Detection {i+1}: {detection['class_name']} "
                  f"(conf: {detection['confidence']:.2f})")
        
        detection_results.append(results)
        
        # Save annotated image
        annotated = engine._draw_results(image, results)
        output_path = f"examples/output/annotated_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated)
        print(f"  Annotated image saved: {output_path}")
    
    return detection_results

def demonstrate_manual_class_creation(engine, image_paths):
    """Demonstrate manual class creation"""
    print("\n" + "="*60)
    print("MANUAL CLASS CREATION DEMONSTRATION")
    print("="*60)
    
    # Group images by type for class creation
    class_groups = {
        "geometric_shapes": [p for p in image_paths if 'synthetic_objects' in p],
        "animals": [p for p in image_paths if any(animal in p.lower() for animal in ['cat', 'dog'])],
        "vehicles": [p for p in image_paths if 'car' in p.lower()]
    }
    
    created_classes = []
    
    for class_name, class_images in class_groups.items():
        if not class_images:
            continue
            
        print(f"\nCreating class: {class_name}")
        print(f"Sample images: {len(class_images)}")
        
        try:
            class_id = engine.add_manual_class(
                class_name, 
                class_images[:3],  # Use first 3 images
                f"Manually created class for {class_name}"
            )
            print(f"  Successfully created class with ID: {class_id}")
            created_classes.append((class_id, class_name))
            
        except Exception as e:
            print(f"  Error creating class: {e}")
    
    return created_classes

def demonstrate_classification(engine, test_images):
    """Demonstrate classification with existing classes"""
    print("\n" + "="*60)
    print("CLASSIFICATION DEMONSTRATION")
    print("="*60)
    
    classification_results = []
    
    for image_path in test_images:
        print(f"\nClassifying: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        results = engine.process_image(image, image_path)
        
        # Show classification results
        if results['classifications']:
            print("  Classifications:")
            for classification in results['classifications']:
                print(f"    {classification['class_name']} "
                      f"(confidence: {classification['confidence']:.3f})")
        
        if results['unknown_objects']:
            print("  Unknown objects:")
            for unknown in results['unknown_objects']:
                print(f"    Unknown object "
                      f"(best match: {unknown.get('confidence', 0):.3f})")
        
        classification_results.append(results)
    
    return classification_results

def demonstrate_clustering(engine):
    """Demonstrate clustering and new class discovery"""
    print("\n" + "="*60)
    print("CLUSTERING AND CLASS DISCOVERY DEMONSTRATION")
    print("="*60)
    
    # Check unknown objects count
    stats = engine.get_statistics()
    unknown_count = stats['num_unknown_objects']
    
    print(f"Unknown objects in buffer: {unknown_count}")
    
    if unknown_count < 3:
        print("Not enough unknown objects for clustering demonstration")
        print("Adding some synthetic unknown objects...")
        
        # Create some additional synthetic objects
        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            embedding = engine.feature_extractor.extract_features(img)
            engine.db_manager.add_unknown_object(embedding, f"synthetic_{i}.jpg")
        
        unknown_count = engine.db_manager.get_database_stats()['num_unknown_objects']
        print(f"Added synthetic objects. New count: {unknown_count}")
    
    # Perform clustering
    print("\nPerforming K-Means clustering...")
    kmeans_results = engine.discover_new_classes(method='kmeans')
    
    print(f"K-Means results: {kmeans_results['status']}")
    if kmeans_results['status'] == 'success':
        print(f"  Clusters found: {len(kmeans_results['cluster_analysis'])}")
        print(f"  Objects clustered: {kmeans_results['clustered_objects']}")
        
        # Show cluster details
        for cluster_id, info in kmeans_results['cluster_analysis'].items():
            suggested_name = kmeans_results['suggested_names'].get(str(cluster_id))
            print(f"  Cluster {cluster_id}: {info['size']} objects - '{suggested_name}'")
    
    # Try DBSCAN as well
    print("\nPerforming DBSCAN clustering...")
    dbscan_results = engine.discover_new_classes(method='dbscan')
    
    print(f"DBSCAN results: {dbscan_results['status']}")
    if dbscan_results['status'] == 'success':
        print(f"  Clusters found: {len(dbscan_results['cluster_analysis'])}")
        print(f"  Objects clustered: {dbscan_results['clustered_objects']}")
    
    return kmeans_results, dbscan_results

def demonstrate_system_statistics(engine):
    """Show comprehensive system statistics"""
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    
    stats = engine.get_statistics()
    
    print("Processing Statistics:")
    print(f"  Total objects processed: {stats['total_processed']}")
    print(f"  Successfully classified: {stats['classified_objects']}")
    print(f"  Unknown objects: {stats['unknown_objects']}")
    print(f"  New classes discovered: {stats['new_classes_discovered']}")
    
    print("\nDatabase Statistics:")
    print(f"  Total classes: {stats['num_classes']}")
    print(f"  Total embeddings: {stats['num_embeddings']}")
    print(f"  Unknown objects in buffer: {stats['num_unknown_objects']}")
    
    print("\nConfiguration:")
    print(f"  Similarity threshold: {stats['similarity_threshold']}")
    print(f"  Unknown buffer size: {stats['unknown_buffer_size']}")
    
    # Show class details
    classes = engine.db_manager.get_classes()
    if classes:
        print("\nClass Details:")
        for class_info in classes:
            print(f"  {class_info['name']}: {class_info['sample_count']} samples")

def save_demonstration_report(results, output_file='examples/output/demo_report.json'):
    """Save demonstration results to a JSON report"""
    print(f"\nSaving demonstration report to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Report saved successfully!")

def main():
    """Run comprehensive demonstration"""
    print("AI CLASSIFICATION SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("="*70)
    
    # Create output directory
    os.makedirs('examples/output', exist_ok=True)
    
    # Initialize the system
    print("Initializing AI Classification System...")
    engine = ClassificationEngine()
    
    # Prepare demonstration data
    print("\nPreparing demonstration data...")
    
    # Download sample images
    real_images = download_sample_images()
    print(f"Downloaded {len(real_images)} real images")
    
    # Create synthetic objects
    synthetic_images = create_synthetic_objects()
    print(f"Created {len(synthetic_images)} synthetic images")
    
    all_images = real_images + synthetic_images
    
    # Store results for report
    demo_results = {
        'initialization': 'success',
        'sample_images': len(all_images),
        'demonstrations': {}
    }
    
    # 1. Object Detection Demonstration
    detection_results = demonstrate_object_detection(engine, all_images[:6])
    demo_results['demonstrations']['object_detection'] = {
        'images_processed': len(detection_results),
        'total_detections': sum(len(r['detections']) for r in detection_results)
    }
    
    # 2. Manual Class Creation
    created_classes = demonstrate_manual_class_creation(engine, all_images)
    demo_results['demonstrations']['manual_classes'] = {
        'classes_created': len(created_classes),
        'class_names': [name for _, name in created_classes]
    }
    
    # 3. Classification with existing classes
    classification_results = demonstrate_classification(engine, all_images[6:])
    demo_results['demonstrations']['classification'] = {
        'images_classified': len(classification_results),
        'successful_classifications': sum(len(r['classifications']) for r in classification_results)
    }
    
    # 4. Clustering and Class Discovery
    kmeans_results, dbscan_results = demonstrate_clustering(engine)
    demo_results['demonstrations']['clustering'] = {
        'kmeans_status': kmeans_results['status'],
        'dbscan_status': dbscan_results['status']
    }
    
    # 5. System Statistics
    demonstrate_system_statistics(engine)
    final_stats = engine.get_statistics()
    demo_results['final_statistics'] = final_stats
    
    # Save comprehensive report
    save_demonstration_report(demo_results)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nKey Achievements:")
    print(f"✓ Processed {demo_results['sample_images']} images")
    print(f"✓ Created {len(created_classes)} classes manually")
    print(f"✓ Performed object detection and classification")
    print(f"✓ Demonstrated clustering algorithms")
    print(f"✓ Generated comprehensive system report")
    
    print(f"\nFinal System State:")
    print(f"  Classes: {final_stats['num_classes']}")
    print(f"  Embeddings: {final_stats['num_embeddings']}")
    print(f"  Objects processed: {final_stats['total_processed']}")
    
    print(f"\nOutput files generated in: examples/output/")
    print(f"Web interface available at: http://localhost:12000")
    
    print("\nNext steps:")
    print("1. Open the web interface to explore the system interactively")
    print("2. Upload your own images to test classification")
    print("3. Use the camera interface for real-time detection")
    print("4. Review and create classes from discovered clusters")

if __name__ == '__main__':
    main()