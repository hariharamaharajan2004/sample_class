#!/usr/bin/env python3
"""
AI-Powered Image Identification and Classification System
Main entry point for the application
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.classification_engine import ClassificationEngine
from src.ui.web_app import run_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI-Powered Image Identification and Classification System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=12000, help='Port to bind to')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # CLI commands
    cli_parser = subparsers.add_parser('cli', help='Command line interface')
    cli_subparsers = cli_parser.add_subparsers(dest='cli_command', help='CLI commands')
    
    # Process image command
    process_parser = cli_subparsers.add_parser('process', help='Process an image')
    process_parser.add_argument('image_path', help='Path to image file')
    process_parser.add_argument('--save', action='store_true', help='Save results to database')
    process_parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Camera command
    camera_parser = cli_subparsers.add_parser('camera', help='Start camera processing')
    camera_parser.add_argument('--source', type=int, default=0, help='Camera source')
    camera_parser.add_argument('--save', action='store_true', help='Save detections')
    
    # Add class command
    add_class_parser = cli_subparsers.add_parser('add-class', help='Add a new class')
    add_class_parser.add_argument('name', help='Class name')
    add_class_parser.add_argument('images', nargs='+', help='Sample image paths')
    add_class_parser.add_argument('--description', help='Class description')
    
    # Discover classes command
    discover_parser = cli_subparsers.add_parser('discover', help='Discover new classes')
    discover_parser.add_argument('--method', choices=['kmeans', 'dbscan'], 
                                default='kmeans', help='Clustering method')
    
    # Stats command
    stats_parser = cli_subparsers.add_parser('stats', help='Show system statistics')
    
    # List classes command
    list_parser = cli_subparsers.add_parser('list-classes', help='List all classes')
    
    args = parser.parse_args()
    
    if args.command == 'web':
        run_web_interface(args)
    elif args.command == 'cli':
        run_cli_command(args)
    else:
        parser.print_help()

def run_web_interface(args):
    """Run the web interface"""
    logger.info(f"Starting web interface on {args.host}:{args.port}")
    run_app(host=args.host, port=args.port, debug=args.debug)

def run_cli_command(args):
    """Run CLI commands"""
    engine = ClassificationEngine()
    
    if args.cli_command == 'process':
        process_image_cli(engine, args)
    elif args.cli_command == 'camera':
        start_camera_cli(engine, args)
    elif args.cli_command == 'add-class':
        add_class_cli(engine, args)
    elif args.cli_command == 'discover':
        discover_classes_cli(engine, args)
    elif args.cli_command == 'stats':
        show_stats_cli(engine)
    elif args.cli_command == 'list-classes':
        list_classes_cli(engine)
    else:
        print("Unknown CLI command")

def process_image_cli(engine, args):
    """Process a single image via CLI"""
    import cv2
    import json
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return
    
    print(f"Processing image: {args.image_path}")
    
    # Load and process image
    image = cv2.imread(args.image_path)
    if image is None:
        print("Error: Could not load image")
        return
    
    results = engine.process_image(image, args.image_path)
    
    # Display results
    print("\n=== PROCESSING RESULTS ===")
    print(f"Detections: {len(results.get('detections', []))}")
    print(f"Classifications: {len(results.get('classifications', []))}")
    print(f"Unknown objects: {len(results.get('unknown_objects', []))}")
    
    # Show classifications
    if results.get('classifications'):
        print("\nClassified Objects:")
        for i, classification in enumerate(results['classifications'], 1):
            print(f"  {i}. {classification['class_name']} "
                  f"(confidence: {classification['confidence']:.2f})")
    
    # Show unknown objects
    if results.get('unknown_objects'):
        print("\nUnknown Objects:")
        for i, unknown in enumerate(results['unknown_objects'], 1):
            print(f"  {i}. Unknown object "
                  f"(best match: {unknown.get('confidence', 0):.2f})")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

def start_camera_cli(engine, args):
    """Start camera processing via CLI"""
    print(f"Starting camera processing (source: {args.source})")
    print("Press 'q' to quit")
    
    try:
        engine.process_video_stream(source=args.source, save_detections=args.save)
    except KeyboardInterrupt:
        print("\nCamera processing stopped")

def add_class_cli(engine, args):
    """Add a new class via CLI"""
    print(f"Adding new class: {args.name}")
    
    # Validate image paths
    valid_images = []
    for image_path in args.images:
        if os.path.exists(image_path):
            valid_images.append(image_path)
        else:
            print(f"Warning: Image '{image_path}' not found, skipping")
    
    if not valid_images:
        print("Error: No valid images provided")
        return
    
    try:
        class_id = engine.add_manual_class(
            args.name, valid_images, args.description or ""
        )
        print(f"Successfully created class '{args.name}' with ID {class_id}")
        print(f"Added {len(valid_images)} sample images")
    except Exception as e:
        print(f"Error creating class: {e}")

def discover_classes_cli(engine, args):
    """Discover new classes via CLI"""
    print(f"Discovering new classes using {args.method.upper()}")
    
    try:
        results = engine.discover_new_classes(method=args.method)
        
        if results['status'] == 'success':
            print(f"\nClustering completed successfully!")
            print(f"Method: {results['method'].upper()}")
            print(f"Objects processed: {results['total_objects']}")
            print(f"Objects clustered: {results['clustered_objects']}")
            print(f"Clusters found: {len(results['cluster_analysis'])}")
            
            if results.get('clustering_info', {}).get('silhouette_score'):
                print(f"Silhouette score: {results['clustering_info']['silhouette_score']:.3f}")
            
            # Show cluster details
            if results['cluster_analysis']:
                print("\nDiscovered clusters:")
                for cluster_id, info in results['cluster_analysis'].items():
                    suggested_name = results['suggested_names'].get(str(cluster_id), f"Cluster {cluster_id}")
                    print(f"  Cluster {cluster_id}: {info['size']} objects - '{suggested_name}'")
                
                print("\nUse the web interface to review and create classes from these clusters.")
        
        elif results['status'] == 'insufficient_data':
            print(f"Insufficient data: only {results['count']} unknown objects available")
            print("Need at least 3 objects for clustering")
        
        else:
            print(f"Clustering failed: {results.get('message', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error discovering classes: {e}")

def show_stats_cli(engine):
    """Show system statistics via CLI"""
    stats = engine.get_statistics()
    
    print("=== SYSTEM STATISTICS ===")
    print(f"Total objects processed: {stats['total_processed']}")
    print(f"Classified objects: {stats['classified_objects']}")
    print(f"Unknown objects: {stats['unknown_objects']}")
    print(f"New classes discovered: {stats['new_classes_discovered']}")
    print(f"Total classes: {stats['num_classes']}")
    print(f"Total embeddings: {stats['num_embeddings']}")
    print(f"Unknown objects in buffer: {stats['num_unknown_objects']}")
    print(f"Similarity threshold: {stats['similarity_threshold']}")

def list_classes_cli(engine):
    """List all classes via CLI"""
    classes = engine.db_manager.get_classes()
    
    if not classes:
        print("No classes found")
        return
    
    print(f"=== CLASSES ({len(classes)}) ===")
    for class_info in classes:
        print(f"ID: {class_info['id']}")
        print(f"Name: {class_info['name']}")
        print(f"Description: {class_info['description'] or 'No description'}")
        print(f"Samples: {class_info['sample_count']}")
        print(f"Created: {class_info['created_at']}")
        print("-" * 40)

if __name__ == '__main__':
    main()