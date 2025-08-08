import numpy as np
import cv2
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from .object_detector import ObjectDetector
from .feature_extractor import FeatureExtractor
from ..database.db_manager import DatabaseManager
from ..clustering.cluster_manager import ClusterManager

logger = logging.getLogger(__name__)

class ClassificationEngine:
    """
    Main engine that orchestrates object detection, feature extraction, and classification
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 unknown_buffer_size: int = 50,
                 clustering_interval: int = 20):
        """
        Initialize the classification engine
        
        Args:
            similarity_threshold: Threshold for matching objects to existing classes
            unknown_buffer_size: Size of unknown objects buffer before clustering
            clustering_interval: Number of unknown objects to accumulate before clustering
        """
        self.similarity_threshold = similarity_threshold
        self.unknown_buffer_size = unknown_buffer_size
        self.clustering_interval = clustering_interval
        
        # Initialize components
        self.detector = ObjectDetector()
        self.feature_extractor = FeatureExtractor()
        self.db_manager = DatabaseManager()
        self.cluster_manager = ClusterManager()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'classified_objects': 0,
            'unknown_objects': 0,
            'new_classes_discovered': 0
        }
        
        logger.info("Classification engine initialized")
    
    def process_image(self, image: np.ndarray, image_path: str = "") -> Dict[str, Any]:
        """
        Process a single image: detect objects, extract features, and classify
        
        Args:
            image: Input image as numpy array
            image_path: Path to the image file
            
        Returns:
            Processing results with detections and classifications
        """
        try:
            # Detect objects
            detections = self.detector.detect_objects(image)
            
            results = {
                'image_path': image_path,
                'timestamp': datetime.now().isoformat(),
                'detections': [],
                'classifications': [],
                'unknown_objects': []
            }
            
            for detection in detections:
                # Extract features from detected object
                object_crop = detection['crop']
                if object_crop.size == 0:
                    continue
                
                embedding = self.feature_extractor.extract_features(object_crop)
                
                # Try to classify the object
                classification_result = self._classify_object(embedding, detection, image_path)
                
                # Update results
                results['detections'].append(detection)
                
                if classification_result['is_known']:
                    results['classifications'].append(classification_result)
                    self.stats['classified_objects'] += 1
                else:
                    results['unknown_objects'].append(classification_result)
                    self.stats['unknown_objects'] += 1
                
                self.stats['total_processed'] += 1
            
            # Check if we should perform clustering
            self._check_and_perform_clustering()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {'error': str(e)}
    
    def process_video_stream(self, source: int = 0, save_detections: bool = True):
        """
        Process live video stream
        
        Args:
            source: Camera source
            save_detections: Whether to save detections to database
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {source}")
            return
        
        logger.info("Starting video stream processing. Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Process every nth frame to reduce computational load
                if frame_count % 5 == 0:  # Process every 5th frame
                    results = self.process_image(frame, f"camera_frame_{frame_count}")
                    
                    # Draw results on frame
                    annotated_frame = self._draw_results(frame, results)
                else:
                    # Just detect and draw for real-time feedback
                    detections = self.detector.detect_objects(frame)
                    annotated_frame = self.detector.draw_detections(frame, detections)
                
                # Display the frame
                cv2.imshow('AI Classification System', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/images/captured_frame_{timestamp}.jpg"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    cv2.imwrite(filename, frame)
                    logger.info(f"Saved frame to {filename}")
                    
        except KeyboardInterrupt:
            logger.info("Video stream interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _classify_object(self, embedding: np.ndarray, detection: Dict[str, Any], 
                        image_path: str) -> Dict[str, Any]:
        """
        Classify an object based on its embedding
        
        Args:
            embedding: Feature embedding of the object
            detection: Detection information
            image_path: Path to source image
            
        Returns:
            Classification result
        """
        try:
            # Get all class centroids
            centroids, class_ids = self.db_manager.get_all_class_centroids()
            
            if len(centroids) == 0:
                # No existing classes, add to unknown buffer
                unknown_id = self.db_manager.add_unknown_object(
                    embedding, image_path, detection['bbox']
                )
                
                return {
                    'is_known': False,
                    'unknown_id': unknown_id,
                    'detection': detection,
                    'confidence': 0.0
                }
            
            # Find most similar class
            best_match_idx, similarity = self.feature_extractor.find_most_similar(
                embedding, centroids, self.similarity_threshold
            )
            
            if best_match_idx is not None:
                # Object matches existing class
                class_id = class_ids[best_match_idx]
                
                # Add embedding to the class
                embedding_id = self.db_manager.add_embedding(
                    class_id, embedding, image_path, similarity, detection['bbox']
                )
                
                # Get class info
                classes = self.db_manager.get_classes()
                class_info = next((c for c in classes if c['id'] == class_id), None)
                
                return {
                    'is_known': True,
                    'class_id': class_id,
                    'class_name': class_info['name'] if class_info else 'Unknown',
                    'confidence': similarity,
                    'embedding_id': embedding_id,
                    'detection': detection
                }
            else:
                # Object doesn't match any existing class
                unknown_id = self.db_manager.add_unknown_object(
                    embedding, image_path, detection['bbox']
                )
                
                return {
                    'is_known': False,
                    'unknown_id': unknown_id,
                    'detection': detection,
                    'confidence': similarity,
                    'best_match_similarity': similarity
                }
                
        except Exception as e:
            logger.error(f"Error classifying object: {e}")
            return {'is_known': False, 'error': str(e)}
    
    def _check_and_perform_clustering(self):
        """Check if clustering should be performed and execute it"""
        try:
            # Get current unknown objects count
            stats = self.db_manager.get_database_stats()
            unknown_count = stats['num_unknown_objects']
            
            if unknown_count >= self.clustering_interval:
                logger.info(f"Performing clustering on {unknown_count} unknown objects")
                self.discover_new_classes()
                
        except Exception as e:
            logger.error(f"Error checking clustering condition: {e}")
    
    def discover_new_classes(self, method: str = 'kmeans') -> Dict[str, Any]:
        """
        Discover new classes by clustering unknown objects
        
        Args:
            method: Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            Clustering results and discovered classes
        """
        try:
            # Get unknown objects
            unknown_objects = self.db_manager.get_unknown_objects()
            
            if len(unknown_objects) < self.cluster_manager.min_cluster_size:
                logger.info(f"Not enough unknown objects for clustering: {len(unknown_objects)}")
                return {'status': 'insufficient_data', 'count': len(unknown_objects)}
            
            # Extract embeddings
            embeddings = np.array([obj['embedding'] for obj in unknown_objects])
            
            # Perform clustering
            if method == 'kmeans':
                cluster_labels, clustering_info = self.cluster_manager.cluster_embeddings_kmeans(embeddings)
            else:
                cluster_labels, clustering_info = self.cluster_manager.cluster_embeddings_dbscan(embeddings)
            
            # Update cluster assignments in database
            cluster_assignments = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Not noise
                    cluster_assignments[unknown_objects[i]['id']] = int(label)
            
            self.db_manager.update_cluster_assignments(cluster_assignments)
            
            # Analyze clusters
            cluster_analysis = self.cluster_manager.analyze_clusters(embeddings, cluster_labels)
            
            # Get existing class names
            existing_classes = [c['name'] for c in self.db_manager.get_classes()]
            
            # Suggest names for clusters
            suggested_names = self.cluster_manager.suggest_cluster_names(
                cluster_analysis, existing_classes
            )
            
            # Get representative samples
            representatives = self.cluster_manager.get_cluster_representatives(
                embeddings, cluster_labels
            )
            
            results = {
                'status': 'success',
                'method': method,
                'clustering_info': clustering_info,
                'cluster_analysis': cluster_analysis,
                'suggested_names': suggested_names,
                'representatives': representatives,
                'total_objects': len(unknown_objects),
                'clustered_objects': len(cluster_assignments)
            }
            
            logger.info(f"Clustering completed: {len(cluster_analysis)} clusters discovered")
            
            return results
            
        except Exception as e:
            logger.error(f"Error discovering new classes: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_class_from_cluster(self, cluster_id: int, class_name: str, 
                                 description: str = "") -> int:
        """
        Create a new class from a cluster
        
        Args:
            cluster_id: ID of the cluster to promote
            class_name: Name for the new class
            description: Description for the new class
            
        Returns:
            New class ID
        """
        try:
            class_id = self.db_manager.promote_cluster_to_class(
                cluster_id, class_name, description
            )
            
            self.stats['new_classes_discovered'] += 1
            logger.info(f"Created new class '{class_name}' from cluster {cluster_id}")
            
            return class_id
            
        except Exception as e:
            logger.error(f"Error creating class from cluster: {e}")
            raise
    
    def add_manual_class(self, class_name: str, image_paths: List[str], 
                        description: str = "") -> int:
        """
        Manually add a new class with sample images
        
        Args:
            class_name: Name of the new class
            image_paths: List of paths to sample images
            description: Class description
            
        Returns:
            New class ID
        """
        try:
            # Create the class
            class_id = self.db_manager.add_class(class_name, description)
            
            # Process sample images
            for image_path in image_paths:
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Extract features from the entire image
                        embedding = self.feature_extractor.extract_features(image)
                        
                        # Add to class
                        self.db_manager.add_embedding(
                            class_id, embedding, image_path, 1.0
                        )
            
            logger.info(f"Manually added class '{class_name}' with {len(image_paths)} samples")
            return class_id
            
        except Exception as e:
            logger.error(f"Error adding manual class: {e}")
            raise
    
    def _draw_results(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Draw classification results on image"""
        try:
            annotated_image = image.copy()
            
            # Draw classified objects in green
            for classification in results.get('classifications', []):
                detection = classification['detection']
                x1, y1, x2, y2 = detection['bbox']
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{classification['class_name']}: {classification['confidence']:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Draw unknown objects in red
            for unknown in results.get('unknown_objects', []):
                detection = unknown['detection']
                x1, y1, x2, y2 = detection['bbox']
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                label = f"Unknown: {unknown.get('confidence', 0.0):.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 0, 255), -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"Error drawing results: {e}")
            return image
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        db_stats = self.db_manager.get_database_stats()
        
        return {
            **self.stats,
            **db_stats,
            'similarity_threshold': self.similarity_threshold,
            'unknown_buffer_size': self.unknown_buffer_size
        }