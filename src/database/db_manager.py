import sqlite3
import json
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manage SQLite database for storing embeddings, classes, and metadata
    """
    
    def __init__(self, db_path: str = "data/classification_system.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"Database initialized at {db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Classes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS classes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sample_count INTEGER DEFAULT 0,
                    centroid_embedding BLOB
                )
            ''')
            
            # Embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_id INTEGER,
                    embedding BLOB NOT NULL,
                    image_path TEXT,
                    confidence REAL,
                    bbox TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (class_id) REFERENCES classes (id)
                )
            ''')
            
            # Unknown objects buffer table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding BLOB NOT NULL,
                    image_path TEXT,
                    bbox TEXT,
                    cluster_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Clusters table for new class discovery
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    centroid_embedding BLOB NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    proposed_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def add_class(self, name: str, description: str = "", centroid_embedding: Optional[np.ndarray] = None) -> int:
        """
        Add a new class to the database
        
        Args:
            name: Class name
            description: Class description
            centroid_embedding: Initial centroid embedding
            
        Returns:
            Class ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                centroid_blob = None
                if centroid_embedding is not None:
                    centroid_blob = pickle.dumps(centroid_embedding)
                
                cursor.execute('''
                    INSERT INTO classes (name, description, centroid_embedding)
                    VALUES (?, ?, ?)
                ''', (name, description, centroid_blob))
                
                class_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added new class '{name}' with ID {class_id}")
                return class_id
                
        except sqlite3.IntegrityError:
            logger.error(f"Class '{name}' already exists")
            raise ValueError(f"Class '{name}' already exists")
        except Exception as e:
            logger.error(f"Error adding class: {e}")
            raise
    
    def add_embedding(self, class_id: int, embedding: np.ndarray, 
                     image_path: str = "", confidence: float = 1.0, 
                     bbox: List[int] = None) -> int:
        """
        Add an embedding to a class
        
        Args:
            class_id: Class ID
            embedding: Feature embedding
            image_path: Path to source image
            confidence: Classification confidence
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Embedding ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                embedding_blob = pickle.dumps(embedding)
                bbox_json = json.dumps(bbox) if bbox else None
                
                cursor.execute('''
                    INSERT INTO embeddings (class_id, embedding, image_path, confidence, bbox)
                    VALUES (?, ?, ?, ?, ?)
                ''', (class_id, embedding_blob, image_path, confidence, bbox_json))
                
                embedding_id = cursor.lastrowid
                
                # Update class sample count and centroid
                self._update_class_centroid(cursor, class_id)
                
                conn.commit()
                return embedding_id
                
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            raise
    
    def add_unknown_object(self, embedding: np.ndarray, image_path: str = "", 
                          bbox: List[int] = None) -> int:
        """
        Add an unknown object to the buffer
        
        Args:
            embedding: Feature embedding
            image_path: Path to source image
            bbox: Bounding box coordinates
            
        Returns:
            Unknown object ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                embedding_blob = pickle.dumps(embedding)
                bbox_json = json.dumps(bbox) if bbox else None
                
                cursor.execute('''
                    INSERT INTO unknown_objects (embedding, image_path, bbox)
                    VALUES (?, ?, ?)
                ''', (embedding_blob, image_path, bbox_json))
                
                unknown_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added unknown object with ID {unknown_id}")
                return unknown_id
                
        except Exception as e:
            logger.error(f"Error adding unknown object: {e}")
            raise
    
    def get_class_embeddings(self, class_id: int) -> np.ndarray:
        """
        Get all embeddings for a specific class
        
        Args:
            class_id: Class ID
            
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT embedding FROM embeddings WHERE class_id = ?
                ''', (class_id,))
                
                rows = cursor.fetchall()
                
                if not rows:
                    return np.array([])
                
                embeddings = [pickle.loads(row[0]) for row in rows]
                return np.array(embeddings)
                
        except Exception as e:
            logger.error(f"Error getting class embeddings: {e}")
            return np.array([])
    
    def get_all_class_centroids(self) -> Tuple[np.ndarray, List[int]]:
        """
        Get centroids of all classes
        
        Returns:
            (centroids_array, class_ids)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, centroid_embedding FROM classes 
                    WHERE centroid_embedding IS NOT NULL
                ''')
                
                rows = cursor.fetchall()
                
                if not rows:
                    return np.array([]), []
                
                class_ids = [row[0] for row in rows]
                centroids = [pickle.loads(row[1]) for row in rows]
                
                return np.array(centroids), class_ids
                
        except Exception as e:
            logger.error(f"Error getting class centroids: {e}")
            return np.array([]), []
    
    def get_unknown_objects(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get unknown objects from buffer
        
        Args:
            limit: Maximum number of objects to return
            
        Returns:
            List of unknown objects with metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT id, embedding, image_path, bbox, cluster_id, created_at
                    FROM unknown_objects
                    ORDER BY created_at DESC
                '''
                
                if limit:
                    query += f' LIMIT {limit}'
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                unknown_objects = []
                for row in rows:
                    obj = {
                        'id': row[0],
                        'embedding': pickle.loads(row[1]),
                        'image_path': row[2],
                        'bbox': json.loads(row[3]) if row[3] else None,
                        'cluster_id': row[4],
                        'created_at': row[5]
                    }
                    unknown_objects.append(obj)
                
                return unknown_objects
                
        except Exception as e:
            logger.error(f"Error getting unknown objects: {e}")
            return []
    
    def get_classes(self) -> List[Dict[str, Any]]:
        """
        Get all classes with metadata
        
        Returns:
            List of classes with metadata
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, description, created_at, sample_count
                    FROM classes
                    ORDER BY name
                ''')
                
                rows = cursor.fetchall()
                
                classes = []
                for row in rows:
                    class_info = {
                        'id': row[0],
                        'name': row[1],
                        'description': row[2],
                        'created_at': row[3],
                        'sample_count': row[4]
                    }
                    classes.append(class_info)
                
                return classes
                
        except Exception as e:
            logger.error(f"Error getting classes: {e}")
            return []
    
    def update_cluster_assignments(self, cluster_assignments: Dict[int, int]):
        """
        Update cluster assignments for unknown objects
        
        Args:
            cluster_assignments: Dict mapping unknown_object_id to cluster_id
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for obj_id, cluster_id in cluster_assignments.items():
                    cursor.execute('''
                        UPDATE unknown_objects 
                        SET cluster_id = ? 
                        WHERE id = ?
                    ''', (cluster_id, obj_id))
                
                conn.commit()
                logger.info(f"Updated cluster assignments for {len(cluster_assignments)} objects")
                
        except Exception as e:
            logger.error(f"Error updating cluster assignments: {e}")
            raise
    
    def promote_cluster_to_class(self, cluster_id: int, class_name: str, 
                                description: str = "") -> int:
        """
        Promote a cluster to a new class
        
        Args:
            cluster_id: Cluster ID to promote
            class_name: Name for the new class
            description: Description for the new class
            
        Returns:
            New class ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all objects in the cluster
                cursor.execute('''
                    SELECT id, embedding, image_path, bbox
                    FROM unknown_objects
                    WHERE cluster_id = ?
                ''', (cluster_id,))
                
                cluster_objects = cursor.fetchall()
                
                if not cluster_objects:
                    raise ValueError(f"No objects found in cluster {cluster_id}")
                
                # Create new class
                class_id = self.add_class(class_name, description)
                
                # Move objects to the new class
                for obj in cluster_objects:
                    obj_id, embedding_blob, image_path, bbox_json = obj
                    embedding = pickle.loads(embedding_blob)
                    bbox = json.loads(bbox_json) if bbox_json else None
                    
                    self.add_embedding(class_id, embedding, image_path, 1.0, bbox)
                
                # Remove objects from unknown buffer
                cursor.execute('''
                    DELETE FROM unknown_objects WHERE cluster_id = ?
                ''', (cluster_id,))
                
                conn.commit()
                
                logger.info(f"Promoted cluster {cluster_id} to class '{class_name}' with {len(cluster_objects)} objects")
                return class_id
                
        except Exception as e:
            logger.error(f"Error promoting cluster to class: {e}")
            raise
    
    def _update_class_centroid(self, cursor, class_id: int):
        """Update class centroid and sample count"""
        try:
            # Get all embeddings for the class
            cursor.execute('''
                SELECT embedding FROM embeddings WHERE class_id = ?
            ''', (class_id,))
            
            rows = cursor.fetchall()
            
            if rows:
                embeddings = [pickle.loads(row[0]) for row in rows]
                embeddings_array = np.array(embeddings)
                
                # Compute centroid
                centroid = np.mean(embeddings_array, axis=0)
                centroid_blob = pickle.dumps(centroid)
                
                # Update class
                cursor.execute('''
                    UPDATE classes 
                    SET centroid_embedding = ?, sample_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (centroid_blob, len(embeddings), class_id))
                
        except Exception as e:
            logger.error(f"Error updating class centroid: {e}")
    
    def clear_unknown_objects(self):
        """Clear all unknown objects from buffer"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM unknown_objects')
                conn.commit()
                logger.info("Cleared unknown objects buffer")
                
        except Exception as e:
            logger.error(f"Error clearing unknown objects: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count classes
                cursor.execute('SELECT COUNT(*) FROM classes')
                num_classes = cursor.fetchone()[0]
                
                # Count embeddings
                cursor.execute('SELECT COUNT(*) FROM embeddings')
                num_embeddings = cursor.fetchone()[0]
                
                # Count unknown objects
                cursor.execute('SELECT COUNT(*) FROM unknown_objects')
                num_unknown = cursor.fetchone()[0]
                
                return {
                    'num_classes': num_classes,
                    'num_embeddings': num_embeddings,
                    'num_unknown_objects': num_unknown
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'num_classes': 0, 'num_embeddings': 0, 'num_unknown_objects': 0}