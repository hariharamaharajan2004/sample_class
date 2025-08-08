import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class ClusterManager:
    """
    Manage clustering of unknown objects to discover new classes
    """
    
    def __init__(self, min_cluster_size: int = 3, max_clusters: int = 10):
        """
        Initialize cluster manager
        
        Args:
            min_cluster_size: Minimum number of objects to form a cluster
            max_clusters: Maximum number of clusters to consider
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        
    def cluster_embeddings_kmeans(self, embeddings: np.ndarray, 
                                 n_clusters: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster embeddings using K-Means
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            (cluster_labels, clustering_info)
        """
        try:
            if len(embeddings) < self.min_cluster_size:
                logger.warning(f"Not enough samples for clustering: {len(embeddings)} < {self.min_cluster_size}")
                return np.zeros(len(embeddings)), {'method': 'kmeans', 'n_clusters': 0}
            
            # Normalize embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            if n_clusters is None:
                # Determine optimal number of clusters
                n_clusters = self._find_optimal_clusters_kmeans(embeddings_scaled)
            
            if n_clusters < 2:
                logger.info("Optimal number of clusters is less than 2, no clustering performed")
                return np.zeros(len(embeddings)), {'method': 'kmeans', 'n_clusters': 0}
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_scaled)
            
            # Calculate clustering metrics
            silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
            
            # Filter out small clusters
            cluster_labels = self._filter_small_clusters(cluster_labels)
            
            clustering_info = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'cluster_centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }
            
            logger.info(f"K-Means clustering completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
            
            return cluster_labels, clustering_info
            
        except Exception as e:
            logger.error(f"Error in K-Means clustering: {e}")
            return np.zeros(len(embeddings)), {'method': 'kmeans', 'n_clusters': 0}
    
    def cluster_embeddings_dbscan(self, embeddings: np.ndarray, 
                                 eps: float = 0.5, min_samples: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster embeddings using DBSCAN
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples in a cluster
            
        Returns:
            (cluster_labels, clustering_info)
        """
        try:
            if len(embeddings) < self.min_cluster_size:
                logger.warning(f"Not enough samples for clustering: {len(embeddings)} < {self.min_cluster_size}")
                return np.full(len(embeddings), -1), {'method': 'dbscan', 'n_clusters': 0}
            
            if min_samples is None:
                min_samples = max(2, self.min_cluster_size)
            
            # Normalize embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(embeddings_scaled)
            
            # Count clusters (excluding noise points labeled as -1)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            clustering_info = {
                'method': 'dbscan',
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'eps': eps,
                'min_samples': min_samples
            }
            
            # Calculate silhouette score if we have clusters
            if n_clusters > 1:
                # Only calculate for non-noise points
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(
                        embeddings_scaled[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    clustering_info['silhouette_score'] = silhouette_avg
            
            logger.info(f"DBSCAN clustering completed: {n_clusters} clusters, {n_noise} noise points")
            
            return cluster_labels, clustering_info
            
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {e}")
            return np.full(len(embeddings), -1), {'method': 'dbscan', 'n_clusters': 0}
    
    def _find_optimal_clusters_kmeans(self, embeddings: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        try:
            max_k = min(self.max_clusters, len(embeddings) - 1)
            if max_k < 2:
                return 0
            
            inertias = []
            silhouette_scores = []
            k_range = range(2, max_k + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                inertias.append(kmeans.inertia_)
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Find elbow point
            optimal_k = self._find_elbow_point(list(k_range), inertias)
            
            # Also consider silhouette score
            best_silhouette_k = k_range[np.argmax(silhouette_scores)]
            
            # Choose the smaller k if silhouette scores are close
            if abs(silhouette_scores[optimal_k - 2] - max(silhouette_scores)) < 0.1:
                return optimal_k
            else:
                return best_silhouette_k
                
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {e}")
            return 2
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in the inertia curve"""
        try:
            if len(k_values) < 3:
                return k_values[0] if k_values else 2
            
            # Calculate the rate of change
            diffs = np.diff(inertias)
            diff_ratios = np.diff(diffs) / diffs[:-1]
            
            # Find the point with maximum change in rate
            elbow_idx = np.argmax(diff_ratios) + 1
            return k_values[elbow_idx]
            
        except Exception as e:
            logger.error(f"Error finding elbow point: {e}")
            return k_values[0] if k_values else 2
    
    def _filter_small_clusters(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Filter out clusters smaller than min_cluster_size"""
        try:
            # Count samples in each cluster
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # Find clusters that are too small
            small_clusters = unique_labels[counts < self.min_cluster_size]
            
            # Reassign small clusters to noise (-1)
            filtered_labels = cluster_labels.copy()
            for small_cluster in small_clusters:
                filtered_labels[cluster_labels == small_cluster] = -1
            
            return filtered_labels
            
        except Exception as e:
            logger.error(f"Error filtering small clusters: {e}")
            return cluster_labels
    
    def analyze_clusters(self, embeddings: np.ndarray, 
                        cluster_labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Analyze cluster properties
        
        Args:
            embeddings: Array of embeddings
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with cluster analysis
        """
        try:
            cluster_analysis = {}
            unique_labels = np.unique(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_embeddings = embeddings[cluster_mask]
                
                # Calculate cluster properties
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate intra-cluster distances
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                
                cluster_info = {
                    'size': int(np.sum(cluster_mask)),
                    'centroid': centroid.tolist(),  # Convert numpy array to list
                    'mean_distance_to_centroid': float(np.mean(distances)),
                    'std_distance_to_centroid': float(np.std(distances)),
                    'max_distance_to_centroid': float(np.max(distances)),
                    'indices': np.where(cluster_mask)[0].tolist()
                }
                
                cluster_analysis[int(label)] = cluster_info
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing clusters: {e}")
            return {}
    
    def suggest_cluster_names(self, cluster_analysis: Dict[int, Dict[str, Any]], 
                             existing_classes: List[str]) -> Dict[int, str]:
        """
        Suggest names for new clusters
        
        Args:
            cluster_analysis: Cluster analysis results
            existing_classes: List of existing class names
            
        Returns:
            Dictionary mapping cluster_id to suggested name
        """
        try:
            suggestions = {}
            
            for cluster_id, info in cluster_analysis.items():
                size = info['size']
                
                # Generate name based on cluster properties
                base_name = f"unknown_class_{cluster_id}"
                
                # Add size information
                if size >= 10:
                    size_desc = "large"
                elif size >= 5:
                    size_desc = "medium"
                else:
                    size_desc = "small"
                
                suggested_name = f"{base_name}_{size_desc}"
                
                # Ensure uniqueness
                counter = 1
                original_name = suggested_name
                while suggested_name in existing_classes or suggested_name in suggestions.values():
                    suggested_name = f"{original_name}_{counter}"
                    counter += 1
                
                suggestions[cluster_id] = suggested_name
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting cluster names: {e}")
            return {}
    
    def get_cluster_representatives(self, embeddings: np.ndarray, 
                                  cluster_labels: np.ndarray, 
                                  n_representatives: int = 3) -> Dict[int, List[int]]:
        """
        Get representative samples from each cluster
        
        Args:
            embeddings: Array of embeddings
            cluster_labels: Cluster assignments
            n_representatives: Number of representatives per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of representative indices
        """
        try:
            representatives = {}
            unique_labels = np.unique(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_embeddings = embeddings[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_embeddings) <= n_representatives:
                    # If cluster is small, use all samples
                    representatives[int(label)] = cluster_indices.tolist()
                else:
                    # Find samples closest to centroid
                    centroid = np.mean(cluster_embeddings, axis=0)
                    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                    
                    # Get indices of closest samples
                    closest_indices = np.argsort(distances)[:n_representatives]
                    representatives[int(label)] = cluster_indices[closest_indices].tolist()
            
            return representatives
            
        except Exception as e:
            logger.error(f"Error getting cluster representatives: {e}")
            return {}