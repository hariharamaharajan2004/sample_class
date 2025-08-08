import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extract feature embeddings from images using pre-trained CNNs
    """
    
    def __init__(self, model_name: str = 'mobilenet_v3_small', embedding_dim: int = 512):
        """
        Initialize the feature extractor
        
        Args:
            model_name: Pre-trained model to use ('mobilenet_v3_small', 'efficientnet_b0', 'resnet18')
            embedding_dim: Dimension of output embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load pre-trained model
        self.model = self._load_model()
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Initialized {model_name} feature extractor on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load and modify pre-trained model for feature extraction"""
        
        if self.model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            # Remove classifier and add custom embedding layer
            # MobileNet v3 small has 576 features before the classifier
            model.classifier = nn.Sequential(
                nn.Linear(576, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # Remove classifier and add custom embedding layer
            # EfficientNet B0 has 1280 features before the classifier
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, self.embedding_dim),
                nn.ReLU()
            )
            
        elif self.model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            # Remove final classification layer
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model.to(self.device)
    
    def extract_features(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract feature embeddings from an image
        
        Args:
            image: Input image (numpy array in BGR format or torch tensor)
            
        Returns:
            Feature embedding as numpy array
        """
        try:
            # Convert BGR to RGB if numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Preprocess image
            if isinstance(image, np.ndarray):
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            else:
                input_tensor = image.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                # Normalize features
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(self.embedding_dim)
    
    def extract_batch_features(self, images: list) -> np.ndarray:
        """
        Extract features from a batch of images
        
        Args:
            images: List of images (numpy arrays)
            
        Returns:
            Feature embeddings as numpy array (batch_size, embedding_dim)
        """
        try:
            batch_tensors = []
            
            for image in images:
                # Convert BGR to RGB if needed
                if isinstance(image, np.ndarray):
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                tensor = self.transform(image)
                batch_tensors.append(tensor)
            
            # Stack tensors into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(batch_tensor)
                # Normalize features
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting batch features: {e}")
            return np.zeros((len(images), self.embedding_dim))
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize embeddings
            embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         embeddings_db: np.ndarray, 
                         threshold: float = 0.7) -> tuple:
        """
        Find the most similar embedding in a database
        
        Args:
            query_embedding: Query embedding
            embeddings_db: Database of embeddings (n_samples, embedding_dim)
            threshold: Similarity threshold for matching
            
        Returns:
            (best_match_index, similarity_score) or (None, 0.0) if no match above threshold
        """
        try:
            if len(embeddings_db) == 0:
                return None, 0.0
            
            # Normalize query embedding
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            # Normalize database embeddings
            embeddings_db_norm = embeddings_db / (np.linalg.norm(embeddings_db, axis=1, keepdims=True) + 1e-8)
            
            # Compute similarities
            similarities = np.dot(embeddings_db_norm, query_embedding)
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= threshold:
                return int(best_idx), float(best_similarity)
            else:
                return None, float(best_similarity)
                
        except Exception as e:
            logger.error(f"Error finding most similar: {e}")
            return None, 0.0