import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Real-time object detection using YOLOv8
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the object detector
        
        Args:
            model_name: YOLO model name (yolov8n.pt for nano, yolov8s.pt for small, etc.)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Load YOLO model
            self.model = YOLO(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded YOLO model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected objects with bounding boxes, confidence, and class info
        """
        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy()
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Crop the detected object
                        object_crop = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                            'crop': object_crop
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: Input image
            detections: List of detections from detect_objects()
            
        Returns:
            Image with drawn bounding boxes and labels
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def process_video_stream(self, source: int = 0):
        """
        Process live video stream from camera
        
        Args:
            source: Camera source (0 for default camera)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {source}")
            return
        
        logger.info("Starting video stream processing. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Draw detections
                result_frame = self.draw_detections(frame, detections)
                
                # Display the frame
                cv2.imshow('Object Detection', result_frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Video stream interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()