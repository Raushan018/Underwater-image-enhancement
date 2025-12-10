import cv2
import numpy as np
from ultralytics import YOLO

class DetectionService:
    def __init__(self, model_path='yolov8s-world.pt'):
        """
        Initialize the YOLO-World model for open-vocabulary detection.
        """
        # This will download the weights if not present
        self.model = YOLO(model_path)
        
        # Set custom classes for underwater detection
        # Note: YOLO-World allows setting custom prompts
        self.classes = ["fish", "coral", "diver", "rock"]
        self.model.set_classes(self.classes)

    def detect_objects(self, image):
        """
        Run detection on the given image (numpy array BGR).
        Returns a list of detected objects.
        """
        # Run inference
        # conf=0.25 is default, let's keep it or adjust
        results = self.model.predict(image, conf=0.25, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Class ID and Name
                cls_id = int(box.cls[0])
                # Safe lookup in case model classes differ from our custom set list access
                # YOLO-World set_classes updates the names dict
                label = self.model.names[cls_id]
                
                # Confidence
                conf = float(box.conf[0])
                
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'label': label,
                    'confidence': round(conf, 2)
                })
                
        return detections
