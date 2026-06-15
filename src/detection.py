import cv2
import numpy as np

class DetectionService:
    def __init__(self, model_path='yolov8s-world.pt'):
        """
        Initialize the YOLO-World model for open-vocabulary detection.
        """
        # Lazy import to avoid loading heavy torch/ultralytics libraries unless needed
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "The 'ultralytics' package is not installed. "
                "Please install it using 'pip install ultralytics' to enable object detection."
            )
        
        # This will download the weights if not present
        self.model = YOLO(model_path)
        
        # Custom classes with synonyms for better recall
        # Structure: Canonical Name -> [List of synonyms]
        self.class_map = {
            "scuba diver": ["scuba diver", "diver", "person", "swimmer", "human"],
            "underwater rock": ["underwater rock", "rock", "stone", "boulder", "reef", "seabed"],
            "fish": ["fish", "school of fish"],
            "coral reef": ["coral reef", "coral"],
            "jellyfish": ["jellyfish"],
            "sea turtle": ["sea turtle", "turtle"],
            "shark": ["shark"],
            "starfish": ["starfish"]
        }
        
        # Flatten classes for the model
        self.classes = []
        for synonyms in self.class_map.values():
            self.classes.extend(synonyms)
            
        # Remove duplicates
        self.classes = list(set(self.classes))
        self.model.set_classes(self.classes)

    def detect_objects(self, image):
        """
        Run detection on the given image (numpy array BGR).
        Returns a list of detected objects and composition stats.
        """
        # Run inference with lower threshold for better recall
        results = self.model.predict(image, conf=0.10, verbose=False)
        
        detections = []
        # Track areas by Canonical Name
        class_areas = {canonical: 0 for canonical in self.class_map.keys()}
        total_image_area = image.shape[0] * image.shape[1]
        
        # Reverse map for easy lookup: synonym -> canonical
        synonym_to_canonical = {}
        for canonical, synonyms in self.class_map.items():
            for syn in synonyms:
                synonym_to_canonical[syn] = canonical
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Class ID and Name
                cls_id = int(box.cls[0])
                if cls_id < len(self.model.names):
                    raw_label = self.model.names[cls_id]
                else: 
                     raw_label = "unknown"

                # Map to canonical label
                label = synonym_to_canonical.get(raw_label, raw_label)

                # Confidence
                conf = float(box.conf[0])
                
                # Calculate Area
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Add to stats
                if label in class_areas:
                    class_areas[label] += area

                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'label': label, # Display the clean canonical name
                    'confidence': round(conf, 2)
                })
        
        # Calculate Composition percentages
        composition = {}
        detected_area_sum = sum(class_areas.values())
        
        # Avoid division by zero
        if detected_area_sum > 0:
            for cls, area in class_areas.items():
                if area > 0:
                    composition[cls] = round((area / detected_area_sum) * 100, 1)
        
        # Sort by percentage descending
        composition = dict(sorted(composition.items(), key=lambda item: item[1], reverse=True))

        return detections, composition
