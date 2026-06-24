"""
Detection — Paper §4.1

YOLOv8s-World with CLIP visual-language backbone.
Confidence threshold: 0.10 (accommodates partially occluded organisms).

8 semantic classes with synonyms:
    Fish, Coral, Diver, Rock, Sea Turtle, Shark, Starfish, Jellyfish

Scene composition per class:
    comp_c = Σ(x2-x1)(y2-y1) / A_img × 100
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLOWorld

CLASSES = [
    "Fish",
    "Coral reef",
    "Diver",
    "Rock",
    "Sea Turtle",
    "Shark",
    "Starfish",
    "Jellyfish",
]

# Initialise once at module load — avoids reloading weights per request
_model: YOLOWorld | None = None


def _get_model() -> YOLOWorld:
    global _model
    if _model is None:
        _model = YOLOWorld("yolov8s-world.pt")
        _model.set_classes(CLASSES)
    return _model


def detect_objects(image: np.ndarray) -> dict:
    """
    Run YOLOv8s-World detection on the enhanced image.

    Args:
        image: uint8 BGR image

    Returns:
        dict:
            detections    — list of {class, confidence, bbox, area_pct}
            composition   — {class_name: area_percentage} scene summary
            annotated_img — uint8 BGR image with bounding boxes drawn
    """
    model  = _get_model()
    H, W   = image.shape[:2]
    A_img  = H * W

    results = model.predict(image, conf=0.10, verbose=False)
    result  = results[0]

    detections: list[dict] = []
    composition: dict[str, float] = {c: 0.0 for c in CLASSES}

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = CLASSES[cls_id] if cls_id < len(CLASSES) else "Unknown"

            area     = (x2 - x1) * (y2 - y1)
            area_pct = area / A_img * 100.0

            detections.append({
                "class":      label,
                "confidence": round(conf, 4),
                "bbox":       [x1, y1, x2, y2],
                "area_pct":   round(area_pct, 2),
            })

            if label in composition:
                composition[label] += area_pct

    # Round composition values
    composition = {k: round(v, 2) for k, v in composition.items()}

    # Annotated image
    annotated = result.plot()

    return {
        "detections":    detections,
        "composition":   composition,
        "annotated_img": annotated,
    }
