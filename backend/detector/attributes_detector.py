# attributes_detector.py
import cv2
import numpy as np
from ultralytics import YOLO

class AttributeDetector:
    def __init__(self, make_model_path="ai-models/vehicle_make.pt"):
        # optional model to classify make; fallback to None
        try:
            self.make_model = YOLO(make_model_path)
        except Exception:
            self.make_model = None

    def dominant_color(self, crop):
        if crop is None or crop.size == 0:
            return "unknown"
        # resize small for speed
        small = cv2.resize(crop, (64,64))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = int(hsv[:,:,0].mean())
        if h < 10 or h > 160:
            return "red"
        if 10 <= h < 25:
            return "yellow"
        if 25 <= h < 70:
            return "green"
        if 70 <= h < 130:
            return "blue"
        if 130 <= h < 160:
            return "purple"
        return "unknown"

    def detect_make(self, crop):
        if self.make_model:
            try:
                res = self.make_model(crop, verbose=False)[0]
                # pick top class if exists
                if res.boxes is not None and len(res.boxes) > 0:
                    # if make model returns single label, use names mapping
                    # else fallback
                    cls_id = int(res.boxes.cls[0])
                    return self.make_model.names.get(cls_id, "")
            except Exception:
                return ""
        return ""

    def detect_attributes(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        h,w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        crop = frame[y1:y2, x1:x2]
        color = self.dominant_color(crop)
        make = self.detect_make(crop)
        return {"color": color, "make": make}
