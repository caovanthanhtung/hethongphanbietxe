# vehicle_detector.py
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt", conf=0.35):
        self.model = YOLO(model_path)
        self.conf = conf
        # names mapping
        self.names = self.model.names

    def detect(self, frame):
        """
        frame: BGR numpy array
        returns: boxes, results_object
        """
        results = self.model(frame, verbose=False, conf=self.conf, imgsz=640)
        boxes = results[0].boxes
        return boxes, results
