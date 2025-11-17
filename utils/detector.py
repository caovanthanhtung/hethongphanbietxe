from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.names

    def detect(self, frame):
        """Trả về danh sách kết quả nhận dạng"""
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes
        return boxes, results
