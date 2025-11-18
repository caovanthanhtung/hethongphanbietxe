# plate_detector.py
import cv2
from ultralytics import YOLO
import easyocr

class PlateDetector:
    def __init__(self, model_path="ai-models/license_plate.pt"):
        # If you don't have a plate YOLO model, you can skip and rely on cropping near vehicles (not ideal)
        try:
            self.detector = YOLO(model_path)
        except Exception:
            self.detector = None
        # easyocr reader (will download language models on first run)
        self.reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if available

    def detect_and_read(self, frame):
        plates = []
        if self.detector:
            results = self.detector(frame, verbose=False)[0]
            if results and results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # clamp
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    try:
                        ocr = self.reader.readtext(crop, detail=0)
                        plate = ocr[0] if len(ocr) > 0 else ""
                    except Exception:
                        plate = ""
                    plates.append({"bbox":[x1,y1,x2,y2], "plate": plate})
        return plates
