# plate_detector.py
import cv2
from ultralytics import YOLO
import easyocr

class PlateDetector:
    def __init__(self, model_path="ai-models/license_plate.pt"):
        # YOLO plate detection model
        try:
            self.detector = YOLO(model_path)
        except Exception:
            self.detector = None

        # EASYOCR lazy init (tránh chặn khi start server)
        self.reader = None
        self._reader_langs = ['en']
        self._reader_gpu = False  # set True nếu máy bạn có GPU

    def detect_and_read(self, frame):
        plates = []

        if self.detector:
            results = self.detector(frame, verbose=False)[0]

            if results and results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Clamp to frame boundary
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # -------------------------------
                    # Lazy init EasyOCR
                    # -------------------------------
                    if self.reader is None:
                        try:
                            print("[PLATE] Initializing EasyOCR...")
                            self.reader = easyocr.Reader(self._reader_langs, gpu=self._reader_gpu)
                            print("[PLATE] EasyOCR loaded!")
                        except Exception as e:
                            print(f"[PLATE] easyocr init failed: {e}")
                            self.reader = None

                    # -------------------------------
                    # OCR
                    # -------------------------------
                    if self.reader:
                        try:
                            ocr = self.reader.readtext(crop, detail=0)
                            plate = ocr[0] if len(ocr) > 0 else ""
                        except Exception as e:
                            print(f"[PLATE] OCR error: {e}")
                            plate = ""
                    else:
                        plate = ""

                    plates.append({
                        "bbox": [x1, y1, x2, y2],
                        "plate": plate
                    })

        return plates
