import os
import time
import asyncio
from datetime import datetime
import requests
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from detector.vehicle_detector import VehicleDetector
from detector.plate_detector import PlateDetector
from detector.attributes_detector import AttributeDetector
from counter import VehicleCounter
from database import detections_col
from websocket_manager import ws_manager
from detector.utils import draw_boxes_on_frame


# ============================================================
#  CHỌN NGUỒN CAMERA
# ============================================================

# --- ESP32 CAMERA (MẶC ĐỊNH CŨ) ---
ESP32_URL = "http://192.168.1.7:81/stream"

# --- WEBCAM MÁY TÍNH ---
# 0 = webcam mặc định
# Bạn có thể đổi từ biến môi trường:
# VIDEO_SOURCE=0  hoặc  VIDEO_SOURCE=1
VIDEO_SOURCE = int(os.getenv("VIDEO_SOURCE", "0"))

# Ghi chú:
# • Nếu muốn dùng lại ESP32 chỉ cần đổi:
#       VIDEO_SOURCE = ESP32_URL
# ============================================================

app = FastAPI()

# ======= detectors =======
vehicle_det = VehicleDetector(model_path=os.getenv("VEHICLE_MODEL", "models/yolov8n.pt"))
plate_det = PlateDetector(model_path=os.getenv("PLATE_MODEL", "ai-models/license_plate.pt"))
attr_det = AttributeDetector(make_model_path=os.getenv("MAKE_MODEL", "ai-models/vehicle_make.pt"))

counter = VehicleCounter(
    line_position=int(os.getenv("LINE_POS", "300")),
    offset=int(os.getenv("OFFSET", "12"))
)

latest_frame = None


# ===================== MJPEG STREAM OUTPUT ====================
def mjpeg_gen():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.03)
            continue
        ok, jpeg = cv2.imencode('.jpg', latest_frame)
        if not ok:
            continue
        frame_bytes = jpeg.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )
        time.sleep(0.03)


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ===================== MJPEG READER (ESP32) ====================
def mjpeg_reader(url):
    """
    ONLY FOR ESP32
    Đọc luồng MJPEG từ ESP32 thông qua HTTP
    """
    while True:
        try:
            stream = requests.get(url, stream=True, timeout=5)
            bytes_data = b''

            for chunk in stream.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        yield img

        except Exception:
            time.sleep(1)
            continue


# ===================== WEBCAM READER =========================
def webcam_reader(cam_id=0):
    """
    Dùng webcam máy tính
    """
    cap = cv2.VideoCapture(cam_id)

    # SET độ phân giải nếu cần
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        yield frame


# ===================== DETECTION LOOP =========================
async def detection_loop():
    global latest_frame

    while True:
        try:
            # =======================================================
            # CHỌN CAMERA Ở ĐÂY:
            #
            # --- Nếu dùng webcam:
            frame_source = webcam_reader(VIDEO_SOURCE)

            # --- Nếu muốn ESP32 → bỏ comment dòng dưới
            # frame_source = mjpeg_reader(ESP32_URL)
            # =======================================================

            for frame in frame_source:
                if frame is None:
                    continue

                # VEHICLE detection
                boxes, results = vehicle_det.detect(frame)

                if boxes is not None and len(boxes.xyxy) > 0:
                    counter.update_counts(boxes, vehicle_det.names)

                counts = counter.get_counts()
                events = []

                # PLATE + ATTR
                if boxes is not None and len(boxes.xyxy) > 0:
                    for i, box in enumerate(boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box[:4])
                        cls_id = int(boxes.cls[i])
                        label = vehicle_det.names.get(cls_id, "vehicle")

                        crop = frame[y1:y2, x1:x2]
                        if crop is None or crop.size == 0:
                            continue

                        plates = plate_det.detect_and_read(crop)
                        attrs = attr_det.detect_attributes(frame, (x1, y1, x2, y2))

                        event = {
                            "timestamp": datetime.utcnow(),
                            "class": label,
                            "bbox": [x1, y1, x2, y2],
                            "score": float(box[4]) if len(box) > 4 else None,
                            "color": attrs.get("color", ""),
                            "make": attrs.get("make", ""),
                            "plate": plates[0]["plate"] if len(plates) > 0 else ""
                        }
                        events.append(event)

                        await detections_col.insert_one(event)

                # vẽ khung
                annotated = draw_boxes_on_frame(frame.copy(), results)

                cv2.line(
                    annotated,
                    (0, counter.line_position),
                    (annotated.shape[1], counter.line_position),
                    (0, 0, 255), 2
                )

                # draw counters
                y = 30
                for k, v in counts.items():
                    cv2.putText(
                        annotated,
                        f"{k}: {v}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    y += 25

                latest_frame = annotated

                summary = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "counts": counts,
                    "events": events
                }
                await ws_manager.broadcast_json(summary)
                await asyncio.sleep(0.02)

        except Exception as e:
            print(f"[DETECTION LOOP ERROR] {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detection_loop())


# ===================== WEBSOCKET =========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
