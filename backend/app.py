# app.py
import os
import cv2
import time
import asyncio
import numpy as np
from datetime import datetime
from collections import deque

from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse

from detector.vehicle_detector import VehicleDetector
from detector.plate_detector import PlateDetector
from detector.attributes_detector import AttributeDetector
from detector.utils import draw_boxes_on_frame

from counter import VehicleCounter
from database import detections_col
from websocket_manager import ws_manager

# ---------------------------
# CONFIG (tweak as needed)
# ---------------------------
DETECT_FPS = float(os.getenv("DETECT_FPS", "1.0"))  # YOLO runs this many times per second
TRACK_FPS = float(os.getenv("TRACK_FPS", "15.0"))   # tracker update rate (frames/s)
OCR_COOLDOWN = float(os.getenv("OCR_COOLDOWN", "2.0"))  # seconds between OCR attempts per vehicle
EXIT_TIMEOUT = float(os.getenv("EXIT_TIMEOUT", "3.0"))  # seconds since last seen -> vehicle considered exited
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Line position (y coordinate) for crossing detection
LINE_POSITION = int(os.getenv("LINE_POSITION", "300"))
LINE_OFFSET = int(os.getenv("LINE_OFFSET", "12"))

# Zone rectangle (x1,y1,x2,y2) relative to frame - default center area
ZONE = os.getenv("ZONE", "")  # e.g. "300,200,1000,600"
if ZONE:
    zx1, zy1, zx2, zy2 = map(int, ZONE.split(","))
else:
    zx1, zy1, zx2, zy2 = 200, 150, 1080, 550

# models
VEHICLE_MODEL = os.getenv("VEHICLE_MODEL", "ai-models/vehicle.pt")
PLATE_MODEL = os.getenv("PLATE_MODEL", "ai-models/license_plate.pt")
MAKE_MODEL = os.getenv("MAKE_MODEL", "ai-models/vehicle_make.pt")

# ---------------------------
# Simple Centroid Tracker
# ---------------------------
class CentroidTracker:
    def __init__(self, max_disappeared_seconds=EXIT_TIMEOUT):
        self.next_id = 1
        self.objects = {}  # id -> centroid (x,y)
        self.last_seen = {}  # id -> timestamp
        self.bboxes = {}  # id -> bbox
        self.max_disappeared = max_disappeared_seconds

    def register(self, centroid, bbox):
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = centroid
        self.last_seen[oid] = time.time()
        self.bboxes[oid] = bbox
        return oid

    def deregister(self, oid):
        self.objects.pop(oid, None)
        self.last_seen.pop(oid, None)
        self.bboxes.pop(oid, None)

    def update(self, detections):
        """
        detections: list of bbox tuples (x1,y1,x2,y2)
        returns mapping id -> bbox for current detected objects
        """
        now = time.time()
        input_centroids = []
        input_bboxes = []

        for (x1, y1, x2, y2) in detections:
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            input_centroids.append((cX, cY))
            input_bboxes.append((x1, y1, x2, y2))

        # Nếu chưa có object nào
        if len(self.objects) == 0:
            for i, cent in enumerate(input_centroids):
                self.register(cent, input_bboxes[i])
            return {oid: self.bboxes[oid] for oid in self.objects.keys()}

        # Nếu không có detection mới, chỉ check timeout
        if len(input_centroids) == 0:
            object_ids = list(self.objects.keys())
            for oid in object_ids:
                if time.time() - self.last_seen.get(oid, 0) > self.max_disappeared:
                    self.deregister(oid)
            return {oid: self.bboxes[oid] for oid in self.objects.keys()}

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]

        # build distance matrix
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.bboxes[oid] = input_bboxes[c]
            self.last_seen[oid] = now
            used_rows.add(r)
            used_cols.add(c)

        # register new detections (cols not used)
        for j in range(len(input_centroids)):
            if j not in used_cols:
                self.register(input_centroids[j], input_bboxes[j])

        # mark disappeared
        for i, oid in enumerate(object_ids):
            if i not in used_rows:
                if time.time() - self.last_seen.get(oid, 0) > self.max_disappeared:
                    self.deregister(oid)

        return {oid: self.bboxes[oid] for oid in self.objects.keys()}


# ---------------------------
# Event Engine / Vehicle state
# ---------------------------
class VehicleState:
    def __init__(self):
        self.entered = set()
        self.zone_entered = set()
        self.line_crossed = set()
        self.last_plate = {}
        self.last_ocr_time = {}
        self.last_seen = {}
        self.last_centroid = {}
        self.exit_time = {}

vehicle_state = VehicleState()

# ---------------------------
# INIT detectors
# ---------------------------
vehicle_det = VehicleDetector(VEHICLE_MODEL)
plate_det = PlateDetector(PLATE_MODEL)
attr_det = AttributeDetector(MAKE_MODEL)

counter = VehicleCounter(line_position=LINE_POSITION, offset=LINE_OFFSET)
tracker = CentroidTracker(max_disappeared_seconds=EXIT_TIMEOUT)
latest_frame = None

async def broadcast_event(event_type, data):
    payload = {"event": event_type, "data": data}
    try:
        await ws_manager.broadcast_json(payload)
    except Exception:
        pass

# ---------------------------
# Utility helpers
# ---------------------------
def centroid_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y1 + y2)/2)

def in_zone(bbox):
    cx, cy = centroid_from_bbox(bbox)
    return (cx >= zx1 and cx <= zx2 and cy >= zy1 and cy <= zy2)

def crossed_line(last_centroid, cur_centroid, line_y=LINE_POSITION):
    if last_centroid is None:
        return False
    _, y0 = last_centroid
    _, y1 = cur_centroid
    return (y0 < line_y and y1 >= line_y) or (y0 > line_y and y1 <= line_y)

# ---------------------------
# Detection & Tracking Loop
# ---------------------------
async def detection_loop():
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] cannot open camera index", CAMERA_INDEX)
        return

    last_detect_time = 0.0
    detect_interval = 1.0 / max(0.0001, DETECT_FPS)
    track_interval = 1.0 / max(0.0001, TRACK_FPS)
    last_track_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        latest_frame = frame.copy()
        now = time.time()
        do_detect = (now - last_detect_time) >= detect_interval
        do_track = (now - last_track_time) >= track_interval

        detections = []
        if do_detect:
            try:
                results = vehicle_det.model(frame, verbose=False)
                last_detect_time = now
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box[:4])
                        detections.append((x1, y1, x2, y2))
            except Exception as e:
                print("[DETECT ERROR]", e)

        if do_track:
            mapped = tracker.update(detections)
            last_track_time = now

            for vid, bbox in mapped.items():
                cx, cy = centroid_from_bbox(bbox)
                prev_cent = vehicle_state.last_centroid.get(vid)
                vehicle_state.last_centroid[vid] = (cx, cy)
                vehicle_state.last_seen[vid] = now

                # Vehicle Enter
                if vid not in vehicle_state.entered:
                    vehicle_state.entered.add(vid)
                    event = {
                        "id": vid,
                        "timestamp": datetime.utcnow().isoformat(),
                        "class": "vehicle",
                        "bbox": bbox,
                        "plate": "",
                        "make": "",
                        "color": "",
                        "type": "enter"
                    }
                    attrs = attr_det.detect_attributes(frame, bbox)
                    event["color"] = attrs.get("color", "")
                    event["make"] = attrs.get("make", "")
                    try:
                        res = await detections_col.insert_one(event)
                        event["_id"] = str(res.inserted_id)
                    except Exception as e:
                        print("[DB ERROR insert enter]", e)
                        event["_id"] = None
                    await broadcast_event("vehicle_enter", event)

                # Zone enter
                if vid not in vehicle_state.zone_entered and in_zone(bbox):
                    vehicle_state.zone_entered.add(vid)
                    evt = {
                        "id": vid,
                        "timestamp": datetime.utcnow().isoformat(),
                        "bbox": bbox,
                        "type": "zone_enter"
                    }
                    try:
                        res = await detections_col.insert_one(evt)
                        evt["_id"] = str(res.inserted_id)
                    except Exception:
                        evt["_id"] = None
                    await broadcast_event("zone_enter", evt)

                # Line cross
                if crossed_line(prev_cent, (cx, cy)) and vid not in vehicle_state.line_crossed:
                    vehicle_state.line_crossed.add(vid)
                    counter.counts["vehicles"] = counter.counts.get("vehicles", 0) + 1
                    evt = {
                        "id": vid,
                        "timestamp": datetime.utcnow().isoformat(),
                        "bbox": bbox,
                        "type": "line_cross"
                    }
                    try:
                        res = await detections_col.insert_one(evt)
                        evt["_id"] = str(res.inserted_id)
                    except Exception:
                        evt["_id"] = None
                    await broadcast_event("line_cross", evt)

                # OCR
                last_ocr = vehicle_state.last_ocr_time.get(vid, 0)
                if time.time() - last_ocr >= OCR_COOLDOWN:
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if crop is not None and crop.size != 0:
                        plates = plate_det.detect_and_read(crop)
                        if plates and len(plates) > 0:
                            plate = plates[0].get("plate", "")
                            if plate and vehicle_state.last_plate.get(vid) != plate:
                                vehicle_state.last_plate[vid] = plate
                                vehicle_state.last_ocr_time[vid] = time.time()
                                evt = {
                                    "id": vid,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "bbox": bbox,
                                    "plate": plate,
                                    "type": "plate_detected"
                                }
                                try:
                                    res = await detections_col.insert_one(evt)
                                    evt["_id"] = str(res.inserted_id)
                                except Exception:
                                    evt["_id"] = None
                                await broadcast_event("plate_detected", evt)

        # exited vehicles
        exited = []
        for vid, last in list(vehicle_state.last_seen.items()):
            if time.time() - last > EXIT_TIMEOUT:
                exited.append(vid)

        for vid in exited:
            evt = {
                "id": vid,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "vehicle_exit"
            }
            try:
                res = await detections_col.insert_one(evt)
                evt["_id"] = str(res.inserted_id)
            except Exception:
                evt["_id"] = None

            vehicle_state.entered.discard(vid)
            vehicle_state.zone_entered.discard(vid)
            vehicle_state.line_crossed.discard(vid)
            vehicle_state.last_plate.pop(vid, None)
            vehicle_state.last_ocr_time.pop(vid, None)
            vehicle_state.last_seen.pop(vid, None)
            vehicle_state.last_centroid.pop(vid, None)
            tracker.deregister(vid)
            await broadcast_event("vehicle_exit", evt)

        await asyncio.sleep(0.001)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detection_loop())

# MJPEG
def mjpeg_generator():
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.03)
            continue
        ok, jpg = cv2.imencode(".jpg", latest_frame)
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpg.tobytes() + b"\r\n")
        time.sleep(0.03)

@app.get("/video")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/snapshot")
def api_snapshot():
    global latest_frame
    if latest_frame is None:
        return {"success": False, "error": "no frame"}
    ok, jpeg = cv2.imencode(".jpg", latest_frame)
    return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg")

@app.get("/api/history")
async def api_history(limit: int = Query(50, ge=1, le=1000)):
    cursor = detections_col.find().sort("_id", -1).limit(limit)
    docs = []
    async for d in cursor:
        d["_id"] = str(d["_id"])
        docs.append(d)
    return {"success": True, "data": docs}

@app.get("/api/stats/hourly")
async def api_stats_hourly():
    pipeline = [
        {"$addFields": {"ts_date": {"$toDate": "$timestamp"}}},
        {"$group": {"_id": {"hour": {"$hour": "$ts_date"}}, "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    cursor = detections_col.aggregate(pipeline)
    out = []
    async for row in cursor:
        out.append({"hour": row["_id"]["hour"], "count": row["count"]})
    return {"success": True, "data": out}

@app.get("/api/stats/daily")
async def api_stats_daily():
    pipeline = [
        {"$addFields": {"ts_date": {"$toDate": "$timestamp"}}},
        {"$group": {
            "_id": {
                "y": {"$year": "$ts_date"},
                "m": {"$month": "$ts_date"},
                "d": {"$dayOfMonth": "$ts_date"}
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}}
    ]
    cursor = detections_col.aggregate(pipeline)
    out = []
    async for row in cursor:
        date = f"{row['_id']['y']}-{row['_id']['m']:02}-{row['_id']['d']:02}"
        out.append({"date": date, "count": row["count"]})
    return {"success": True, "data": out}

# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

@app.get("/")
def root():
    return {"status": "ok"}
