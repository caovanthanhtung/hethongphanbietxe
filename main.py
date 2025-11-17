import cv2
import numpy as np
from ultralytics import YOLO
import time

ESP32_URL = "http://192.168.1.3:81/stream"

def main():
    print(" Bắt đầu nhận dạng phương tiện từ ESP32-CAM...")
    print(" Kết nối:", ESP32_URL)

    # Load YOLO
    model = YOLO("yolov8n.pt")

    # Mở stream
    cap = cv2.VideoCapture(ESP32_URL)
    if not cap.isOpened():
        print("Không thể mở stream. Kiểm tra ESP32-CAM!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream bị ngắt, reconnect sau 1s...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(ESP32_URL)
            continue

        # YOLO detect
        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("ESP32 Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
