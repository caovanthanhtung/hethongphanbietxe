import cv2

def draw_boxes(frame, results):
    """Vẽ bounding box và label"""
    annotated_frame = results[0].plot()
    return annotated_frame

def draw_vehicle_count(frame, counts, line_y=300):
    """Hiển thị số lượng xe và vẽ vạch đếm"""
    y = 30
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    for vehicle, count in counts.items():
        text = f"{vehicle}: {count}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        y += 25
    return frame
