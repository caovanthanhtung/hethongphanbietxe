# utils.py
import cv2

def draw_boxes_on_frame(frame, results, names=None):
    """Use ultralytics results[0].plot or draw custom"""
    if results and len(results) > 0:
        return results[0].plot()  # quick
    return frame
