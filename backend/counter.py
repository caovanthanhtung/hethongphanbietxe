# counter.py
from collections import defaultdict
import time

class VehicleCounter:
    def __init__(self, line_position=300, offset=12):
        self.line_position = line_position
        self.offset = offset
        self.counts = defaultdict(int)
        self.tracked_objects = set()
        self._last_cleanup = time.time()

    def update_counts(self, boxes, names):
        """
        boxes: ultralytics boxes object, supports .xyxy and .cls
        names: mapping id->str
        """
        # cleanup tracked_objects occasionally to avoid memory growth
        if time.time() - self._last_cleanup > 600:
            self.tracked_objects.clear()
            self._last_cleanup = time.time()

        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            cls_id = int(boxes.cls[i])
            label = names.get(cls_id, str(cls_id))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if abs(cy - self.line_position) < self.offset:
                obj_id = f"{label}_{i}_{cx}_{cy}"
                if obj_id not in self.tracked_objects:
                    self.counts[label] += 1
                    self.tracked_objects.add(obj_id)

    def get_counts(self):
        return dict(self.counts)
