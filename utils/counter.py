from collections import defaultdict

class VehicleCounter:
    def __init__(self, line_position=300, offset=10):
        """
        line_position: vị trí y của vạch đếm
        offset: độ sai lệch cho phép khi so sánh trung tâm vật thể với vạch
        """
        self.line_position = line_position
        self.offset = offset
        self.counts = defaultdict(int)
        self.tracked_objects = set()

    def update_counts(self, boxes, names):
        """Đếm phương tiện đi qua vạch"""
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            cls_id = int(boxes.cls[i])
            label = names[cls_id]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Nếu trung tâm gần vạch => coi như đã đi qua
            if abs(cy - self.line_position) < self.offset:
                obj_id = f"{label}_{i}_{cx}_{cy}"
                if obj_id not in self.tracked_objects:
                    self.counts[label] += 1
                    self.tracked_objects.add(obj_id)

    def get_counts(self):
        return dict(self.counts)
