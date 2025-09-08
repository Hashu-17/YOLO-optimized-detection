from ultralytics import YOLO

def load_model(path):
    return YOLO(path)

def track_frame(model, frame, conf):
    results = model.track(frame, persist=True, verbose=False)
    boxes = results[0].boxes
    detections = []
    if boxes.id is None:
        return detections
    for box, cls, track_id, score in zip(
        boxes.xyxy.cpu(),
        boxes.cls.cpu(),
        boxes.id.cpu(),
        boxes.conf.cpu(),
    ):
        if score < conf:
            continue
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "cls": int(cls),
            "track_id": int(track_id),
            "conf": float(score),
            "center": (cx, cy),
        })
    return detections
