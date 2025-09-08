import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import yt_dlp
import os

# -----------------------
# Download/Stream with yt_dlp
# -----------------------
url = "https://youtu.be/FsGPxhidwGg?si=Qcgs-4NUwzTKPySn"
ydl_opts = {"format": "best[ext=mp4]", "quiet": True}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    stream_url = info["url"]

# -----------------------
# YOLO setup
# -----------------------
track_history = defaultdict(lambda: [])
counted_ids = set()   # store unique IDs already counted

if not (os.path.exists("yolov8n_ncnn_model.param") and os.path.exists("yolov8n_ncnn_model.bin")):
    print("Exporting Model to NCNN format...")
    YOLO("yolov8n.pt").export(format="ncnn")
model = YOLO("yolov8n.pt")  
names = model.model.names

cap = cv2.VideoCapture(stream_url)
assert cap.isOpened(), "Error opening video stream"

fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

result = cv2.VideoWriter("object_tracking.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps,
                         (640, 360))  # match resized frame

# -----------------------
# Frame loop
# -----------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    results = model.track(frame, persist=True, verbose=False)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        annotator = Annotator(frame, line_width=2)

        for box, cls, track_id, conf in zip(boxes, clss, track_ids, confs):
            if conf < 0.5:
                continue

            # Direction check
            track = track_history[track_id]
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            track.append((cx, cy))
            if len(track) > 30:
                track.pop(0)

            if len(track) >= 2:
                dy = track[-1][1] - track[0][1]  # movement in y
                if dy > 20 and track_id not in counted_ids:  # moving down
                    counted_ids.add(track_id)
                    with open("footfall.txt", "w") as f:
                        f.write(str(len(counted_ids)))

            # Draw box + trail
            annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} {conf:.2f}")
            cv2.circle(frame, (cx, cy), 7, colors(int(cls), True), -1)
            cv2.polylines(frame, [np.array(track, dtype=np.int32).reshape((-1, 1, 2))],
                          False, colors(int(cls), True), 2)

    result.write(frame)
    cv2.imshow("Tracking", cv2.resize(frame, (800, 600)))

    if cv2.waitKey(1) in [ord("q"), 27]:
        break

result.release()
cap.release()
cv2.destroyAllWindows()
