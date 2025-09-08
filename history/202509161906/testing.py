
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import yt_dlp

# -----------------------
# Download/Stream with yt_dlp
# -----------------------
import yt_dlp

url = "https://youtu.be/BLc8s-_tsiQ?si=-K8GrM0AbDlN9QSg"

ydl_opts = {
    "format": "best",  # safe default, will always resolve to a single playable stream
    "quiet": True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    stream_url = info.get("url")  # always present with format="best"

print("Stream URL:", stream_url)


# -----------------------

# YOLO setup
# -----------------------
track_history = defaultdict(lambda: [])
model = YOLO("yolov8m.pt")  # or .to("cuda") if GPU
names = model.model.names

cap = cv2.VideoCapture(stream_url)
assert cap.isOpened(), "Error opening video stream"

w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

result = cv2.VideoWriter("object_tracking.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps,
                         (w, h))

# -----------------------
# Frame loop
# -----------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, verbose=False)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        annotator = Annotator(frame, line_width=2)

        for box, cls, track_id, conf in zip(boxes, clss, track_ids, confs):
            if conf < 0.5:  # skip low-confidence
                continue

            annotator.box_label(box, color=colors(int(cls), True), label=f"{names[int(cls)]} {conf:.2f}")

            # Tracking trail
            track = track_history[track_id]
            track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            if len(track) > 30:
                track.pop(0)

            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.circle(frame, track[-1], 7, colors(int(cls), True), -1)
            cv2.polylines(frame, [points], False, colors(int(cls), True), 2)

    result.write(frame)
    cv2.imshow("Tracking", cv2.resize(frame, (800, 600)))

    if cv2.waitKey(1) in [ord("q"), 27]:
        break

result.release()
cap.release()
cv2.destroyAllWindows()
