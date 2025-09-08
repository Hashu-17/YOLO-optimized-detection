import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
import os

# -----------------------
# Download/Stream with yt_dlp
# -----------------------
url = "https://youtu.be/FsGPxhidwGg?si=Qcgs-4NUwzTKPySn"

ydl_opts = {
    "format": "best[ext=mp4]",
    "quiet": True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    stream_url = info["url"]

# -----------------------
# YOLO setup
# -----------------------
if not (os.path.exists("yolov8n_ncnn_model.param") and os.path.exists("yolov8n_ncnn_model.bin")):
    print("Exporting Model to NCNN format...")
    YOLO("yolov8n.pt").export(format="ncnn")

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(stream_url)
assert cap.isOpened(), "Error opening video stream"

fps = cap.get(cv2.CAP_PROP_FPS) or 30
out_size = (320, 180)  # same as your resize
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter("object_detection.mp4", fourcc, fps, out_size)


# -----------------------
# Tracking + Counting
# -----------------------
unique_ids = set()
frame_count = 0
skip = 5  # detect every 5th frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip != 0:
        continue
    frame = cv2.resize(frame, (320, 180))

    # Run YOLO with tracking (ByteTrack by default)
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:  # if tracker gave IDs
        for box, cls, track_id, conf in zip(
            results[0].boxes.xyxy.cpu(),
            results[0].boxes.cls.cpu(),
            results[0].boxes.id.cpu(),
            results[0].boxes.conf.cpu()
        ):
            if conf < 0.5:
                continue

            # Save unique track IDs
            unique_ids.add(int(track_id))

            # Draw box + ID
            label = f"ID {int(track_id)} {model.model.names[int(cls)]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Write count to file
    with open("footfall.txt", "w") as f:
        f.write(str(len(unique_ids)))

    # Save + Show
    result.write(frame)
    cv2.imshow("Detections + Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------
# Cleanup
# -----------------------
result.release()
cap.release()
cv2.destroyAllWindows()
