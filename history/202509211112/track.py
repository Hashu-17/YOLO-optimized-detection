import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import os
# --- YOUTUBE DOWNLOADER ---
import yt_dlp  # make sure you have this: pip install yt-dlp

# URL of the YouTube video
url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Output filename


video_file = "input_video.mp4"

# Download if not already downloaded
if not os.path.exists(video_file):
    ydl_opts = {
        'outtmpl': video_file,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# --- YOLO + TRACKING ---
track_history = defaultdict(lambda: [])
model = YOLO("yolov8m.pt")
names = model.model.names

cap = cv2.VideoCapture(video_file)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))

result = cv2.VideoWriter("object_tracking.av_

