import cv2
import yt_dlp

def resolve_stream(url):
    ydl_opts = {"format": "best[ext=mp4]", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]

def open_video_source(url, use_stream=True):
    source = resolve_stream(url) if use_stream else url
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Error opening video source")
    return cap
