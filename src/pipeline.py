import cv2

from .io.source import open_video_source
from .tracking.tracker import load_model, track_frame

def run_pipeline(config):
    url = config.get("video_file") or config.get("url")
    use_stream = config.get("use_stream", True) and not config.get("video_file")
    if not url:
        raise SystemExit("Missing video source")
    cap = open_video_source(url, use_stream)
    model = load_model(config.get("model", "yolov8n.pt"))
    conf = float(config.get("conf", 0.5))
    frame_skip = max(1, int(config.get("frame_skip", 1)))
    resize = config.get("resize")
    if resize:
        out_w, out_h = int(resize[0]), int(resize[1])
    else:
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if config.get("save_video", True):
        writer = cv2.VideoWriter(
            config.get("output_path", "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS) or 30,
            (out_w, out_h),
        )

    count_mode = config.get("count_mode", "ids")
    seen_ids = set()
    counter = None

    frame_index = 0
    last_count = -1
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_index += 1
        if frame_skip > 1 and frame_index % frame_skip != 0:
            continue
        if resize:
            frame = cv2.resize(frame, (out_w, out_h))

        detections = track_frame(model, frame, conf)
        if counter is not None:
            count = counter.update(detections)
        else:
            for det in detections:
                seen_ids.add(det["track_id"])
            count = len(seen_ids)

        if config.get("write_footfall", True) and count != last_count:
            with open("footfall.txt", "w", encoding="utf-8") as handle:
                handle.write(str(count))
            last_count = count

        fps_now = 0.0
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
