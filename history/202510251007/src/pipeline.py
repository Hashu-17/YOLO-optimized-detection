import cv2
import time
import csv

from .io.source import open_video_source
from .tracking.tracker import load_model, track_frame
from .counting.line_counter import LineCounter
from .metrics.fps import FpsMeter

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
    if count_mode == "line":
        line_y = int(config.get("line_y", out_h // 2))
        direction = config.get("direction", "down")
        counter = LineCounter(line_y=line_y, direction=direction)
    meter = FpsMeter()
    csv_handle = None
    csv_writer = None
    csv_interval = int(config.get("csv_interval", 0))
    if csv_interval > 0:
        csv_handle = open(config.get("csv_path", "metrics.csv"), "a", encoding="utf-8", newline="")
        csv_writer = csv.writer(csv_handle)
        if csv_handle.tell() == 0:
            csv_writer.writerow(["frame", "count", "fps"])

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

        metrics_interval = int(config.get("metrics_interval", 0))
        fps_now = meter.update()
        if metrics_interval > 0 and frame_index % metrics_interval == 0:
            print(f"fps={fps_now:.1f} count={count}")
        if csv_writer is not None and csv_interval > 0 and frame_index % csv_interval == 0:
            csv_writer.writerow([frame_index, count, f"{fps_now:.1f}"])
        if writer is not None:
            writer.write(frame)
        if config.get("show_window", True):
            cv2.imshow("Detections + Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if csv_handle is not None:
        csv_handle.close()
    cv2.destroyAllWindows()
