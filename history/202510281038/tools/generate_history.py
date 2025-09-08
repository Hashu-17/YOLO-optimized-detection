#!/usr/bin/env python3

"""History snapshot generator for YOLO-optimized-detection.

Creates sequential snapshots in history/<timestamp> by copying the previous
snapshot (or repo root for the first entry) and applying a small change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple
import argparse
import json
import shutil


REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = REPO_ROOT / "history"

EXCLUDE_DIRS = {
    ".git",
    "history",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".mypy_cache",
    "yolov8m_ncnn_model",
    "yolov8n_ncnn_model",
}
EXCLUDE_FILES = {
    "object_detection.mp4",
    "object_tracking.mp4",
    "output.mp4",
    "yolov8m.pt",
    "yolov8n.pt",
    "yolov8m.torchscript",
    "yolov8n.torchscript",
}
EXCLUDE_SUFFIXES = {".mp4", ".pt", ".torchscript", ".bin", ".param"}


@dataclass
class Step:
    timestamp: str
    index: int


@dataclass
class Change:
    message: str
    writes: List[Tuple[Path, str]]


@dataclass
class State:
    video_url: str = "https://youtu.be/FsGPxhidwGg?si=Qcgs-4NUwzTKPySn"
    use_stream: bool = True
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.5
    frame_skip: int = 5
    resize_w: int = 320
    resize_h: int = 180
    line_y: int = 180
    line_direction: str = "down"
    metrics_interval: int = 30
    csv_interval: int = 30
    csv_path: str = "metrics.csv"
    output_path: str = "object_detection.mp4"
    output_revision: int = 0
    count_mode: str = "ids"
    save_video: bool = True
    show_window: bool = True
    write_footfall: bool = True
    pipeline_line: bool = False
    pipeline_metrics: bool = False
    pipeline_csv: bool = False
    pipeline_preview: bool = False
    pipeline_writer: bool = False
    run_supports_config: bool = False
    run_supports_preset: bool = False
    run_supports_video: bool = False
    run_supports_overrides: bool = False
    has_src_init: bool = False
    has_io_init: bool = False
    has_io_source: bool = False
    has_tracking_init: bool = False
    has_tracker: bool = False
    has_counting_init: bool = False
    has_line_counter: bool = False
    has_metrics_init: bool = False
    has_fps: bool = False
    has_pipeline: bool = False
    has_run: bool = False
    has_config_default: bool = False
    has_config_edge: bool = False
    has_config_highres: bool = False
    has_docs_quickstart: bool = False
    has_docs_pipeline: bool = False
    has_docs_optimization: bool = False
    has_samples_basic: bool = False
    has_samples_line: bool = False
    has_tests_line: bool = False
    has_tests_config: bool = False
    has_cpp_readme: bool = False
    has_cpp_preprocess_h: bool = False
    has_cpp_preprocess_cpp: bool = False
    has_cpp_cmake: bool = False
    has_cpp_nms_h: bool = False
    has_cpp_nms_cpp: bool = False
    has_root_readme: bool = False
    conf_direction: int = -1
    skip_direction: int = -1
    line_direction_delta: int = 1
    metrics_direction: int = 1
    csv_direction: int = 1
    cpp_stage: int = 0
    nms_stage: int = 0


def load_steps() -> List[Step]:
    timestamps: List[str] = []
    if HISTORY_DIR.exists():
        for item in HISTORY_DIR.iterdir():
            if item.is_dir() and item.name.isdigit() and len(item.name) == 12:
                timestamps.append(item.name)
    timestamps.sort()
    return [Step(timestamp=ts, index=idx) for idx, ts in enumerate(timestamps)]


def should_skip(item: Path) -> bool:
    if item.name in EXCLUDE_DIRS:
        return True
    if item.is_file():
        if item.name in EXCLUDE_FILES:
            return True
        if item.suffix in EXCLUDE_SUFFIXES:
            return True
    return False


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if should_skip(item):
            continue
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def copy_repo_root(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in REPO_ROOT.iterdir():
        if should_skip(item):
            continue
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            shutil.copy2(item, target)


def copy_snapshot(src: Path, dst: Path) -> None:
    copy_tree(src, dst)


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def prepare_snapshot_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"snapshot not empty: {path}")
        clear_dir(path)
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def change(message: str, *writes: Tuple[Path, str]) -> Change:
    if len(writes) == 0:
        raise SystemExit("No file updates declared for step")
    if len(writes) > 2:
        raise SystemExit("Too many files touched in one step")
    return Change(message=message, writes=list(writes))


def config_payload(state: State) -> dict:
    payload = {
        "url": state.video_url,
        "use_stream": state.use_stream,
        "model": state.model_path,
        "conf": round(state.conf_threshold, 2),
        "frame_skip": state.frame_skip,
        "resize": [state.resize_w, state.resize_h],
        "count_mode": state.count_mode,
        "line_y": state.line_y,
        "direction": state.line_direction,
        "save_video": state.save_video if state.pipeline_writer else False,
        "output_path": state.output_path,
        "write_footfall": state.write_footfall,
        "show_window": state.show_window if state.pipeline_preview else False,
        "metrics_interval": state.metrics_interval if state.pipeline_metrics else 0,
        "csv_interval": state.csv_interval if state.pipeline_csv else 0,
        "csv_path": state.csv_path,
    }
    return payload


def render_config_default(state: State) -> str:
    return json.dumps(config_payload(state), indent=2) + "\n"


def render_config_edge(state: State) -> str:
    payload = config_payload(state)
    payload["model"] = "yolov8n.pt"
    payload["frame_skip"] = max(payload["frame_skip"], 6)
    payload["resize"] = [320, 180]
    payload["count_mode"] = "line"
    payload["line_y"] = state.line_y
    return json.dumps(payload, indent=2) + "\n"


def render_config_highres(state: State) -> str:
    payload = config_payload(state)
    payload["model"] = "yolov8m.pt"
    payload["frame_skip"] = max(1, state.frame_skip - 1)
    payload["resize"] = [640, 360]
    payload["count_mode"] = "ids"
    return json.dumps(payload, indent=2) + "\n"


def render_root_readme(state: State) -> str:
    lines = [
        "# edge-traffic-footfall-test",
        "",
        "YOLO tracking and footfall counting experiments.",
        "",
        "## Quick start",
        "python newtesting2.py",
    ]
    if state.has_run:
        lines.extend(["", "## Pipeline", "python run.py --config config/default.json"])
    lines.append("")
    return "\n".join(lines)


def render_quickstart(state: State) -> str:
    lines = [
        "# Quickstart",
        "",
        "## Install",
        "pip install -r requirements.txt",
        "",
        "## Run",
    ]
    if state.has_run:
        lines.append("python run.py --config config/default.json")
        if state.run_supports_preset:
            lines.append("python run.py --preset edge")
        lines.extend(["", "## Legacy", "python newtesting2.py"])
    else:
        lines.append("python newtesting2.py")
    lines.append("")
    return "\n".join(lines)


def render_pipeline_doc(state: State) -> str:
    count_mode = "line" if state.pipeline_line else "ids"
    lines = [
        "# Pipeline",
        "",
        "The pipeline uses small modules under src/ to keep IO, tracking,",
        "counting, and metrics separate.",
        "",
        "## Stages",
        "- io: resolve stream or local video",
        "- tracking: wrap YOLO track results",
        "- counting: count unique IDs or line crossings",
        "- metrics: fps and simple counters",
        "",
        f"Default count mode: {count_mode}",
        "",
    ]
    return "\n".join(lines)


def render_optimization_doc(state: State) -> str:
    lines = [
        "# Optimization Notes",
        "",
        "Optional native helpers live in cpp/ for preprocessing and NMS.",
        "They are stubs meant for experimentation.",
        "",
        "## Build",
        "mkdir -p cpp/build",
        "cd cpp/build",
        "cmake ..",
        "cmake --build .",
        "",
    ]
    return "\n".join(lines)


def render_src_init(state: State) -> str:
    return "\n".join([
        "\"\"\"Pipeline package.\"\"\"",
        "",
        "__all__ = [\"pipeline\"]",
        "",
    ])


def render_io_init(state: State) -> str:
    return "\n".join([
        "\"\"\"Input helpers.\"\"\"",
        "",
    ])


def render_source_py(state: State) -> str:
    return "\n".join([
        "import cv2",
        "import yt_dlp",
        "",
        "def resolve_stream(url):",
        "    ydl_opts = {\"format\": \"best[ext=mp4]\", \"quiet\": True}",
        "    with yt_dlp.YoutubeDL(ydl_opts) as ydl:",
        "        info = ydl.extract_info(url, download=False)",
        "        return info[\"url\"]",
        "",
        "def open_video_source(url, use_stream=True):",
        "    source = resolve_stream(url) if use_stream else url",
        "    cap = cv2.VideoCapture(source)",
        "    if not cap.isOpened():",
        "        raise RuntimeError(\"Error opening video source\")",
        "    return cap",
        "",
    ])


def render_tracking_init(state: State) -> str:
    return "\n".join([
        "\"\"\"Tracking helpers.\"\"\"",
        "",
    ])


def render_tracker_py(state: State) -> str:
    return "\n".join([
        "from ultralytics import YOLO",
        "",
        "def load_model(path):",
        "    return YOLO(path)",
        "",
        "def track_frame(model, frame, conf):",
        "    results = model.track(frame, persist=True, verbose=False)",
        "    boxes = results[0].boxes",
        "    detections = []",
        "    if boxes.id is None:",
        "        return detections",
        "    for box, cls, track_id, score in zip(",
        "        boxes.xyxy.cpu(),",
        "        boxes.cls.cpu(),",
        "        boxes.id.cpu(),",
        "        boxes.conf.cpu(),",
        "    ):",
        "        if score < conf:",
        "            continue",
        "        x1, y1, x2, y2 = map(int, box)",
        "        cx = (x1 + x2) // 2",
        "        cy = (y1 + y2) // 2",
        "        detections.append({",
        "            \"bbox\": (x1, y1, x2, y2),",
        "            \"cls\": int(cls),",
        "            \"track_id\": int(track_id),",
        "            \"conf\": float(score),",
        "            \"center\": (cx, cy),",
        "        })",
        "    return detections",
        "",
    ])


def render_counting_init(state: State) -> str:
    return "\n".join([
        "\"\"\"Counting helpers.\"\"\"",
        "",
    ])


def render_line_counter_py(state: State) -> str:
    return "\n".join([
        "class LineCounter:",
        "    def __init__(self, line_y, direction=\"down\", history=30):",
        "        self.line_y = line_y",
        "        self.direction = direction",
        "        self.history = history",
        "        self.track_history = {}",
        "        self.counted = set()",
        "",
        "    def update(self, detections):",
        "        for det in detections:",
        "            track_id = det[\"track_id\"]",
        "            cx, cy = det[\"center\"]",
        "            track = self.track_history.setdefault(track_id, [])",
        "            track.append((cx, cy))",
        "            if len(track) > self.history:",
        "                track.pop(0)",
        "            if len(track) < 2:",
        "                continue",
        "            prev_y = track[-2][1]",
        "            curr_y = track[-1][1]",
        "            if self.direction == \"down\":",
        "                crossed = prev_y < self.line_y <= curr_y",
        "            else:",
        "                crossed = prev_y > self.line_y >= curr_y",
        "            if crossed:",
        "                self.counted.add(track_id)",
        "        return len(self.counted)",
        "",
    ])


def render_metrics_init(state: State) -> str:
    return "\n".join([
        "\"\"\"Metrics helpers.\"\"\"",
        "",
    ])


def render_fps_py(state: State) -> str:
    return "\n".join([
        "import time",
        "",
        "class FpsMeter:",
        "    def __init__(self):",
        "        self.start = time.time()",
        "        self.frames = 0",
        "",
        "    def update(self):",
        "        self.frames += 1",
        "        elapsed = max(time.time() - self.start, 0.001)",
        "        return self.frames / elapsed",
        "",
    ])


def render_pipeline_py(state: State) -> str:
    lines: List[str] = ["import cv2"]
    if state.pipeline_metrics:
        lines.append("import time")
    if state.pipeline_csv:
        lines.append("import csv")
    lines.extend([
        "",
        "from .io.source import open_video_source",
        "from .tracking.tracker import load_model, track_frame",
    ])
    if state.pipeline_line:
        lines.append("from .counting.line_counter import LineCounter")
    if state.pipeline_metrics:
        lines.append("from .metrics.fps import FpsMeter")
    lines.extend([
        "",
        "def run_pipeline(config):",
        "    url = config.get(\"video_file\") or config.get(\"url\")",
        "    use_stream = config.get(\"use_stream\", True) and not config.get(\"video_file\")",
        "    if not url:",
        "        raise SystemExit(\"Missing video source\")",
        "    cap = open_video_source(url, use_stream)",
        "    model = load_model(config.get(\"model\", \"yolov8n.pt\"))",
        "    conf = float(config.get(\"conf\", 0.5))",
        "    frame_skip = max(1, int(config.get(\"frame_skip\", 1)))",
        "    resize = config.get(\"resize\")",
        "    if resize:",
        "        out_w, out_h = int(resize[0]), int(resize[1])",
        "    else:",
        "        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))",
        "        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))",
        "",
        "    writer = None",
    ])
    if state.pipeline_writer:
        lines.extend([
            "    if config.get(\"save_video\", True):",
            "        writer = cv2.VideoWriter(",
            "            config.get(\"output_path\", \"output.mp4\"),",
            "            cv2.VideoWriter_fourcc(*\"mp4v\"),",
            "            cap.get(cv2.CAP_PROP_FPS) or 30,",
            "            (out_w, out_h),",
            "        )",
        ])
    lines.extend([
        "",
        "    count_mode = config.get(\"count_mode\", \"ids\")",
        "    seen_ids = set()",
        "    counter = None",
    ])
    if state.pipeline_line:
        lines.extend([
            "    if count_mode == \"line\":",
            "        line_y = int(config.get(\"line_y\", out_h // 2))",
            "        direction = config.get(\"direction\", \"down\")",
            "        counter = LineCounter(line_y=line_y, direction=direction)",
        ])
    if state.pipeline_metrics:
        lines.append("    meter = FpsMeter()")
    if state.pipeline_csv:
        lines.extend([
            "    csv_handle = None",
            "    csv_writer = None",
            "    csv_interval = int(config.get(\"csv_interval\", 0))",
            "    if csv_interval > 0:",
            "        csv_handle = open(config.get(\"csv_path\", \"metrics.csv\"), \"a\", encoding=\"utf-8\", newline=\"\")",
            "        csv_writer = csv.writer(csv_handle)",
            "        if csv_handle.tell() == 0:",
            "            csv_writer.writerow([\"frame\", \"count\", \"fps\"])",
        ])
    lines.extend([
        "",
        "    frame_index = 0",
        "    last_count = -1",
        "    while cap.isOpened():",
        "        success, frame = cap.read()",
        "        if not success:",
        "            break",
        "        frame_index += 1",
        "        if frame_skip > 1 and frame_index % frame_skip != 0:",
        "            continue",
        "        if resize:",
        "            frame = cv2.resize(frame, (out_w, out_h))",
        "",
        "        detections = track_frame(model, frame, conf)",
        "        if counter is not None:",
        "            count = counter.update(detections)",
        "        else:",
        "            for det in detections:",
        "                seen_ids.add(det[\"track_id\"])",
        "            count = len(seen_ids)",
        "",
        "        if config.get(\"write_footfall\", True) and count != last_count:",
        "            with open(\"footfall.txt\", \"w\", encoding=\"utf-8\") as handle:",
        "                handle.write(str(count))",
        "            last_count = count",
        "",
    ])
    if state.pipeline_metrics:
        lines.extend([
            "        metrics_interval = int(config.get(\"metrics_interval\", 0))",
            "        fps_now = meter.update()",
            "        if metrics_interval > 0 and frame_index % metrics_interval == 0:",
            "            print(f\"fps={fps_now:.1f} count={count}\")",
        ])
    else:
        lines.append("        fps_now = 0.0")
    if state.pipeline_csv:
        lines.extend([
            "        if csv_writer is not None and csv_interval > 0 and frame_index % csv_interval == 0:",
            "            csv_writer.writerow([frame_index, count, f\"{fps_now:.1f}\"])",
        ])
    if state.pipeline_writer:
        lines.extend([
            "        if writer is not None:",
            "            writer.write(frame)",
        ])
    if state.pipeline_preview:
        lines.extend([
            "        if config.get(\"show_window\", True):",
            "            cv2.imshow(\"Detections + Tracking\", frame)",
            "            if cv2.waitKey(1) & 0xFF == ord(\"q\"):",
            "                break",
        ])
    lines.extend([
        "",
        "    cap.release()",
    ])
    if state.pipeline_writer:
        lines.extend([
            "    if writer is not None:",
            "        writer.release()",
        ])
    if state.pipeline_csv:
        lines.append("    if csv_handle is not None:")
        lines.append("        csv_handle.close()")
    if state.pipeline_preview:
        lines.append("    cv2.destroyAllWindows()")
    lines.append("")
    return "\n".join(lines)


def render_run_py(state: State) -> str:
    lines = [
        "import argparse",
        "import json",
        "from pathlib import Path",
        "",
        "from src.pipeline import run_pipeline",
        "",
        "def load_json(path):",
        "    if not Path(path).exists():",
        "        return {}",
        "    with open(path, \"r\", encoding=\"utf-8\") as handle:",
        "        return json.load(handle)",
        "",
        "def main():",
        "    parser = argparse.ArgumentParser()",
    ]
    if state.run_supports_config:
        lines.append("    parser.add_argument(\"--config\", default=\"config/default.json\")")
    if state.run_supports_preset:
        lines.append("    parser.add_argument(\"--preset\", choices=[\"edge\", \"highres\"])" )
    lines.append("    parser.add_argument(\"--url\")")
    if state.run_supports_video:
        lines.append("    parser.add_argument(\"--video\")")
    if state.run_supports_overrides:
        lines.append("    parser.add_argument(\"--conf\", type=float)")
        lines.append("    parser.add_argument(\"--skip\", type=int)")
        lines.append("    parser.add_argument(\"--resize\", nargs=2, type=int)")
    lines.append("    parser.add_argument(\"--output\")")
    lines.append("    args = parser.parse_args()")
    lines.append("")
    lines.append("    config = load_json(\"config/default.json\")")
    if state.run_supports_preset:
        lines.append("    if args.preset:")
        lines.append("        config.update(load_json(f\"config/{args.preset}.json\"))")
    if state.run_supports_config:
        lines.append("    if args.config and args.config != \"config/default.json\":")
        lines.append("        config.update(load_json(args.config))")
    lines.extend([
        "    if args.url:",
        "        config[\"url\"] = args.url",
        "        config[\"use_stream\"] = True",
    ])
    if state.run_supports_video:
        lines.extend([
            "    if args.video:",
            "        config[\"video_file\"] = args.video",
            "        config[\"use_stream\"] = False",
        ])
    if state.run_supports_overrides:
        lines.extend([
            "    if args.conf is not None:",
            "        config[\"conf\"] = max(0.1, min(args.conf, 0.95))",
            "    if args.skip:",
            "        config[\"frame_skip\"] = max(1, args.skip)",
            "    if args.resize:",
            "        config[\"resize\"] = [max(1, args.resize[0]), max(1, args.resize[1])]",
        ])
    lines.extend([
        "    if args.output:",
        "        config[\"output_path\"] = args.output",
        "",
        "    run_pipeline(config)",
        "",
        "if __name__ == \"__main__\":",
        "    main()",
        "",
    ])
    return "\n".join(lines)


def render_sample_config(state: State) -> str:
    payload = config_payload(state)
    payload["url"] = state.video_url
    payload["count_mode"] = "ids"
    return json.dumps(payload, indent=2) + "\n"


def render_line_sample(state: State) -> str:
    payload = config_payload(state)
    payload["count_mode"] = "line"
    payload["line_y"] = state.line_y
    payload["direction"] = state.line_direction
    return json.dumps(payload, indent=2) + "\n"


def render_test_line_counter(state: State) -> str:
    return "\n".join([
        "from src.counting.line_counter import LineCounter",
        "",
        "def det(track_id, cx, cy):",
        "    return {\"track_id\": track_id, \"center\": (cx, cy)}",
        "",
        "def test_line_counter_counts_once():",
        "    counter = LineCounter(line_y=100, direction=\"down\")",
        "    counter.update([det(1, 10, 90)])",
        "    counter.update([det(1, 10, 110)])",
        "    assert counter.update([]) == 1",
        "",
        "def test_line_counter_up_direction():",
        "    counter = LineCounter(line_y=100, direction=\"up\")",
        "    counter.update([det(2, 10, 110)])",
        "    counter.update([det(2, 10, 90)])",
        "    assert counter.update([]) == 1",
        "",
    ])


def render_test_config(state: State) -> str:
    return "\n".join([
        "import json",
        "from pathlib import Path",
        "",
        "def test_default_config_has_required_keys():",
        "    root = Path(__file__).resolve().parents[1]",
        "    cfg_path = root / \"config\" / \"default.json\"",
        "    data = json.loads(cfg_path.read_text(encoding=\"utf-8\"))",
        "    required = {\"url\", \"model\", \"conf\", \"frame_skip\"}",
        "    assert required.issubset(set(data.keys()))",
        "",
    ])


def render_cpp_readme(state: State) -> str:
    return "\n".join([
        "# Native helpers",
        "",
        "Small C++ helpers for preprocessing and NMS experiments.",
        "These are optional and are not required for Python runs.",
        "",
    ])


def render_cpp_preprocess_h(state: State) -> str:
    return "\n".join([
        "#pragma once",
        "#include <cstdint>",
        "",
        "void bgr_to_rgb(const uint8_t* src, int width, int height, uint8_t* dst);",
        "void normalize_rgb(const uint8_t* src, int width, int height, float* dst);",
        "",
    ])


def render_cpp_preprocess_cpp(state: State) -> str:
    lines = [
        "#include \"preprocess.h\"",
        "#include <algorithm>",
        "",
        "void bgr_to_rgb(const uint8_t* src, int width, int height, uint8_t* dst) {",
        "    const int total = width * height;",
        "    for (int i = 0; i < total; ++i) {",
        "        const int idx = i * 3;",
        "        dst[idx + 0] = src[idx + 2];",
        "        dst[idx + 1] = src[idx + 1];",
        "        dst[idx + 2] = src[idx + 0];",
        "    }",
        "}",
        "",
        "void normalize_rgb(const uint8_t* src, int width, int height, float* dst) {",
        "    const int total = width * height * 3;",
        "    for (int i = 0; i < total; ++i) {",
        "        dst[i] = static_cast<float>(src[i]) / 255.0f;",
        "    }",
        "}",
    ]
    if state.cpp_stage > 0:
        lines.extend([
            "",
            "void chw_pack(const float* src, int width, int height, float* dst) {",
            "    const int plane = width * height;",
            "    for (int y = 0; y < height; ++y) {",
            "        for (int x = 0; x < width; ++x) {",
            "            const int idx = (y * width + x) * 3;",
            "            dst[y * width + x] = src[idx + 0];",
            "            dst[plane + y * width + x] = src[idx + 1];",
            "            dst[2 * plane + y * width + x] = src[idx + 2];",
            "        }",
            "    }",
            "}",
        ])
    lines.append("")
    return "\n".join(lines)


def render_cpp_nms_h(state: State) -> str:
    return "\n".join([
        "#pragma once",
        "#include <vector>",
        "",
        "struct Box {",
        "    float x1;",
        "    float y1;",
        "    float x2;",
        "    float y2;",
        "    float score;",
        "};",
        "",
        "std::vector<int> nms(const std::vector<Box>& boxes, float iou_threshold);",
        "",
    ])


def render_cpp_nms_cpp(state: State) -> str:
    lines = [
        "#include \"nms.h\"",
        "#include <algorithm>",
        "",
        "static float iou(const Box& a, const Box& b) {",
        "    const float xx1 = std::max(a.x1, b.x1);",
        "    const float yy1 = std::max(a.y1, b.y1);",
        "    const float xx2 = std::min(a.x2, b.x2);",
        "    const float yy2 = std::min(a.y2, b.y2);",
        "    const float w = std::max(0.0f, xx2 - xx1);",
        "    const float h = std::max(0.0f, yy2 - yy1);",
        "    const float inter = w * h;",
        "    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);",
        "    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);",
        "    const float uni = area_a + area_b - inter;",
        "    if (uni <= 0.0f) {",
        "        return 0.0f;",
        "    }",
        "    return inter / uni;",
        "}",
        "",
        "std::vector<int> nms(const std::vector<Box>& boxes, float iou_threshold) {",
        "    std::vector<int> order(boxes.size());",
        "    for (size_t i = 0; i < boxes.size(); ++i) {",
        "        order[i] = static_cast<int>(i);",
        "    }",
        "    std::sort(order.begin(), order.end(), [&](int a, int b) {",
        "        return boxes[a].score > boxes[b].score;",
        "    });",
        "    std::vector<int> keep;",
        "    std::vector<bool> suppressed(boxes.size(), false);",
        "    for (size_t i = 0; i < order.size(); ++i) {",
        "        int idx = order[i];",
        "        if (suppressed[idx]) {",
        "            continue;",
        "        }",
        "        keep.push_back(idx);",
        "        for (size_t j = i + 1; j < order.size(); ++j) {",
        "            int next = order[j];",
        "            if (suppressed[next]) {",
        "                continue;",
        "            }",
        "            if (iou(boxes[idx], boxes[next]) >= iou_threshold) {",
        "                suppressed[next] = true;",
        "            }",
        "        }",
        "    }",
        "    return keep;",
        "}",
        "",
    ]
    if state.nms_stage > 0:
        lines.insert(0, "// Simple NMS helper for experiments")
    return "\n".join(lines)


def render_cpp_cmake(state: State) -> str:
    return "\n".join([
        "cmake_minimum_required(VERSION 3.16)",
        "project(yolo_optim CXX)",
        "",
        "add_library(yolo_preprocess STATIC preprocess.cpp)",
        "target_include_directories(yolo_preprocess PUBLIC ${CMAKE_CURRENT_LIST_DIR})",
        "",
        "add_library(yolo_nms STATIC nms.cpp)",
        "target_include_directories(yolo_nms PUBLIC ${CMAKE_CURRENT_LIST_DIR})",
        "",
    ])


def add_docs_quickstart(state: State) -> Change:
    state.has_docs_quickstart = True
    return change("add quickstart doc", (Path("docs/quickstart.md"), render_quickstart(state)))


def add_config_default(state: State) -> Change:
    state.has_config_default = True
    return change("add default config", (Path("config/default.json"), render_config_default(state)))


def add_src_init(state: State) -> Change:
    state.has_src_init = True
    return change("add src package", (Path("src/__init__.py"), render_src_init(state)))


def add_io_init(state: State) -> Change:
    state.has_io_init = True
    return change("add io package", (Path("src/io/__init__.py"), render_io_init(state)))


def add_io_source(state: State) -> Change:
    state.has_io_source = True
    return change("add video source helper", (Path("src/io/source.py"), render_source_py(state)))


def add_tracking_init(state: State) -> Change:
    state.has_tracking_init = True
    return change("add tracking package", (Path("src/tracking/__init__.py"), render_tracking_init(state)))


def add_tracker(state: State) -> Change:
    state.has_tracker = True
    return change("add tracker wrapper", (Path("src/tracking/tracker.py"), render_tracker_py(state)))


def add_counting_init(state: State) -> Change:
    state.has_counting_init = True
    return change("add counting package", (Path("src/counting/__init__.py"), render_counting_init(state)))


def add_line_counter(state: State) -> Change:
    state.has_line_counter = True
    return change("add line counter", (Path("src/counting/line_counter.py"), render_line_counter_py(state)))


def add_metrics_init(state: State) -> Change:
    state.has_metrics_init = True
    return change("add metrics package", (Path("src/metrics/__init__.py"), render_metrics_init(state)))


def add_fps(state: State) -> Change:
    state.has_fps = True
    return change("add fps meter", (Path("src/metrics/fps.py"), render_fps_py(state)))


def add_pipeline(state: State) -> Change:
    state.has_pipeline = True
    return change("add pipeline skeleton", (Path("src/pipeline.py"), render_pipeline_py(state)))


def add_run(state: State) -> Change:
    state.has_run = True
    state.run_supports_config = True
    return change("add run entrypoint", (Path("run.py"), render_run_py(state)))


def add_docs_pipeline(state: State) -> Change:
    state.has_docs_pipeline = True
    return change("add pipeline doc", (Path("docs/pipeline.md"), render_pipeline_doc(state)))


def add_samples_basic(state: State) -> Change:
    state.has_samples_basic = True
    return change("add sample config", (Path("samples/sample_config.json"), render_sample_config(state)))


def add_config_edge(state: State) -> Change:
    state.has_config_edge = True
    return change("add edge preset", (Path("config/edge.json"), render_config_edge(state)))


def add_config_highres(state: State) -> Change:
    state.has_config_highres = True
    return change("add highres preset", (Path("config/highres.json"), render_config_highres(state)))


def add_samples_line(state: State) -> Change:
    state.has_samples_line = True
    return change("add line sample config", (Path("samples/line_count.json"), render_line_sample(state)))


def add_tests_line(state: State) -> Change:
    state.has_tests_line = True
    return change("add line counter tests", (Path("tests/test_line_counter.py"), render_test_line_counter(state)))


def add_tests_config(state: State) -> Change:
    state.has_tests_config = True
    return change("add config tests", (Path("tests/test_config.py"), render_test_config(state)))


def add_docs_optimization(state: State) -> Change:
    state.has_docs_optimization = True
    return change("add optimization notes", (Path("docs/optimization.md"), render_optimization_doc(state)))


def add_cpp_readme(state: State) -> Change:
    state.has_cpp_readme = True
    return change("add cpp readme", (Path("cpp/README.md"), render_cpp_readme(state)))


def add_cpp_preprocess_h(state: State) -> Change:
    state.has_cpp_preprocess_h = True
    return change("add preprocess header", (Path("cpp/preprocess.h"), render_cpp_preprocess_h(state)))


def add_cpp_preprocess_cpp(state: State) -> Change:
    state.has_cpp_preprocess_cpp = True
    state.cpp_stage = 0
    return change("add preprocess implementation", (Path("cpp/preprocess.cpp"), render_cpp_preprocess_cpp(state)))


def add_cpp_nms_h(state: State) -> Change:
    state.has_cpp_nms_h = True
    return change("add nms header", (Path("cpp/nms.h"), render_cpp_nms_h(state)))


def add_cpp_nms_cpp(state: State) -> Change:
    state.has_cpp_nms_cpp = True
    state.nms_stage = 0
    return change("add nms implementation", (Path("cpp/nms.cpp"), render_cpp_nms_cpp(state)))


def add_cpp_cmake(state: State) -> Change:
    state.has_cpp_cmake = True
    return change("add cpp cmake", (Path("cpp/CMakeLists.txt"), render_cpp_cmake(state)))


def update_root_readme(state: State) -> Change:
    state.has_root_readme = True
    return change("refresh root readme", (Path("README.md"), render_root_readme(state)))


def enable_pipeline_writer(state: State) -> Change:
    if not state.pipeline_writer:
        state.pipeline_writer = True
        return change("add video writer", (Path("src/pipeline.py"), render_pipeline_py(state)))
    state.output_revision += 1
    state.output_path = f"object_detection_v{state.output_revision}.mp4"
    return change("bump output name", (Path("config/default.json"), render_config_default(state)))


def enable_pipeline_preview(state: State) -> Change:
    if not state.pipeline_preview:
        state.pipeline_preview = True
        return change("add preview window", (Path("src/pipeline.py"), render_pipeline_py(state)))
    state.show_window = not state.show_window
    return change("toggle preview flag", (Path("config/default.json"), render_config_default(state)))


def enable_pipeline_line(state: State) -> Change:
    if not state.pipeline_line:
        state.pipeline_line = True
        state.count_mode = "line"
        return change("add line counting", (Path("src/pipeline.py"), render_pipeline_py(state)))
    return tune_line(state)


def enable_pipeline_metrics(state: State) -> Change:
    if not state.pipeline_metrics:
        state.pipeline_metrics = True
        return change("add metrics logging", (Path("src/pipeline.py"), render_pipeline_py(state)))
    return tune_metrics(state)


def enable_pipeline_csv(state: State) -> Change:
    if not state.pipeline_csv:
        state.pipeline_csv = True
        return change("add csv logging", (Path("src/pipeline.py"), render_pipeline_py(state)))
    return tune_csv(state)


def enable_run_preset(state: State) -> Change:
    if not state.run_supports_preset:
        state.run_supports_preset = True
        return change("add preset selection", (Path("run.py"), render_run_py(state)))
    return change("refresh presets", (Path("config/edge.json"), render_config_edge(state)))


def enable_run_video(state: State) -> Change:
    if not state.run_supports_video:
        state.run_supports_video = True
        return change("add video override", (Path("run.py"), render_run_py(state)))
    state.use_stream = not state.use_stream
    return change("toggle stream default", (Path("config/default.json"), render_config_default(state)))


def enable_run_overrides(state: State) -> Change:
    if not state.run_supports_overrides:
        state.run_supports_overrides = True
        return change("add cli overrides", (Path("run.py"), render_run_py(state)))
    return tune_conf(state)


def tune_conf(state: State) -> Change:
    state.conf_threshold = round(state.conf_threshold + 0.05 * state.conf_direction, 2)
    if state.conf_threshold <= 0.35 or state.conf_threshold >= 0.65:
        state.conf_direction *= -1
    return change(f"tune confidence to {state.conf_threshold:.2f}", (Path("config/default.json"), render_config_default(state)))


def tune_skip(state: State) -> Change:
    state.frame_skip += state.skip_direction
    if state.frame_skip <= 2 or state.frame_skip >= 7:
        state.skip_direction *= -1
    return change(f"adjust frame skip to {state.frame_skip}", (Path("config/default.json"), render_config_default(state)))


def tune_resize(state: State) -> Change:
    if state.resize_w == 320:
        state.resize_w = 640
        state.resize_h = 360
    else:
        state.resize_w = 320
        state.resize_h = 180
    return change(f"adjust resize to {state.resize_w}x{state.resize_h}", (Path("config/default.json"), render_config_default(state)))


def tune_line(state: State) -> Change:
    state.line_y += 10 * state.line_direction_delta
    if state.line_y <= 140 or state.line_y >= 220:
        state.line_direction_delta *= -1
    return change(f"nudge line to y={state.line_y}", (Path("config/edge.json"), render_config_edge(state)))


def tune_metrics(state: State) -> Change:
    state.metrics_interval += 5 * state.metrics_direction
    if state.metrics_interval <= 15 or state.metrics_interval >= 60:
        state.metrics_direction *= -1
    return change(f"adjust metrics interval to {state.metrics_interval}", (Path("config/default.json"), render_config_default(state)))


def tune_csv(state: State) -> Change:
    state.csv_interval += 5 * state.csv_direction
    if state.csv_interval <= 15 or state.csv_interval >= 60:
        state.csv_direction *= -1
    return change(f"adjust csv interval to {state.csv_interval}", (Path("config/default.json"), render_config_default(state)))


def update_docs_quickstart(state: State) -> Change:
    return change("refresh quickstart", (Path("docs/quickstart.md"), render_quickstart(state)))


def update_docs_pipeline(state: State) -> Change:
    return change("refresh pipeline doc", (Path("docs/pipeline.md"), render_pipeline_doc(state)))


def update_docs_optimization(state: State) -> Change:
    return change("refresh optimization notes", (Path("docs/optimization.md"), render_optimization_doc(state)))


def update_tests_line(state: State) -> Change:
    return change("refresh line counter tests", (Path("tests/test_line_counter.py"), render_test_line_counter(state)))


def update_samples_basic(state: State) -> Change:
    return change("refresh sample config", (Path("samples/sample_config.json"), render_sample_config(state)))


def update_cpp_preprocess(state: State) -> Change:
    state.cpp_stage = min(state.cpp_stage + 1, 1)
    return change("extend preprocess helper", (Path("cpp/preprocess.cpp"), render_cpp_preprocess_cpp(state)))


def update_cpp_nms(state: State) -> Change:
    state.nms_stage = min(state.nms_stage + 1, 1)
    return change("refine nms helper", (Path("cpp/nms.cpp"), render_cpp_nms_cpp(state)))


INTRO_ACTIONS: List[Callable[[State], Change]] = [
    add_docs_quickstart,
    add_config_default,
    add_src_init,
    add_io_init,
    add_io_source,
    add_tracking_init,
    add_tracker,
    add_counting_init,
    add_line_counter,
    add_metrics_init,
    add_fps,
    add_pipeline,
    add_run,
    add_docs_pipeline,
    add_samples_basic,
    add_config_edge,
    add_config_highres,
    add_samples_line,
    add_tests_line,
    add_tests_config,
    add_docs_optimization,
    add_cpp_readme,
    add_cpp_preprocess_h,
    add_cpp_preprocess_cpp,
    add_cpp_nms_h,
    add_cpp_nms_cpp,
    add_cpp_cmake,
    update_root_readme,
]


TUNE_ACTIONS: List[Callable[[State], Change]] = [
    enable_pipeline_writer,
    enable_pipeline_preview,
    enable_pipeline_line,
    enable_pipeline_metrics,
    enable_pipeline_csv,
    enable_run_preset,
    enable_run_video,
    enable_run_overrides,
    tune_conf,
    tune_skip,
    tune_resize,
    tune_line,
    tune_metrics,
    tune_csv,
    update_docs_quickstart,
    update_docs_pipeline,
    update_docs_optimization,
    update_tests_line,
    update_samples_basic,
    update_cpp_preprocess,
    update_cpp_nms,
]


def select_action(step: Step) -> Callable[[State], Change]:
    if step.index < len(INTRO_ACTIONS):
        return INTRO_ACTIONS[step.index]
    tune_index = (step.index - len(INTRO_ACTIONS)) % len(TUNE_ACTIONS)
    return TUNE_ACTIONS[tune_index]


def apply_step(snapshot_dir: Path, step: Step, state: State) -> str:
    action = select_action(step)
    change_set = action(state)
    for rel_path, content in change_set.writes:
        write_file(snapshot_dir / rel_path, content)
    return change_set.message


def write_message(snapshot_dir: Path, message: str) -> None:
    write_file(snapshot_dir / "message.txt", message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    steps = load_steps()
    if not steps:
        raise SystemExit("No timestamp folders found in history/")

    prev_dir: Path | None = None
    state = State()

    for step in steps:
        snapshot_dir = HISTORY_DIR / step.timestamp
        prepare_snapshot_dir(snapshot_dir, overwrite=args.overwrite)

        if prev_dir is None:
            copy_repo_root(snapshot_dir)
        else:
            copy_snapshot(prev_dir, snapshot_dir)

        message = apply_step(snapshot_dir, step, state)
        write_message(snapshot_dir, message)
        prev_dir = snapshot_dir

    print(f"Generated {len(steps)} snapshots in {HISTORY_DIR}")


if __name__ == "__main__":
    main()
