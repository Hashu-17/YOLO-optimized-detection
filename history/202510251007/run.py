import argparse
import json
from pathlib import Path

from src.pipeline import run_pipeline

def load_json(path):
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--preset", choices=["edge", "highres"])
    parser.add_argument("--url")
    parser.add_argument("--video")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--skip", type=int)
    parser.add_argument("--resize", nargs=2, type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_json("config/default.json")
    if args.preset:
        config.update(load_json(f"config/{args.preset}.json"))
    if args.config and args.config != "config/default.json":
        config.update(load_json(args.config))
    if args.url:
        config["url"] = args.url
        config["use_stream"] = True
    if args.video:
        config["video_file"] = args.video
        config["use_stream"] = False
    if args.conf is not None:
        config["conf"] = max(0.1, min(args.conf, 0.95))
    if args.skip:
        config["frame_skip"] = max(1, args.skip)
    if args.resize:
        config["resize"] = [max(1, args.resize[0]), max(1, args.resize[1])]
    if args.output:
        config["output_path"] = args.output

    run_pipeline(config)

if __name__ == "__main__":
    main()
