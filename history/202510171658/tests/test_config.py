import json
from pathlib import Path

def test_default_config_has_required_keys():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "default.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    required = {"url", "model", "conf", "frame_skip"}
    assert required.issubset(set(data.keys()))
