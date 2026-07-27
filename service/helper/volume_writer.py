from datetime import datetime
import os
import json
from pathlib import Path
import cv2
import numpy as np

_DEBUG_DIR = Path(os.environ["DEBUG_DIR"])

def new_trace_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return str(timestamp)


def _unit_dir(trace_id: str) -> Path:
    d = _DEBUG_DIR / trace_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_frame(trace_id: str, stage_name: str, frame: np.ndarray) -> None:
    path = _unit_dir(trace_id) / f"{stage_name}.jpg"
    cv2.imwrite(str(path), frame)


def save_crops(trace_id: str, stage_name: str, crops: list[np.ndarray]) -> None:
    for i, crop in enumerate(crops):
        save_frame(trace_id, f"{stage_name}_{i:02d}", crop)


def save_json(trace_id: str, stage_name: str, data: dict) -> None:
    path = _unit_dir(trace_id) / f"{stage_name}.json"
    path.write_text(json.dumps(data, indent=2, default=str))