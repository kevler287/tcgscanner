from datetime import datetime
import os
import json
from pathlib import Path
import cv2
import numpy as np

_DATA_DIR = Path(os.environ["DATA_DIR"])

def get_date_str() -> str:
    timestamp = datetime.now().strftime("%Y_%m_%d")
    return str(timestamp)

def _unit_dir(trace_id: str) -> Path:
    d = _DATA_DIR / trace_id
    image_dir = d / "images"
    label_dir = d / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir

def collect(crop: np.ndarray, results: dict) -> None:
    image_dir, label_dir = _unit_dir(get_date_str())
    index = len(list(image_dir.iterdir()))

    im_path = image_dir / f"{index}.jpg"
    cv2.imwrite(str(im_path), crop)

    lb_path = label_dir / f"{index}.json"
    lb_path.write_text(json.dumps(results, indent=2, default=str))
