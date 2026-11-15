"""Downloads a JSON file from a fixed URL and converts it to CSV."""
import csv
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://downloads.s3.cardmarket.com/productCatalog/priceGuide/price_guide_3.json"
DATA_DIR = os.environ.get("DATA_DIR", "/data")
JSON_PATH = Path(DATA_DIR) / "price_catalog.json"
CSV_PATH = Path(DATA_DIR) / "price_catalog.csv"
DELIMITER = ","


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        target.write_bytes(resp.read())


def _flatten(obj, parent="", sep="."):
    """Flatten nested dicts into flat key-value pairs (a.b.c)."""
    items = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{parent}{sep}{key}" if parent else str(key)
            items.update(_flatten(value, new_key, sep))
    elif isinstance(obj, list):
        items[parent] = json.dumps(obj, ensure_ascii=False)
    else:
        items[parent] = obj
    return items


def _to_rows(data):
    """Find the list of records in the JSON structure."""
    if isinstance(data, list):
        return [_flatten(r) if isinstance(r, dict) else {"value": r} for r in data]
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if len(lists) == 1:
            return [_flatten(r) for r in lists[0]]
        return [_flatten(data)]
    return [{"value": data}]


def _write_csv(rows, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with target.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(rows)


def run_job() -> int:
    """Download -> read JSON -> write CSV -> delete JSON. Returns the number of rows."""
    _download(URL, JSON_PATH)
    try:
        with JSON_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        rows = _to_rows(data)
        _write_csv(rows, CSV_PATH)
    finally:
        # Always delete the JSON file, even on errors
        JSON_PATH.unlink(missing_ok=True)
    price_df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    return price_df