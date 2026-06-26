import os
import logging
import argparse
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import storage, bigquery

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BUCKET_NAME       = os.getenv("GCS_BUCKET")
GCS_MODELS_PREFIX = os.getenv("GCS_MODELS_PREFIX")
BQ_DATASET        = os.getenv("BQ_DATASET")
BQ_TABLE          = os.getenv("BQ_TABLE", "training_runs")
DATA_YAML         = os.getenv("DATA_YAML", "data_platform/transform/output/data.yaml")
YOLO_RUNS_DIR     = os.getenv("YOLO_RUNS_DIR", "runs/segment")


def get_latest_run_dir() -> Path:
    runs = [p for p in Path(YOLO_RUNS_DIR).iterdir() if p.is_dir()]
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    logger.info("Latest run dir: %s", latest)
    return latest


def get_classes() -> list:
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    return list(data["names"].values())


def upload_model(storage_client, run_dir: Path, model_version: str):
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        logger.error("best.pt not found at %s", weights_path)
        return

    blob_name = f"{GCS_MODELS_PREFIX}{model_version}/best.pt"
    bucket    = storage_client.bucket(BUCKET_NAME)
    blob      = bucket.blob(blob_name)
    blob.upload_from_filename(str(weights_path))
    logger.info("Uploaded best.pt → gs://%s/%s", BUCKET_NAME, blob_name)


def parse_results(run_dir: Path):
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        logger.error("results.csv not found at %s", results_path)
        return None

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    map50_col = [c for c in df.columns if "mAP50" in c and "95" not in c][0]
    best_row  = df.loc[df[map50_col].idxmax()]

    logger.info("Best epoch: %d  mAP50: %.4f", int(best_row["epoch"]), best_row[map50_col])
    return best_row


def parse_train_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        logger.warning("args.yaml not found, hyperparams will be None.")
        return {}

    with open(args_path) as f:
        return yaml.safe_load(f)


def upload_to_bigquery(bq_client, best_row, train_args: dict, model_version: str, classes: list):
    table_id = f"{bq_client.project}.{BQ_DATASET}.{BQ_TABLE}"

    row = {
        "model_version": model_version,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "classes":       classes,
        "epochs":        train_args.get("epochs"),
        "batch_size":    train_args.get("batch"),
        "learning_rate": train_args.get("lr0"),
        "optimizer":     train_args.get("optimizer"),
        "amp":           train_args.get("amp", False),
        "best_epoch":    int(best_row["epoch"]),
        "box_loss":      float(best_row.get("train/box_loss", 0)),
        "seg_loss":      float(best_row.get("train/seg_loss", 0)),
        "cls_loss":      float(best_row.get("train/cls_loss", 0)),
        "dfl_loss":      float(best_row.get("train/dfl_loss", 0)),
        "precision":     float(best_row.get("metrics/precision(B)", 0)),
        "recall":        float(best_row.get("metrics/recall(B)", 0)),
        "map50":         float(best_row.get("metrics/mAP50(B)", 0)),
        "map50_95":      float(best_row.get("metrics/mAP50-95(B)", 0)),
    }

    errors = bq_client.insert_rows_json(table_id, [row])
    if errors:
        logger.error("BigQuery insert errors: %s", errors)
    else:
        logger.info("Inserted run into BigQuery: %s.%s", BQ_DATASET, BQ_TABLE)


def load(model_version: str, run_dir: Path = None):
    run_dir = run_dir or get_latest_run_dir()
    classes = get_classes()

    storage_client = storage.Client()
    bq_client      = bigquery.Client()

    upload_model(storage_client, run_dir, model_version)

    best_row   = parse_results(run_dir)
    train_args = parse_train_args(run_dir)

    if best_row is not None:
        upload_to_bigquery(bq_client, best_row, train_args, model_version, classes)

    logger.info("Load complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True, help="Model version, e.g. v1")
    parser.add_argument("--run-dir",       default=None,  help="YOLO run dir (optional, defaults to latest)")
    args = parser.parse_args()

    load(
        model_version = args.model_version,
        run_dir       = Path(args.run_dir) if args.run_dir else None,
    )