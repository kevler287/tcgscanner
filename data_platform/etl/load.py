import os
import logging
import argparse
import yaml
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import storage, bigquery
from data_platform.config import CONFIG

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_YAML         = os.getenv("DATA_YAML", "data_platform/etl/output/data.yaml")
YOLO_RUNS_DIR     = os.getenv("YOLO_RUNS_DIR")


def get_classes() -> list:
    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    return list(data["names"].values())


def upload_model(storage_client, run_dir: Path, model_version: str):
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        logger.error("best.pt not found at %s", weights_path)
        return

    blob_name = f"{CONFIG.bucket.seg_models_prefix}{model_version}/best.pt"
    bucket    = storage_client.bucket(CONFIG.bucket.name)
    blob      = bucket.blob(blob_name)
    blob.upload_from_filename(str(weights_path))
    logger.info("Uploaded best.pt → gs://%s/%s", CONFIG.bucket.name, blob_name)


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



def load(model_version: str, run_dir: Path = None):
    run_dir = run_dir or get_latest_run_dir()
    classes = get_classes()

    storage_client = storage.Client()
    bq_client      = bigquery.Client()

    upload_model(storage_client, run_dir, model_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True, help="Model version, e.g. v1")
    parser.add_argument("--run-dir",       required=True)
    args = parser.parse_args()

    load(
        model_version = args.model_version,
        run_dir       = Path(args.run_dir) if args.run_dir else None,
    )