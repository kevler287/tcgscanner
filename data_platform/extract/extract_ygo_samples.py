import os
import logging
from pathlib import Path

import kagglehub
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("GCS_BUCKET")
RAW_PREFIX = os.getenv("GCS_RAW_PREFIX")
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET")

def download_dataset() -> Path:
    logger.info("Downloading dataset from Kaggle: %s", KAGGLE_DATASET)

    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)

    logger.info("Dataset available at: %s", dataset_path)

    return Path(dataset_path)


def upload_images(dataset_dir: Path):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    images = (
        list(dataset_dir.rglob("*.jpg"))
        + list(dataset_dir.rglob("*.jpeg"))
        + list(dataset_dir.rglob("*.png"))
    )

    if not images:
        logger.warning("No images found in %s", dataset_dir)
        return

    logger.info("Found %d images, starting upload...", len(images))

    for i, path in enumerate(images, 1):
        blob_name = f"{RAW_PREFIX}{path.name}"
        blob = bucket.blob(blob_name)

        if blob.exists():
            logger.info(
                "[%d/%d] Skipping %s (already exists)",
                i,
                len(images),
                path.name,
            )
            continue

        blob.upload_from_filename(str(path))

        logger.info(
            "[%d/%d] Uploaded %s",
            i,
            len(images),
            path.name,
        )

    logger.info("Upload complete.")


if __name__ == "__main__":
    dataset_dir = download_dataset()
    upload_images(dataset_dir)