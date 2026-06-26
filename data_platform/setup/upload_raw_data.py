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
GCS_CARDS_PREFIX = os.getenv("GCS_CARDS_PREFIX")
GCS_BG_PREFIX    = os.getenv("GCS_BG_PREFIX")

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

def download_dataset(source_path: str) -> Path:
    logger.info("Downloading dataset from Kaggle: %s", source_path)

    local_path = kagglehub.dataset_download(source_path)

    logger.info("Dataset available at: %s", local_path)

    return Path(local_path)


def upload_images(dataset_dir: Path, bucket_prefix: str):

    images = (
        list(dataset_dir.rglob("*.jpg"))
        + list(dataset_dir.rglob("*.jpeg"))
    )

    if not images:
        logger.warning("No images found in %s", dataset_dir)
        return

    logger.info("Found %d images, starting upload...", len(images))

    for i, path in enumerate(images, 1):
        blob_name = f"{bucket_prefix}{path.name}"
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

    # ygo card templates
    dataset_dir = download_dataset(source_path="yelbuzz/yugioh-card-images-and-data")
    upload_images(dataset_dir, bucket_prefix=GCS_CARDS_PREFIX)

    # background images
    dataset_dir = download_dataset(source_path="haaroonafroz/material-dataset-new")
    upload_images(dataset_dir, bucket_prefix=GCS_BG_PREFIX)