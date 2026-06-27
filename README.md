# TCG Scanner

TCG Scanner is an end-to-end **computer vision** pipeline for **real-time** trading card detection and OCR, built with production-grade **MLOps/Data Engineering** practices.

A synthetic training dataset is generated programmatically from 10,000+ high-resolution card images using randomized perspective transforms, glare simulation, and augmentation — eliminating the need for manual labeling. The pipeline trains a YOLO segmentation model to detect cards under real-world conditions (partial occlusion, motion blur, varying lighting) and applies inverse perspective warping for precise card extraction.

The data platform is built on **Google Cloud**, using GCS as a data lake for raw assets and model artifacts, BigQuery for experiment tracking and training metrics, and Prefect for pipeline orchestration. The service runs in a Dockerized microservice with GPU acceleration via CUDA, exposed via a REST API.

Tech Stack: **Python** · **PyTorch** · **YOLO** · **OpenCV** · **PaddleOCR** · **Docker** · **Google Cloud Storage** · **BigQuery** · **Prefect** · **CUDA** · **ONNX Runtime**

## Requirements

- Docker + Docker Compose
- NVIDIA GPU + nvidia-container-toolkit

```bash
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Usage

```bash
docker compose up -d
```

The first build takes a few minutes — PaddleOCR models are downloaded and baked into the image. Subsequent starts are fast.

```bash
docker compose logs -f   # wait for "OCR Service ready."
docker compose down
```
