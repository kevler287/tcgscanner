# TCG Scanner

> [!NOTE]
> Work in progress

TCG Scanner is an end-to-end **computer vision** pipeline for **real-time** trading card detection and OCR, built with production-grade **MLOps/Data Engineering** practices.

A YOLO segmentation model localizes cards in the camera feed, applies inverse **perspective warping** for precise extraction, and feeds the result into a GPU-accelerated **PaddleOCR** service for text recognition — all running in a **Dockerized** microservice with a REST API.

**Tech Stack:** Python · PyTorch · YOLOv11 · OpenCV · PaddleOCR · Docker · Google Cloud Storage · BigQuery · CUDA · ONNX Runtime

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
python client/video_test.py
```

The first build takes a few minutes — PaddleOCR models are downloaded and baked into the image. Subsequent starts are fast.

```bash
docker compose logs -f   # wait for "OCR Service ready."
docker compose down
```


## Related

- [tcgscanner-ml](https://github.com/kevler287/tcgscanner-ml) — Training pipeline, synthetic data generation, and model weights