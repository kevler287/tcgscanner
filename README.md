# TCG Scanner

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
