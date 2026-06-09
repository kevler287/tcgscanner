FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    paddlepaddle-gpu==2.6.2 \
    paddleocr==2.7.3 \
    opencv-python-headless \
    "numpy<2.0" \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Pre-download PaddleOCR models during build so they're cached in the image
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)"

WORKDIR /app
COPY card_scanner.py .

CMD ["python3", "card_scanner.py"]