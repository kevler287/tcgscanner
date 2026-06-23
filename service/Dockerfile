# devel version needed for cudnn shared libraries
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Paddle separately due to custom index URL
RUN pip install --no-cache-dir \
    paddlepaddle-gpu==2.6.2 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Install remaining dependencies via requirements.txt
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache PaddleOCR models at build time
RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)"

# Cache YOLO dependencies at build time
# (ultralytics downloads additional dependencies on first YOLO() call)
RUN python3 -c "from ultralytics import YOLO"

WORKDIR /app
COPY ../shared ./shared
COPY ./service .

HEALTHCHECK --interval=5s --timeout=3s --start-period=60s \
    CMD curl -f http://localhost:8000/health/ready || exit 

CMD ["uvicorn", "card_scanner:app", "--host", "0.0.0.0", "--port", "8000"]