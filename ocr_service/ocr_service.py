"""
OCR Service
===========
Runs as an HTTP server inside the container.
POST /ocr  with a JPEG/PNG as body → returns recognized text as JSON.

Example:
  curl -X POST http://localhost:8000/ocr --data-binary @card.jpg
"""

import json
import logging
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from paddleocr import PaddleOCR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
logger.info("OCR Service ready.")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress per-request console spam

    def do_POST(self):
        if self.path != "/ocr":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)

        # Decode bytes to numpy array and run OCR
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        result = ocr.ocr(frame, cls=True)
        lines = []
        if result and result[0]:
            for line in result[0]:
                text, confidence = line[1]
                lines.append({"text": text, "confidence": round(confidence, 4)})

        response = json.dumps(lines, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)


if __name__ == "__main__":
    logger.info("Listening on :8000")
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    server.serve_forever()