"""
Local Test Client
=================
Sends all JPGs from a directory to the OCR service and prints the results.
"""

import sys
import glob
import requests

IMAGE_DIR = "/home/kevin/datasets/ygo"
SERVICE_URL = "http://localhost:8000/ocr"

files = sorted(glob.glob(f"{IMAGE_DIR}/*.jpg") + glob.glob(f"{IMAGE_DIR}/*.jpeg"))

if not files:
    print(f"No JPG files found in {IMAGE_DIR}.")
    sys.exit(1)

for path in files:
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print('='*60)

    with open(path, "rb") as f:
        response = requests.post(SERVICE_URL, data=f.read())

    lines = response.json()
    if lines:
        for line in lines:
            print(f"  [{line['confidence']:.2f}] {line['text']}")
    else:
        print("  No text detected.")