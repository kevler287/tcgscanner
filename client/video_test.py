import json
import os
import time

import cv2
from pathlib import Path
import numpy as np
import httpx

from client.postprocessing.output_stabilizer import OutputStabilizer
from shared.tcg_config import TCGConfig

SERVICE_URL = "http://localhost:8000"
CONFIG_PATH = os.path.join(Path(__file__).parent.parent, "shared/yugioh.json")

cap = cv2.VideoCapture("http://127.0.0.1:8080/video")
cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

with open(CONFIG_PATH, "r") as f:
    data = json.load(f)
response = httpx.post(
    f"{SERVICE_URL}/configure",
    json=data
)
response.raise_for_status()

stabilizer = OutputStabilizer(tcg_config=TCGConfig.load(CONFIG_PATH))

counter = 0
while True:
    ret, img = cap.read()
    if not ret:
        break

    cv2.imshow("Inference  –  [N] Next  [Q] Quit", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter+=1
    if counter%10 != 0:
        continue

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # if frame_width > frame_height:
    #     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


    # Encode frame and send to service
    start = time.time()
    _, buffer = cv2.imencode(".jpg", img)
    response = httpx.post(
        f"{SERVICE_URL}/scan",
        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    )
    end = time.time()
    # print(f"Inference: {end-start}s")

    if response.status_code == 200:
        data = response.json()
        text = data.get("text")
        pts = data.get("pts")

        # # Draw polylines if card was detected
        # if pts is not None:
        #     pts_array = np.array(pts, dtype=np.int32)
        #     cv2.polylines(img, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

        # Print and draw text if OCR returned result
        if text is not None:
            result = stabilizer.forward(ocr_output=text)
            if result:
                print(result)
    else:
        print(response.status_code)

cap.release()
cv2.destroyAllWindows()