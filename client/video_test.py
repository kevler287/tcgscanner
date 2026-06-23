import json
import os

import cv2
from pathlib import Path
import numpy as np
import httpx

SERVICE_URL = "http://localhost:8000"
CONFIG_PATH = os.path.join(Path(__file__).parent.parent, "shared/ygo_config.json")

cap = cv2.VideoCapture("/home/kevin/Downloads/test3.mp4")
cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

with open(CONFIG_PATH, "r") as f:
    data = json.load(f)
response = httpx.post(
    f"{SERVICE_URL}/configure",
    json=data
)
response.raise_for_status()

while True:
    ret, img = cap.read()
    if not ret:
        break

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width > frame_height:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


    # Encode frame and send to service
    _, buffer = cv2.imencode(".jpg", img)
    response = httpx.post(
        f"{SERVICE_URL}/scan",
        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    )

    if response.status_code == 200:
        data = response.json()
        text = data.get("text")
        pts = data.get("pts")

        # Draw polylines if card was detected
        if pts is not None:
            pts_array = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

        # Print and draw text if OCR returned result
        if text is not None:
            print(f"Card: {text}")
            cv2.putText(
                img, str(text),
                org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.2,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )
    else:
        print(response.status_code)

    cv2.imshow("Inference  –  [N] Next  [Q] Quit", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()