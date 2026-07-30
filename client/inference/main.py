import time

import cv2
import httpx

from client.inference.yugioh.stabilizer import YugiohStabilizer
from client.inference.yugioh.csv_builder import YugiohCSVBuilder
from shared.tcg_config import TCGConfig

SERVICE_URL = "http://localhost:8000"

cap = cv2.VideoCapture("http://127.0.0.1:8080/video")
cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

yugioh_config = TCGConfig.load("shared/yugioh.json")
response = httpx.post(
    f"{SERVICE_URL}/configure",
    json=yugioh_config.to_json()
)
response.raise_for_status()

response = httpx.get(f"{SERVICE_URL}/toggle-debug")
response.raise_for_status()

stabilizer = YugiohStabilizer()
csv_builder = YugiohCSVBuilder()

collected = []
ts = None

counter = 0
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Inference  –  [N] Next  [Q] Quit", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter+=1
    if counter%10 != 0:
        continue

    # when timer is set (=card was detected) wait 1 sec for new card before scanning
    if ts is not None and (time.time() - ts) < 1:
        continue

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
        editions = data.get("editions", {})

        if text is not None:
            card_scanned, progress = stabilizer.forward(ocr_output=text, edition_dets=editions)
            print("-----------------------------------------")
            for field, progress in progress.items():
                print(f"{field}: {progress:.0%}")

            # if stable result is present: store result, clear stabilizer and set timer
            if card_scanned:
                print("########## CARD REGISTERED ##########")
                collected.append(progress)
                stabilizer.clear()
                ts = time.time()
    else:
        print(response.status_code)

cap.release()
cv2.destroyAllWindows()

csv_builder.detections_to_csv(ygo_dets=collected)