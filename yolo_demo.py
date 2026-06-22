import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from service.card_configs.card_config import CardConfig

IMAGE_DIR  = "/home/kevin/.cache/kagglehub/datasets/yelbuzz/yugioh-card-images-and-data/versions/3/YuGiOhImages/images"
CONF       = 0.5
YGO_CFG = CardConfig.load("service/card_configs/ygo_card.json")

images = sorted(Path(IMAGE_DIR).glob("*.jpg"))

if not images:
    print(f"No images found in {IMAGE_DIR}")
    exit()

print(f"Found {len(images)} images. [N] Next / [Q] Quit")
cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

for img_path in images:
    img     = cv2.imread(str(img_path))
    h, w, _ = img.shape
    setcode_bb = YGO_CFG.ocr["set_code"]
    cv2.rectangle(
        img, 
        (int(w*setcode_bb[0][0]), int(h*setcode_bb[0][1])), 
        (int(w*setcode_bb[1][0]), int(h*setcode_bb[1][1])), 
        color=(0, 255, 0), 
        thickness=2
    )
    # results = model(str(img_path), conf=CONF)[0]

    # if results.masks is None:
    #     print(f"{img_path.name}: No detections.")
    # else:
    #     for i, (mask, box) in enumerate(zip(results.masks.xy, results.boxes)):
    #         pts     = mask.astype(np.int32)
    #         epsilon = 0.02 * cv2.arcLength(pts.reshape(-1, 1, 2), True)
    #         approx  = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon, True)

    #         print(f"{img_path.name} – Detection {i}: conf={box.conf.item():.4f}  reduced to {len(approx)} points")
    #         print(approx.reshape(-1, 2))

    #         cv2.polylines(img, [approx], True, (0, 255, 0), 2)
    #         for pt in approx.reshape(-1, 2):
    #             cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)

    cv2.imshow("Inference  –  [N] Next  [Q] Quit", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()