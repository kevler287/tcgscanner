import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/segment/seg_v2-2/weights/best.pt"
IMAGE_DIR  = "data_platform/transform/output/images/test"
CONF       = 0.5

model  = YOLO(MODEL_PATH)
images = sorted(Path(IMAGE_DIR).glob("*.jpg"))

if not images:
    print(f"No images found in {IMAGE_DIR}")
    exit()

print(f"Found {len(images)} images. [N] Next / [Q] Quit")
cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

for img_path in images:
    img     = cv2.imread(str(img_path))
    results = model(str(img_path), conf=CONF)[0]

    if results.masks is None:
        print(f"{img_path.name}: No detections.")
    else:
        for i, (mask, box) in enumerate(zip(results.masks.xy, results.boxes)):
            pts     = mask.astype(np.int32)
            epsilon = 0.02 * cv2.arcLength(pts.reshape(-1, 1, 2), True)
            approx  = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon, True)

            print(f"{img_path.name} – Detection {i}: conf={box.conf.item():.4f}  reduced to {len(approx)} points")
            print(approx.reshape(-1, 2))

            cv2.polylines(img, [approx], True, (0, 255, 0), 2)
            for pt in approx.reshape(-1, 2):
                cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)

    cv2.imshow("Inference  –  [N] Next  [Q] Quit", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()