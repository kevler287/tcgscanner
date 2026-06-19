import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/segment/seg_v2-2/weights/best.pt"
CONF       = 0.5

model  = YOLO(MODEL_PATH)
cap = cv2.VideoCapture("/home/kevin/Downloads/test2.mp4")
cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

while True:
    _, img = cap.read()
    print(img.shape)
    results = model(img, conf=CONF)[0]

    if results.masks is None:
        cv2.putText(img, "No detections", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    else:
        for i, (mask, box) in enumerate(zip(results.masks.xy, results.boxes)):
            pts     = mask.astype(np.int32)
            epsilon = 0.02 * cv2.arcLength(pts.reshape(-1, 1, 2), True)
            approx  = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon, True)

            cv2.polylines(img, [approx], True, (0, 255, 0), 2)
            for pt in approx.reshape(-1, 2):
                cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)

    cv2.imshow("Inference  –  [N] Next  [Q] Quit", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()