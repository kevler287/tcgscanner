import cv2
import numpy as np
from service.card_configs.card_config import CardConfig
from service.yolo.card_segmentation import CardSegmentor

ygo_config = CardConfig.load("service/card_configs/ygo_card.json")
card_seg = CardSegmentor(model_path="service/yolo/best.pt", ygo_config=ygo_config)
cap = cv2.VideoCapture("/home/kevin/Downloads/packopening1.mp4")
cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

while True:
    _, img = cap.read()
    warped, sorted_pts = card_seg.segment_and_warp(img)

    if sorted_pts is not None:
        cv2.polylines(img, [sorted_pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=5)

    scale = ygo_config.h / img.shape[0]
    w_scaled = int(img.shape[1] * scale)
    img_resized = cv2.resize(img, (w_scaled, ygo_config.h))

    if warped is None:
        warped = np.zeros((card_seg.ygo_config.h, card_seg.ygo_config.w, 3), dtype=np.uint8)

    combined = np.hstack([img_resized, warped])
    cv2.imshow("Inference  –  [N] Next  [Q] Quit", combined)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()