import cv2
from paddleocr import PaddleOCR

OCR_EVERY_N_FRAMES = 5  # <-- anpassen
BATCH_SIZE = 4          # <-- anpassen

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

cap = cv2.VideoCapture("http://localhost:8080/video")
frame_count = 0
batch = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % OCR_EVERY_N_FRAMES == 0:
        batch.append(frame)

        if len(batch) >= BATCH_SIZE:
            results = ocr.ocr(batch, cls=True, det=False)
            for result in results:
                if result:
                    for line in result:
                        print(line[1])
            batch = []

    frame_count += 1

    cv2.imshow("Phone Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()