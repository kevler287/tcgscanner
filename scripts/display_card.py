import json
import cv2

CONFIG_PATH = "shared/yugioh.json"
IMAGE_PATH = "scripts/test_card.jpeg"

# ---------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

ocr_fields = config["ocr_fields"]

# ---------------------------------------------------------------------
# Load image
# ---------------------------------------------------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Failed to load image: {IMAGE_PATH}")

height, width = image.shape[:2]

# ---------------------------------------------------------------------
# Draw OCR regions
# ---------------------------------------------------------------------
for field_name, field in ocr_fields.items():
    (x1_rel, y1_rel), (x2_rel, y2_rel) = field["position"]

    x1 = int(x1_rel * width)
    y1 = int(y1_rel * height)
    x2 = int(x2_rel * width)
    y2 = int(y2_rel * height)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw field name
    cv2.putText(
        image,
        field_name,
        (x1, max(10, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
    )

# ---------------------------------------------------------------------
# Display image
# ---------------------------------------------------------------------
cv2.namedWindow("OCR Regions", cv2.WINDOW_NORMAL)
cv2.imshow("OCR Regions", image)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q")):  # ESC or Q
        break

cv2.destroyAllWindows()