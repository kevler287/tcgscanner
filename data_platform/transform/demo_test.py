from pathlib import Path

import cv2
import numpy as np
import random

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_PATH = "data_platform/transform/demo_card.jpg"   # <-- path to a single card image
MAX_ANGLE_DEG = 10               # maximum perspective warp angle
BACKGROUND_SIZE = (800, 600)     # output canvas size
BACKGROUND_DIR = "/home/kevin/.cache/kagglehub/datasets/haaroonafroz/JPEGImages"

# ── Transform ─────────────────────────────────────────────────────────────────

def load_random_background(bg_size):
    files = []

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(Path(BACKGROUND_DIR).rglob(ext))

    bg_path = random.choice(files)

    bg = cv2.imread(str(bg_path))

    h, w = bg.shape[:2]
    target_w, target_h = bg_size

    scale = max(target_w / w, target_h / h)

    bg = cv2.resize(
        bg,
        (int(w * scale), int(h * scale))
    )

    h, w = bg.shape[:2]

    x = random.randint(0, w - target_w)
    y = random.randint(0, h - target_h)

    bg = bg[y:y+target_h, x:x+target_w]

    return bg

def random_perspective_warp(image, max_angle_deg=10):
    """
    Apply a random perspective warp to an image.
    Returns the warped image and the 4 corner points (for YOLO labels).
    """
    h, w = image.shape[:2]

    max_offset = int(min(h, w) * np.tan(np.radians(max_angle_deg)))

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    dst = np.float32([
        [random.randint(0, max_offset), random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), h - random.randint(0, max_offset)],
        [random.randint(0, max_offset), h - random.randint(0, max_offset)],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (w, h))

    return warped, dst


def paste_on_random_background(card, corners, bg_size=(800, 600)):
    """
    Paste warped card onto a random real background image.
    """

    bg = load_random_background(bg_size)

    h, w = card.shape[:2]

    x = random.randint(0, max(0, bg_size[0] - w))
    y = random.randint(0, max(0, bg_size[1] - h))

    mask = np.zeros((h, w), dtype=np.uint8)

    pts = corners.astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    mask_inv = cv2.bitwise_not(mask)

    roi = bg[y:y+h, x:x+w]

    bg_part = cv2.bitwise_and(
        roi,
        roi,
        mask=mask_inv
    )

    card_part = cv2.bitwise_and(
        card,
        card,
        mask=mask
    )

    bg[y:y+h, x:x+w] = cv2.add(
        bg_part,
        card_part
    )

    return bg, (x, y)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    card = cv2.imread(IMAGE_PATH)
    if card is None:
        print(f"Could not load image: {IMAGE_PATH}")
        return

    print("Press any key to generate a new sample, [Q] to quit.")

    cv2.namedWindow("Transform Test  –  [Q] Quit", cv2.WINDOW_NORMAL)

    while True:
        warped, corners = random_perspective_warp(card, MAX_ANGLE_DEG)
        composite, (ox, oy) = paste_on_random_background(warped, corners, BACKGROUND_SIZE)

        cv2.imshow("Transform Test  –  [Q] Quit", composite)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()