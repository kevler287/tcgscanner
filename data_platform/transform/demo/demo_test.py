from pathlib import Path
import cv2
import numpy as np
import random

# ── Config ────────────────────────────────────────────────────────────────────

IMAGE_PATH = "data_platform/transform/demo_card.jpg"
MAX_ANGLE_DEG = 10
BACKGROUND_SIZE = (600, 800)
BACKGROUND_DIR = "/home/kevin/.cache/kagglehub/datasets/haaroonafroz/JPEGImages"

# Random resize scale range (1.0 = original size)
MIN_SCALE = 1.5
MAX_SCALE = 3

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
    bg = cv2.resize(bg, (int(w * scale), int(h * scale)))
    h, w = bg.shape[:2]
    x = random.randint(0, w - target_w)
    y = random.randint(0, h - target_h)
    return bg[y:y+target_h, x:x+target_w]


def random_perspective_warp(image, max_angle_deg=10):
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

def add_glare(image, corners):
    """
    Simulates sleeve glare with a random gradient highlight over the card polygon.
    """
    h, w = image.shape[:2]

    # Random glare direction and intensity
    angle = random.uniform(0, 2 * np.pi)
    intensity = random.uniform(0.1, 1)

    # Vectorized gradient
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(xs, ys)
    glare = np.cos(angle) * xv + np.sin(angle) * yv

    # Normalize to [0, intensity * 255]
    glare = (glare - glare.min()) / (glare.max() - glare.min())
    glare = (glare * intensity * 255).astype(np.uint8)

    # Apply only inside card polygon
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
    glare = cv2.bitwise_and(glare, glare, mask=mask)

    # Blend onto all channels
    result = image.copy()
    for c in range(3):
        result[:, :, c] = np.clip(
            result[:, :, c].astype(np.int32) + glare, 0, 255
        ).astype(np.uint8)

    return result

def add_blur(image):
    """Apply random gaussian blur to the entire image."""
    # Kernel size must be odd
    k = random.choice([1, 3, 5, 7])
    return cv2.GaussianBlur(image, (k, k), 0)

def paste_on_random_background(card, corners, bg_size=(600, 800)):
    bg = load_random_background(bg_size)
    bg_w, bg_h = bg_size
    h, w = card.shape[:2]

    # Randomly choose card position with visible setcode (right side center)
    x = random.randint(-int(w/3), bg_w - int(w*0.9))
    y = random.randint(-int(h/2), bg_h - int(h*0.8))
    # print(f"x bounds: ({-int(w/2)}|{bg_w - w}), w={w}, bg_w={bg_w}")
    # print(f"y bounds: ({-int(h/2)}|{bg_h - int(h*0.8)}), h={h}, bg_h={bg_h}")
    # print(f"({x}|{y})")

    # Build polygon mask for the card
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = corners.astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Compute the visible (clipped) region
    x1_card = max(0, -x)
    y1_card = max(0, -y)
    x2_card = min(w, bg_w - x)
    y2_card = min(h, bg_h - y)

    x1_bg = max(0, x)
    y1_bg = max(0, y)
    x2_bg = x1_bg + (x2_card - x1_card)
    y2_bg = y1_bg + (y2_card - y1_card)

    if x2_card <= x1_card or y2_card <= y1_card:
        return bg, (x, y)

    # Crop to visible region
    card_crop = card[y1_card:y2_card, x1_card:x2_card]
    mask_crop = mask[y1_card:y2_card, x1_card:x2_card]
    mask_inv = cv2.bitwise_not(mask_crop)

    roi = bg[y1_bg:y2_bg, x1_bg:x2_bg]
    bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
    card_part = cv2.bitwise_and(card_crop, card_crop, mask=mask_crop)
    bg[y1_bg:y2_bg, x1_bg:x2_bg] = cv2.add(bg_part, card_part)

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
        scale = random.uniform(MIN_SCALE, MAX_SCALE)
        card_resized = cv2.resize(card, (int(card.shape[1] * scale), int(card.shape[0] * scale)))
        warped, corners = random_perspective_warp(card_resized, MAX_ANGLE_DEG)
        warped = add_glare(warped, corners)
        warped = add_blur(warped)
        composite, (ox, oy) = paste_on_random_background(warped, corners, BACKGROUND_SIZE)
        composite = add_blur(composite)
        cv2.imshow("Transform Test  –  [Q] Quit", composite)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()