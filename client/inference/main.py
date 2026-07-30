import argparse
import time

import cv2
import httpx
import numpy as np

from client.inference.yugioh.stabilizer import YugiohStabilizer
from client.inference.yugioh.csv_builder import YugiohCSVBuilder
from shared.tcg_config import TCGConfig

SERVICE_URL = "http://localhost:8000"
PANEL_WIDTH = 300
BAR_HEIGHT = 30
BAR_PADDING = 15
FONT = cv2.FONT_HERSHEY_SIMPLEX

stabilizer = YugiohStabilizer()
collected = []
ts = None

def parse_args():
    parser = argparse.ArgumentParser(description="Yugioh card scanner CLI")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--frame-skip", type=int, default=10, help="process every Nth frame")
    return parser.parse_args()

def build_progress_panel(progress: dict, card_scanned: bool, panel_height: int) -> np.ndarray:
    if card_scanned:
        panel = np.full((panel_height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

        center_y = panel_height // 2

        text = "SUCCESS"
        text_size = cv2.getTextSize(text, FONT, 0.8, 2)[0]
        text_x = (PANEL_WIDTH - text_size[0]) // 2
        cv2.putText(panel, text, (text_x, center_y + 20), FONT, 0.8, (0, 200, 0), 2, cv2.LINE_AA)

        return panel
    else:
        panel = np.full((panel_height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

        y = BAR_PADDING
        for field, (value, fraction) in progress.items():
            label = f"{field}: {value}"
            cv2.putText(panel, label, (10, y + 15), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            bar_y = y + 25
            cv2.rectangle(panel, (10, bar_y), (PANEL_WIDTH - 10, bar_y + 10), (80, 80, 80), -1)

            fill_width = int((PANEL_WIDTH - 20) * min(max(fraction, 0), 1))
            color = (0, 200, 0) if fraction >= 1.0 else (0, 165, 255)
            cv2.rectangle(panel, (10, bar_y), (10 + fill_width, bar_y + 10), color, -1)

            y += BAR_HEIGHT + BAR_PADDING

        return panel

def build_live_ui(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    # match panel height to frame height
    if panel.shape[0] != frame.shape[0]:
        resized_panel = np.full((frame.shape[0], PANEL_WIDTH, 3), 30, dtype=np.uint8)
        resized_panel[:panel.shape[0], :] = panel
        panel = resized_panel

    return np.hstack([frame, panel])

def process_frame(frame: np.ndarray, debug: bool):
    # when timer is set (=card was detected) wait 1 sec for new card before scanning
    if ts is not None and (time.time() - ts) < 1:
        return None

    # Encode frame and send to service
    start = time.time()
    _, buffer = cv2.imencode(".jpg", frame)
    response = httpx.post(
        f"{SERVICE_URL}/scan",
        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    )
    end = time.time()
    if debug:
        print(f"Inference: {end-start}s")

    if response.status_code == 200:
        data = response.json()
        text = data.get("text", {})
        editions = data.get("editions", {})

        card_scanned, progress = stabilizer.forward(ocr_output=text, edition_dets=editions)
        if card_scanned:
            collected.append(progress)
            stabilizer.clear()
            ts = time.time()

        progress_panel = build_progress_panel(progress, card_scanned, panel_height=img.shape[0])
        return progress_panel
    else:
        print(response.status_code)
        return None

def run_capture_loop(debug: bool = False, frame_skip: int = 10) -> list:
    cap = cv2.VideoCapture("http://127.0.0.1:8080/video")
    cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

    progress_panel = build_progress_panel({}, card_scanned=False, panel_height=480)  # placeholder, adjust height

    counter = 0
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            if counter % frame_skip == 0:
                progress_panel = process_frame(frame=img, debug=debug)

            display = build_live_ui(img, progress_panel)
            cv2.imshow("Inference  –  [N] Next  [Q] Quit", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            counter += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    yugioh_config = TCGConfig.load("shared/yugioh.json")
    response = httpx.post(
        f"{SERVICE_URL}/configure",
        json=yugioh_config.to_json()
    )
    response.raise_for_status()

    if args.debug:
        response = httpx.get(f"{SERVICE_URL}/toggle-debug")
        response.raise_for_status()

    csv_builder = YugiohCSVBuilder()

    run_capture_loop(debug=args.debug, frame_skip=args.frame_skip)

    csv_builder.detections_to_csv(ygo_dets=collected)


if __name__ == "__main__":
    main()