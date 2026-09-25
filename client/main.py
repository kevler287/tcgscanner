import argparse
import json
import time

import cv2
import httpx
import imageio
import numpy as np
import pandas as pd

from client.common.detectionstate_enum import DetectionState
from client.yugioh.stabilizer import YugiohStabilizer
from client.yugioh.csv_builder import YugiohCSVBuilder
from shared.tcg_layout_model import TCGLayoutConfig

INFERENCE_SRV_URL = "http://localhost:8000"
YGO_SRV_URL = "http://localhost:8001"
PANEL_WIDTH = 900
BAR_HEIGHT = 60
BAR_PADDING = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX

stabilizer = YugiohStabilizer()
csv_builder = YugiohCSVBuilder()
config = TCGLayoutConfig.load("shared/yugioh_layout.json")
ts = None

def parse_args():
    parser = argparse.ArgumentParser(description="Yugioh card scanner CLI")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--record", action="store_true", help="enable stream recording")
    parser.add_argument("--frame-skip", type=int, default=10, help="process every Nth frame")
    parser.add_argument("--condition", required=True, type=str, help="default condition set for all scanned cards")
    return parser.parse_args()

def build_progress_panel(product: dict, progress: dict, status: DetectionState, panel_height: int) -> np.ndarray:
    if status in [DetectionState.IDENTIFIED, DetectionState.AMBIGUOUS, DetectionState.ERRONEOUS]:
        panel = np.full((panel_height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

        center_y = panel_height // 2

        text_size = cv2.getTextSize(text, FONT, 2, 3)[0]
        text_x = (PANEL_WIDTH - text_size[0]) // 2
        if status == DetectionState.IDENTIFIED:
            exp_code = product["expansionCode"]
            cn = product['collectorNumber']
            lang = product['x-language']
            edition = "1st Edition" if product['x-isfirst'] else "Unlimited"
            text = f"{exp_code}-{cn}\n{lang}\n{edition}"
            cv2.putText(panel, text, (text_x, center_y + 20), FONT, 2, status.get_color_gbr(), 3, cv2.LINE_AA)
        else:
            text = status.name
            cv2.putText(panel, text, (text_x, center_y + 20), FONT, 2, status.get_color_gbr(), 3, cv2.LINE_AA)

        return panel
    elif status == DetectionState.RUNNING:
        panel = np.full((panel_height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

        y = BAR_PADDING
        for field, (value, fraction) in progress.items():
            label = f"{field}: {value}"
            cv2.putText(panel, label, (10, y + 15), FONT, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            bar_y = y + 25
            cv2.rectangle(panel, (10, bar_y), (PANEL_WIDTH - 10, bar_y + 10), (80, 80, 80), -1)

            fill_width = int((PANEL_WIDTH - 20) * min(max(abs(fraction), 0), 1))
            color = (0, 200, 0) if abs(fraction) >= 1.0 else (0, 165, 255)
            cv2.rectangle(panel, (10, bar_y), (10 + fill_width, bar_y + 10), color, -1)

            y += BAR_HEIGHT + BAR_PADDING

        return panel
    return None

def build_live_ui(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    # match panel height to frame height
    if panel.shape[0] != frame.shape[0]:
        resized_panel = np.full((frame.shape[0], PANEL_WIDTH, 3), 30, dtype=np.uint8)
        resized_panel[:panel.shape[0], :] = panel
        panel = resized_panel

    return np.hstack([frame, panel])

def process_frame(condition: str, frame: np.ndarray, debug: bool):
    global ts

    # when timer is set (=card was detected) wait 2 sec for new card before scanning
    if ts is not None and (time.time() - ts) < 2:
        return build_progress_panel(None, {}, None, panel_height=frame.shape[0])

    # Encode frame and send to service
    start = time.time()
    _, buffer = cv2.imencode(".jpg", frame)
    response = httpx.post(
        f"{INFERENCE_SRV_URL}/scan",
        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    )
    end = time.time()
    if debug:
        print(f"Inference: {end-start}s")

    if response.status_code != 200:
        print(response.status_code)
        return None
    
    data = response.json()
    text = data.get("text", {})
    editions = data.get("editions", {})

    status = DetectionState.RUNNING
    card_scanned, progress = stabilizer.forward(ocr_output=text, edition_dets=editions)
    if not card_scanned:
        return build_progress_panel(None, progress, status, panel_height=frame.shape[0])
    
    set_code = progress["set_code"][0]
    first_ed_0 = progress["first_ed_0"][0]
    first_ed_1 = progress["first_ed_1"][0]
    response = httpx.post(
        f"{INFERENCE_SRV_URL}/search",
        json={"set_code": set_code}
    )
    if response.status_code == 500:
        print("ygo-srv crashed")
        return None
    if response.status_code == 404:
        print(response)
        return build_progress_panel(None, progress, DetectionState.ERRONEOUS, panel_height=frame.shape[0])
    if response.status_code == 409:
        print(response)
        return build_progress_panel(None, progress, DetectionState.AMBIGUOUS, panel_height=frame.shape[0])

    product = response.json()
    product["x-isfirst"] = first_ed_0 or first_ed_1
    product["x-condition"] = condition
    csv_builder.append(product=product)
    stabilizer.clear()
    ts = time.time()

    progress_panel = build_progress_panel(product, progress, DetectionState.IDENTIFIED, panel_height=frame.shape[0])
    return progress_panel

def run_capture_loop(condition: str, debug: bool = False, frame_skip: int = 10) -> list:
    cap = cv2.VideoCapture("http://127.0.0.1:8080/video")
    cv2.namedWindow("tcgscanner", cv2.WINDOW_NORMAL)

    progress_panel = build_progress_panel(None, {}, status=None, panel_height=480)  # placeholder, adjust height

    frames = []
    counter = 0
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if counter % frame_skip == 0:
                next_panel = process_frame(frame=img, debug=debug, condition=condition)
                if next_panel is not None: progress_panel = next_panel

            display = build_live_ui(img, progress_panel)
            # frames.append(display)
            cv2.imshow("tcgscanner", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            counter += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return frames


def main():
    args = parse_args()

    with open("shared/yugioh_layout.json", "r") as f:
        data = json.load(f)
    response = httpx.post(
        f"{INFERENCE_SRV_URL}/configure",
        json=data
    )
    response.raise_for_status()

    if args.debug:
        response = httpx.get(f"{INFERENCE_SRV_URL}/toggle-debug")
        response.raise_for_status()

    frames = run_capture_loop(debug=args.debug, frame_skip=args.frame_skip, condition=args.condition)

    csv_builder.build()

    if args.record:
        writer = imageio.get_writer("output/recording.mp4", fps=30)
        for f in frames:
            # OpenCV returns BGR, imageio expects RGB
            writer.append_data(f[:, :, ::-1])
        writer.close()


if __name__ == "__main__":
    main()