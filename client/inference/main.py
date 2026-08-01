import argparse
import json
import time

import cv2
import httpx
import numpy as np

from client.inference.common.catalog_srv import ProductCatalogService
from client.inference.common.detectionstate_enum import DetectionState
from client.inference.yugioh.setcode_resolver import resolve_setcode
from client.inference.yugioh.stabilizer import YugiohStabilizer
from client.inference.yugioh.csv_builder import YugiohCSVBuilder
from shared.tcg_config import TCGConfig

SERVICE_URL = "http://localhost:8000"
PANEL_WIDTH = 900
BAR_HEIGHT = 60
BAR_PADDING = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX

stabilizer = YugiohStabilizer()
csv_builder = YugiohCSVBuilder()
config = TCGConfig.load("shared/yugioh.json")
catalog_srv = ProductCatalogService(config=config)
ts = None

def parse_args():
    parser = argparse.ArgumentParser(description="Yugioh card scanner CLI")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--frame-skip", type=int, default=10, help="process every Nth frame")
    return parser.parse_args()

def build_progress_panel(progress: dict, status: DetectionState, panel_height: int) -> np.ndarray:
    if status is not None:
        panel = np.full((panel_height, PANEL_WIDTH, 3), 30, dtype=np.uint8)

        center_y = panel_height // 2

        text = status.name
        text_size = cv2.getTextSize(text, FONT, 0.8, 2)[0]
        text_x = (PANEL_WIDTH - text_size[0]) // 2
        cv2.putText(panel, text, (text_x, center_y + 20), FONT, 0.8, status.get_color(), 3, cv2.LINE_AA)

        return panel
    else:
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

def build_live_ui(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    # match panel height to frame height
    if panel.shape[0] != frame.shape[0]:
        resized_panel = np.full((frame.shape[0], PANEL_WIDTH, 3), 30, dtype=np.uint8)
        resized_panel[:panel.shape[0], :] = panel
        panel = resized_panel

    return np.hstack([frame, panel])

def process_frame(frame: np.ndarray, debug: bool):
    global ts

    # when timer is set (=card was detected) wait 2 sec for new card before scanning
    if ts is not None and (time.time() - ts) < 2:
        return build_progress_panel({}, True, panel_height=frame.shape[0])

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
        status = None
        if card_scanned:
            status, products = find_product_from_detection(det_json=progress)
            csv_builder.append(det_json=progress, products=products)
            stabilizer.clear()
            ts = time.time()

        progress_panel = build_progress_panel(progress, status, panel_height=frame.shape[0])
        return progress_panel
    else:
        print(response.status_code)
        return None

def find_product_from_detection(det_json: dict):  
    set_code = det_json["set_code"][0]
    name = det_json["name"][0]
    first_ed_0 = det_json["first_ed_0"][0]
    first_ed_1 = det_json["first_ed_1"][0]

    if first_ed_0 and first_ed_1:
        det_json["error"] = "1st Edition label was detected in both locations"
        return DetectionState.ERRONEOUS, None
    
    ec_opts, language, cn_opts = resolve_setcode(setcode=set_code)
    det_json.update({"ec_opts": ec_opts, "language": language, "cn_opts": cn_opts})

    if any(x is None for x in [ec_opts, language, cn_opts]):
        det_json["error"] = "Card could not be identified due to set code resolve error"
        return DetectionState.ERRONEOUS, None

    products = catalog_srv.find_yugioh_card(card_name=name, ec_opts=ec_opts, cn_opts=cn_opts)
    if len(products) != 1:
        return DetectionState.AMBIGUOUS, products

    return DetectionState.IDENTIFIED, products

def run_capture_loop(debug: bool = False, frame_skip: int = 10) -> list:
    cap = cv2.VideoCapture("http://127.0.0.1:8080/video")
    cv2.namedWindow("Inference  –  [N] Next  [Q] Quit", cv2.WINDOW_NORMAL)

    progress_panel = build_progress_panel({}, status=False, panel_height=480)  # placeholder, adjust height

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

    with open("shared/yugioh.json", "r") as f:
        data = json.load(f)
    response = httpx.post(
        f"{SERVICE_URL}/configure",
        json=data
    )
    response.raise_for_status()

    if args.debug:
        response = httpx.get(f"{SERVICE_URL}/toggle-debug")
        response.raise_for_status()

    run_capture_loop(debug=args.debug, frame_skip=args.frame_skip)

    csv_builder.build()


if __name__ == "__main__":
    main()