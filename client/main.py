import argparse
import json
import time

import cv2
import httpx
import numpy as np

from client.common.detectionstate_enum import DetectionState
from client.common.frame_grabber import FrameGrabber
from client.common.live_ui import build_live_ui, build_progress_panel
from client.common.scan_worker import ScanWorker
from client.yugioh.stabilizer import YugiohStabilizer
from client.yugioh.csv_builder import YugiohCSVBuilder
from shared.tcg_layout_model import TCGLayoutConfig

INFERENCE_SRV_URL = "http://localhost:8000"
YGO_SRV_URL = "http://localhost:8001"

stabilizer = YugiohStabilizer()
csv_builder = YugiohCSVBuilder()
config = TCGLayoutConfig.load("shared/yugioh_layout.json")
ts = None

def parse_args():
    parser = argparse.ArgumentParser(description="Yugioh card scanner CLI")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--condition", required=True, type=str, help="default condition set for all scanned cards")
    return parser.parse_args()

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
        f"{YGO_SRV_URL}/search",
        json={"set_code": set_code}
    )
    if response.status_code != 200:
        stabilizer.clear()
        ts = time.time()
        if response.status_code == 500:
            print("ygo-srv crashed")
            return None
        if response.status_code == 404:
            print(response.json())
            return build_progress_panel(None, progress, DetectionState.ERRONEOUS, panel_height=frame.shape[0])
        if response.status_code == 409:
            print(response.json())
            return build_progress_panel(None, progress, DetectionState.AMBIGUOUS, panel_height=frame.shape[0])

    product = response.json()
    product["x-isfirst"] = first_ed_0 or first_ed_1
    product["x-condition"] = condition
    csv_builder.append(product=product)
    stabilizer.clear()
    ts = time.time()

    progress_panel = build_progress_panel(product, progress, DetectionState.IDENTIFIED, panel_height=frame.shape[0])
    return progress_panel

def run_capture_loop(condition: str, debug: bool = False, scan_interval: float = 0.3) -> list:
    cv2.namedWindow("tcgscanner", cv2.WINDOW_NORMAL)

    cam = FrameGrabber("http://127.0.0.1:8080/video")
    worker = ScanWorker(
        condition=condition,
        debug=debug,
        stabilizer=stabilizer,
        csv_builder=csv_builder,
        min_interval=scan_interval
    )
    last_img = None
    try:
        while True:
            img = cam.get()
            if img is None:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue

            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if img is not last_img:        # only hand over genuinely new frames
                worker.submit(img)
                last_img = img

            state, progress, product = worker.get()

            cv2.imshow("tcgscanner", build_live_ui(img, product=product, progress=progress, status=state))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        worker.stop()
        cv2.destroyAllWindows()
    return []


def main():
    args = parse_args()

    with open("shared/yugioh_layout.json", "r") as f:
        data = json.load(f)
    response = httpx.post(
        f"{INFERENCE_SRV_URL}/configure",
        json=data
    )
    response.raise_for_status()

    run_capture_loop(debug=args.debug, condition=args.condition)

    csv_builder.build()

if __name__ == "__main__":
    main()