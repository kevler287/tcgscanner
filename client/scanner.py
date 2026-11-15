import argparse
import json
import subprocess
import sys
import traceback

import cv2
import httpx

from client.common.frame_grabber import FrameGrabber
from client.common.live_ui import build_live_ui
from client.common.scan_worker import ScanWorker
from client.yugioh.stabilizer import YugiohStabilizer
from client.yugioh.csv_builder import YugiohCSVBuilder
from shared.tcg_layout_model import TCGLayoutConfig

INFERENCE_SRV_URL = "http://localhost:8000"

stabilizer = YugiohStabilizer()
csv_builder = YugiohCSVBuilder()
config = TCGLayoutConfig.load("shared/yugioh_layout.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Yugioh card scanner CLI")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--condition", required=True, type=str, help="default condition set for all scanned cards")
    return parser.parse_args()

def setup_adb(port: int = 8080) -> None:
    """Ensure a device is connected and the camera port is forwarded. Raises on failure."""
    def adb(*args: str) -> str:
        try:
            result = subprocess.run(
                ["adb", *args], capture_output=True, text=True, timeout=15, check=True
            )
        except FileNotFoundError:
            raise RuntimeError("adb not found in PATH")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"adb {' '.join(args)} timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"adb {' '.join(args)} failed: {(e.stderr or e.stdout).strip()}")
        return result.stdout.strip()

    adb("start-server")
    state = adb("get-state")          # fails with no device; "device" only if ready
    if state != "device":
        raise RuntimeError(f"ADB device not ready (state: {state})")
    adb("forward", f"tcp:{port}", f"tcp:{port}")

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
    try:
        setup_adb(8080)
    except RuntimeError as e:
        print(f"ADB setup failed: {e}")
        sys.exit(1)
    
    args = parse_args()

    with open("shared/yugioh_layout.json", "r") as f:
        data = json.load(f)
    response = httpx.post(
        f"{INFERENCE_SRV_URL}/configure",
        json=data
    )
    response.raise_for_status()

    try:
        run_capture_loop(debug=args.debug, condition=args.condition)
    except:
        print(traceback.format_exc())

    csv_builder.build()

if __name__ == "__main__":
    main()