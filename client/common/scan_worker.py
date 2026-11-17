import threading
import time
import traceback

import cv2
import httpx
import numpy as np

from client.common.detectionstate_enum import DetectionState

INFERENCE_SRV_URL = "http://localhost:8000"
YGO_SRV_URL = "http://localhost:8001"

class ScanWorker:
    """Scans the newest submitted frame in the background. Knows nothing about the camera."""

    def __init__(self, condition: str, debug: bool, stabilizer, scan_csv_builder, sift_csv_builder, det_pause_in_s: float = 2, min_interval: float = 0.0):
        self._http = httpx.Client(timeout=10)
        self.det_pause_in_s = det_pause_in_s
        self.condition = condition
        self.debug = debug
        self.stabilizer = stabilizer
        self.scan_csv_builder = scan_csv_builder
        self.sift_csv_builder = sift_csv_builder
        self.t_scan = None

        self._state = DetectionState.RUNNING
        self._progress = {}
        self._product = None

        self._min_interval = min_interval
        self._pending = None
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Cheap and non-blocking: replaces any frame that is still waiting."""
        with self._lock:
            self._pending = frame
        self._wakeup.set()

    def get(self):
        with self._lock:
            return self._state, self._progress, self._product

    def _loop(self):
        while self._running:
            self._wakeup.wait(timeout=0.1)
            with self._lock:
                frame, self._pending = self._pending, None
                self._wakeup.clear()
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                self._process_frame(frame=frame.copy())
            except Exception:
                traceback.print_exc()      # don't let the thread die silently
                time.sleep(0.5)
                continue

            remaining = self._min_interval - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def stop(self):
        self._running = False
        self._wakeup.set()
        self._thread.join(timeout=3)
        self._http.close()

    def _call_inference_srv(self, frame):
        # Encode frame and send to service
        start = time.time()
        _, buffer = cv2.imencode(".jpg", frame)
        response = self._http.post(
            f"{INFERENCE_SRV_URL}/scan",
            files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        )
        end = time.time()
        if self.debug:
            print(f"Inference: {end-start}s")

        response.raise_for_status()
        return response.json()

    def _identify_product(self, set_code: str):
        response = self._http.post(f"{YGO_SRV_URL}/search", json={"set_code": set_code})
        if response.status_code == 200:
            return DetectionState.IDENTIFIED, response.json()
        print(response.json())
        if response.status_code == 409:
            return DetectionState.AMBIGUOUS, None
        if response.status_code == 404:
            return DetectionState.ERRONEOUS, None
        return None, None

    def _get_price_for_product(self, product_id: str):
        response = self._http.get(f"{YGO_SRV_URL}/price/{product_id}")
        if response.status_code == 200:
            return response.json()
        print(response.json())
        return {}

    def _process_frame(self, frame: np.ndarray):
        # when timer is set (=card was detected) wait 2 sec for new card before scanning
        if self.t_scan is not None and (time.time() - self.t_scan) < self.det_pause_in_s:
            return

        resp = self._call_inference_srv(frame=frame)

        card_scanned, progress = self.stabilizer.forward(ocr_output=resp.get("text", {}), edition_dets=resp.get("editions", {}))
        if not card_scanned:
            with self._lock:
                self._progress = progress
                self._state = DetectionState.RUNNING
            return

        set_code = progress["set_code"][0]
        state, resp = self._identify_product(set_code=set_code)

        self.stabilizer.clear()
        self.t_scan = time.time()

        if state != DetectionState.IDENTIFIED:
            with self._lock:
                self._progress = progress
                self._state = state
            return

        product_id = resp["cardmarketId"]
        price_info = self._get_price_for_product(product_id=product_id)
        resp["trend"] = price_info.get("trend", -1)
        first_ed_0 = progress["first_ed_0"][0]
        first_ed_1 = progress["first_ed_1"][0]
        resp["x-isfirst"] = first_ed_0 or first_ed_1
        resp["x-condition"] = self.condition
        if resp["trend"] < 0.1:
            self.sift_csv_builder.append(product=resp)
        else:
            self.scan_csv_builder.append(product=resp)

        with self._lock:
            self._product = resp
            self._progress = progress
            self._state = state
