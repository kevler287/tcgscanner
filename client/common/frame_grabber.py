import threading, time
import cv2

class FrameGrabber:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while self.running:
                try:
                    ok, f = self.cap.read()
                except cv2.error:
                    if not self.running:
                        break
                    time.sleep(0.05)
                    continue
                if not ok:
                    time.sleep(0.01)
                    continue
                with self.lock:
                    self.frame = f
        finally:
            self.cap.release()

    def get(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=3)