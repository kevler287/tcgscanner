import cv2
import numpy as np
from ultralytics import YOLO

from shared.tcg_config import TCGConfig

class CardSegmentor:
    def __init__(self, model_path: str, tcg_config: TCGConfig):
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.ygo_config = tcg_config

    def segment_and_warp(self, frame: np.ndarray) -> np.ndarray | None:

        results = self.model(frame, conf=0.5)[0]

        if results.masks is None:
            return None, None
        
        for i, (mask, box) in enumerate(zip(results.masks.xy, results.boxes)):
            pts = mask.astype(np.int32)

            # find smallest convex polygon from mask
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))

            # filter outliers with douglas-peucker algorithm
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) != 4:
                return None, approx

            approx = approx.reshape(4, 2) # (4, 1, 2) -> (4, 2)
            sorted_pts = self._sort_pts(approx)

            h_out = self.ygo_config.card_h
            w_out = self.ygo_config.card_w
            out_pts = np.float32([[0, 0], [0, h_out - 1], [w_out - 1, h_out - 1], [w_out - 1, 0]])
            M = cv2.getPerspectiveTransform(sorted_pts, out_pts)
            return cv2.warpPerspective(frame, M, (w_out, h_out)), sorted_pts
        
        return None, None
    
    def _sort_pts(self, pts):
        s = pts.sum(axis=1) # x+y per point
        diff = np.diff(pts, axis=1) #x-y per point
        return np.float32([
            pts[np.argmin(s)],   # top-left: sum of x+y is smallest
            pts[np.argmax(diff)],# top-right: x large, y small -> x-y = large number
            pts[np.argmax(s)],   # bottom-right: sum of x+y is largest
            pts[np.argmin(diff)] # bottom-left: x small, y large -> x-y = small number
        ])
        