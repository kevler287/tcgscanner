import cv2
import numpy as np
from ultralytics import YOLO
import math

class CardSegmentor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.model.to("cuda")

    def segment_and_warp(self, frame: np.ndarray) -> np.ndarray | None:

        results = self.model(frame)

        if results.masks is None:
            return None
        
        for i, (mask, box) in enumerate(zip(results.masks.xy, results.boxes)):
            pts = mask.astype(np.int32)

            # find smallest convex polygon from mask
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))

            # filter outliers with douglas-peucker algorithm
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if len(approx) != 4:
                continue

            approx = approx.reshape(4, 2) # (4, 1, 2) -> (4, 2)
            sorted = self._sort_pts(approx)

            M = self._get_transform_matrix(pts=sorted)
            return cv2.warpPerspective(frame, M, (590,860))

    def _get_transform_matrix(self, pts: np.ndarray):
        dist_a = max(math.dist(pts[0], pts[1]), math.dist(pts[2], pts[3]))
        dist_b = max(math.dist(pts[0], pts[3]), math.dist(pts[1], pts[2]))
        if dist_a < dist_b:
            w, h = dist_a, dist_b
        else:
            w, h = dist_b, dist_a
        
        in_pts = pts
        out_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        return cv2.getPerspectiveTransform(in_pts, out_pts)
    
    def _sort_pts(pts):
        s = pts.sum(axis=1) # x+y per point
        diff = np.diff(pts, axis=1) #x-y per point
        return np.float32([
            pts[np.argmin(s)],   # top-left: sum of x+y is smallest
            pts[np.argmax(diff)],# top-right: x large, y small -> x-y = large number
            pts[np.argmax(s)],   # bottom-right: sum of x+y is largest
            pts[np.argmin(diff)] # bottom-left: x small, y large -> x-y = small number
        ])
        