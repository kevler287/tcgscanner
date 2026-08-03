import cv2

from paddleocr import PaddleOCR
import numpy as np

from shared.tcg_config import TCGConfig

class TextExtractor:
    def __init__(self, tcg_config: TCGConfig, use_gpu: bool = True):
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            use_gpu=use_gpu,
            lang="en",
            rec_image_shape="3, 48, 480",
            drop_score=0.3,
        )
        self.yugioh_config = tcg_config

    def extract(self, card_image: np.ndarray):
        h, w, _ = card_image.shape
        names = []
        crops = []
        for name, pos in self.yugioh_config.ocr_fields.items():
            y1 = int(h*pos[0][1])
            y2 = int(h*pos[1][1])
            x1 = int(w*pos[0][0])
            x2 = int(w*pos[1][0])
            crop = card_image[y1:y2, x1:x2]
            crop = self._preprocess(crop)
            names.append(name)
            crops.append(crop)

        raw = self.ocr.ocr(crops, det=False)
        extracted = {}
        for name, result in zip(names, raw):
            extracted[name] = result

        return extracted, crops

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # nur bei dunklen Crops CLAHE anwenden -> spart Rechenzeit bei den meisten Karten
        if gray.mean() < 110:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)