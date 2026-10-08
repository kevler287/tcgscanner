import cv2

from paddleocr import PaddleOCR
import numpy as np

from shared.tcg_layout_model import TCGLayoutConfig

class TextExtractor:
    def __init__(self, tcg_config: TCGLayoutConfig, use_gpu: bool = True):
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
        extracted, named_crops = {}, {}
        for name, pos in self.yugioh_config.ocr_fields.items():
            y1, y2 = int(h * pos[0][1]), int(h * pos[1][1])
            x1, x2 = int(w * pos[0][0]), int(w * pos[1][0])
            crop = self._preprocess(card_image[y1:y2, x1:x2])
            result = self.ocr.ocr(crop, det=True, rec=True, cls=False)
            extracted[name] = self._unpack_rec_result(result)
            named_crops[name] = crop
        return extracted, named_crops

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # nur bei dunklen Crops CLAHE anwenden -> spart Rechenzeit bei den meisten Karten
        if gray.mean() < 110:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _unpack_rec_result(self, raw) -> tuple[str, float]:
        """
        Expects PaddleOCR output with det=True for ONE image:
        [[[box, [text, conf]], [box, [text, conf]], ...]]
        Returns (text, conf) of the first/best line, or ("", 0.0) if nothing was recognized.
        """
        try:
            lines = raw[0]
            if not lines:
                return "", 0.0
            text, conf = lines[0][1]
            return text, float(conf)
        except (TypeError, IndexError, KeyError, ValueError) as e:
            return "", -1

    def extract_raw(self, img: np.ndarray):
        return self.ocr.ocr(img)

    def debug_rec_input(self, crop: np.ndarray):
        rec = self.ocr.text_recognizer
        h, w = crop.shape[:2]
        imgC, imgH, imgW = rec.rec_image_shape
        max_wh_ratio = max(imgW / imgH, w / h)
        norm_img = rec.resize_norm_img(crop, max_wh_ratio)
        vis = ((norm_img.transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
        return vis