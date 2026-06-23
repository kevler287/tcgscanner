from paddleocr import PaddleOCR
import numpy as np

from shared.tcg_config import TCGConfig

class TextExtractor:
    def __init__(self, tcg_config: TCGConfig, use_gpu: bool = True):
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=use_gpu, lang="en")
        self.ygo_config = tcg_config

    def extract(self, card_image: np.ndarray) -> str | None:
        h, w, _ = card_image.shape
        names = []
        crops = []
        for name, field_cfg in self.ygo_config.ocr_fields.items():
            y1 = int(h*field_cfg.position[0][1])
            y2 = int(h*field_cfg.position[1][1])
            x1 = int(w*field_cfg.position[0][0])
            x2 = int(w*field_cfg.position[1][0])
            crop = card_image[y1:y2, x1:x2]
            names.append(name)
            crops.append(crop)

        raw = self.ocr.ocr(crops, det=False)
        extracted = {}
        for name, result in zip(names, raw):
            extracted[name] = result

        return extracted