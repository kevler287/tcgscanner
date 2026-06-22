from paddleocr import PaddleOCR
import numpy as np

from card_configs.card_config import CardConfig

class TextExtractor:
    def __init__(self, ygo_config: CardConfig, use_gpu: bool = True):
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=use_gpu, lang="en")
        self.ygo_config = ygo_config

    def extract(self, card_image: np.ndarray) -> str | None:
        h, w, _ = card_image.shape
        names = []
        crops = []
        for name, bb in self.ygo_config.ocr.items():
            y1 = int(h*bb[0][1])
            y2 = int(h*bb[1][1])
            x1 = int(w*bb[0][0])
            x2 = int(w*bb[1][0])
            crop = card_image[y1:y2, x1:x2]
            names.append(name)
            crops.append(crop)

        raw = self.ocr.ocr(crops, det=False)
        extracted = {}
        for name, result in zip(names, raw):
            extracted[name] = result

        return extracted