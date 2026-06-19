from paddleocr import PaddleOCR
import numpy as np

class TextExtractor:
    def __init__(self, use_gpu: bool = True):
        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=use_gpu, lang="en")

    def extract_name(self, card_image: np.ndarray) -> str | None:        
        name_crop = card_image[0:60, 30:-30]
        result = self.ocr.ocr(name_crop)