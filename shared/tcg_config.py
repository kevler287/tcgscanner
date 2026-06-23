import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OCRField:
    position: List[List[float]] #[[x1,y1],[x2,y2]]
    forget_rate: float
    stability_factor: float

@dataclass
class TCGConfig:
    tcg: str
    card_w: int
    card_h: int
    ocr_fields: Dict[str, OCRField] 

    def __post_init__(self):
        self.ocr_fields = {
            k: v if isinstance(v, OCRField) else OCRField(**v)
            for k, v in self.ocr_fields.items()
        }

    @classmethod
    def load(cls, path: str) -> "TCGConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)