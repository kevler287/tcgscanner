import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CardConfig:
    w: int
    h: int
    ocr: Dict[str, List[List[float]]] #[[x1,y1],[x2,y2]]

    @classmethod
    def load(cls, path: str) -> "CardConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            w=data["w"],
            h=data["h"],
            ocr=data["ocr"]
        )