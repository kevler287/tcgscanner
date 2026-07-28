import json
from typing import Dict, List
from dataclasses import asdict, dataclass

@dataclass
class TCGConfig:
    tcg: str
    catalog: str
    card_w: int
    card_h: int
    ocr_fields: Dict[str, List[List[float]]] 
    edition_areas: Dict[str, List[List[float]]]

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def load(cls, path: str) -> "TCGConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)