import json
from dataclasses import dataclass

@dataclass
class CardConfig:
    w: int
    h: int

    @classmethod
    def load(cls, path: str) -> "CardConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            w=data["w"],
            h=data["h"]
        )