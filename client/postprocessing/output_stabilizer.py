from pydantic import BaseModel, Field
from typing import Dict
from shared.tcg_config import TCGConfig

class OutputStabilizer:

    def __init__(self, tcg_config: TCGConfig):
        self.card_config = tcg_config
        self.text_stabilizers = {}
        for field, cfg in self.card_config.ocr_fields.items():
            self.text_stabilizers[field] = TextStabilizer(
                forget_rate=cfg.forget_rate,
                stability_factor=cfg.stability_rate
            )

    def new_epoch(self):
        for stab in self.text_stabilizers.values():
            stab.new_epoch()


class TextStabilizer(BaseModel):
    locked: bool = False
    forget_rate: float
    stability_factor: float
    weight_per_element: Dict[str, float] = Field(default_factory=dict)

    def get_limes(self):
        return 1/(1-self.forget_rate)

    def new_epoch(self):
        if self.locked:
            return
        for element in self.weight_per_element.keys():
            self.weight_per_element[element] *= self.forget_rate

    def add_element(self, new_value: str):
        if new_value in self.weight_per_element.keys():
            self.weight_per_element[new_value] += 1
        else:
            self.weight_per_element[new_value] = 1

        dominant_text, weight = self.get_dominant_element()
        if dominant_text is not None:
            threshold = self.get_limes() * self.stability_factor
            if weight >= threshold:
                self.locked = True

    def reset(self):
        self.weight_per_element = {}
        self.locked = False

    def get_dominant_element(self):
        if len(self.weight_per_element) == 0:
            return None, -1
        max_text = max(self.weight_per_element, key=self.weight_per_element.get)
        return max_text, self.weight_per_element[max_text]
