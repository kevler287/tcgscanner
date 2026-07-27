import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Tuple
from shared.tcg_config import TCGConfig

class YugiohStabilizer:

    def __init__(self, tcg_config: TCGConfig):
        self.yugioh_config = tcg_config
        self.text_stabilizers: Dict[str, TextStabilizer] = {}
        self.edition_stabilizers: Dict[str, BinaryStabilizer] = {}
        for field, cfg in self.yugioh_config.ocr_fields.items():
            self.text_stabilizers[field] = TextStabilizer(
                forget_rate=cfg.forget_rate,
                stability_factor=cfg.stability_factor
            )
        for loc_name in self.yugioh_config.edition_areas.keys():
            self.edition_stabilizers[loc_name] = BinaryStabilizer(
                forget_rate=0.7,
                stability_factor=0.8
            )

    def _new_epoch(self):
        for stab in self.text_stabilizers.values():
            stab.new_epoch()
        for stab in self.edition_stabilizers.values():
            stab.new_epoch()

    def _eval(self):
        payload = {loc_name: stab.get_stabilized_class() for loc_name, stab in self.edition_stabilizers.items()}
        text_snapshot = {field: stab.get_dominant_element() for field, stab in self.text_stabilizers.items()}
        if all(stab.locked for stab in self.text_stabilizers.values()) and all(stab.locked for stab in self.edition_stabilizers.values()):
            payload.update({field: stab_text[0] for field, stab_text in text_snapshot.items()})
            return True, payload
        payload.update({field: stab_text[1] for field, stab_text in text_snapshot.items()})
        return False, payload

    def forward(self, ocr_output: dict, edition_dets: dict):
        for field, texts in ocr_output.items():
            if field not in self.text_stabilizers.keys():
                continue
            for text in texts:
                self.text_stabilizers[field].forward(text)
        for loc_name, first_ed_prob in edition_dets.items():
            if field not in self.text_stabilizers.keys():
                continue
            self.edition_stabilizers[loc_name].forward(true_prob=first_ed_prob)
        self._new_epoch()
        return self._eval()
    
    def clear(self):
        for stab in self.text_stabilizers.values():
            stab.reset() 
        for stab in self.edition_stabilizers.values():
            stab.reset() 

class BinaryStabilizer(BaseModel):
    locked: bool = False
    forget_rate: float
    stability_factor: float
    weight: float = 0.0

    def get_threshold(self):
        limes = 1/(1-self.forget_rate)
        return limes * self.stability_factor

    def new_epoch(self):
        if self.locked:
            return
        self.weight *= self.forget_rate

    def forward(self, true_prob: float):
        if self.locked:
            return
        signed_prob = (true_prob - 0.5) * 2
        self.weight += signed_prob

        if abs(self.weight) >= self.get_threshold():
            self.locked = True

    def reset(self):
        self.weight = 0.0
        self.locked = False

    def get_stabilized_class(self):
        return np.clip(self.weight / (self.get_threshold() * self.stability_factor), a_min=-1.0, a_max=1.0)

class TextStabilizer(BaseModel):
    locked: bool = False
    forget_rate: float
    stability_factor: float
    weight_per_element: Dict[str, float] = Field(default_factory=dict)

    def get_threshold(self):
        limes = 1/(1-self.forget_rate)
        return limes * self.stability_factor

    def new_epoch(self):
        if self.locked:
            return
        for element in self.weight_per_element.keys():
            self.weight_per_element[element] *= self.forget_rate

    def forward(self, new_value: Tuple[str, float]):
        if self.locked:
            return
        text, conf = new_value
        if conf == 0.0 or len(text) == 0:
            return
        if text in self.weight_per_element.keys():
            self.weight_per_element[text] += conf
        else:
            self.weight_per_element[text] = conf

        dominant_text, weight = self.get_dominant_element()
        if dominant_text is not None:
            if weight >= self.get_threshold():
                self.locked = True

    def reset(self):
        self.weight_per_element = {}
        self.locked = False

    def get_dominant_element(self):
        if len(self.weight_per_element) == 0:
            return None, -1
        max_text = max(self.weight_per_element, key=self.weight_per_element.get)
        reached = np.clip(self.weight_per_element[max_text] / (self.get_threshold() * self.stability_factor), a_min=0.0, a_max=1.0)
        return max_text, reached
