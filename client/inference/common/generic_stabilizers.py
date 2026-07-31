from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel, Field


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

    def get_progress(self):
        progress = np.clip(self.weight / (self.get_threshold() * self.stability_factor), a_min=-1.0, a_max=1.0)
        return self.weight > 0, progress

class TextStabilizer(BaseModel):
    locked: bool = False
    forget_rate: float = Field(default=0.8) #(-> lim = 5) has been proved to be a good overall value
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

        dominant_text, weight = self.get_progress()
        if dominant_text is not None:
            if weight >= 1.0:
                self.locked = True

    def reset(self):
        self.weight_per_element = {}
        self.locked = False

    def get_progress(self):
        if len(self.weight_per_element) == 0:
            return None, -1
        max_text = max(self.weight_per_element, key=self.weight_per_element.get)
        reached = np.clip(self.weight_per_element[max_text] / (self.get_threshold() * self.stability_factor), a_min=0.0, a_max=1.0)
        return max_text, reached