from typing import Dict, Tuple

from client.inference.common.generic_stabilizers import BinaryStabilizer, TextStabilizer
from client.inference.common.text_normalizer import letter_to_number
from shared.tcg_config import TCGConfig

class YugiohStabilizer:

    def __init__(self, tcg_config: TCGConfig):
        self.yugioh_config = tcg_config

        self.setcode_stabilizer = TextStabilizer(stability_factor=0.6)
        self.name_stabilizer = TextStabilizer(stability_factor=0.4)

        self.edition_stabilizers: Dict[str, BinaryStabilizer] = {}
        for loc_name in self.yugioh_config.edition_areas.keys():
            self.edition_stabilizers[loc_name] = BinaryStabilizer(
                forget_rate=0.7,
                stability_factor=0.8
            )

    def _new_epoch(self):
        for stab in self.edition_stabilizers.values():
            stab.new_epoch()
        self.name_stabilizer.new_epoch()
        self.setcode_stabilizer.new_epoch()

    def _eval(self):
        progress = {
            "set_code": self.setcode_stabilizer.get_progress(),
            "name": self.name_stabilizer.get_progress(),
            **{loc_name: stab.get_progress() for loc_name, stab in self.edition_stabilizers.items()}
        }
        if self.setcode_stabilizer.locked and all(stab.locked for stab in self.edition_stabilizers.values()):
            return True, progress
        return False, progress

    def forward(self, ocr_output: dict, edition_dets: dict):
        setcode_dets = [(letter_to_number(sc), prob) for sc, prob in ocr_output.get("set_code", [])]
        for sc_det in setcode_dets:
            if "-" not in sc_det[0]: continue
            self.setcode_stabilizer.forward(sc_det)

        for name_det in ocr_output.get("name", []):
            self.name_stabilizer.forward(name_det)

        for loc_name, first_ed_prob in edition_dets.items():
            if loc_name not in self.text_stabilizers.keys(): continue
            self.edition_stabilizers[loc_name].forward(true_prob=first_ed_prob)
        
        self._new_epoch()

        return self._eval()
    
    def clear(self):
        for stab in self.text_stabilizers.values():
            stab.reset() 
        for stab in self.edition_stabilizers.values():
            stab.reset()
