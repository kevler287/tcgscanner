from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import product
import json
import re
from typing import List

import pandas as pd

from shared.tcg_config import TCGConfig

SET_CODE_PATTERN = re.compile(
    r"^(?P<expansion_id>[A-Za-z0-9]{3,4})"
    r"-"
    r"(?P<language_code>[A-Za-z]{0,2})"
    r"(?P<collectors_number>\d{3}|[A-Za-z]\d{2})$"
)

AMBIGUOUS_CHARS = {
    'O': '0',
    'I': '1',
    'A': '4',
    'S': '5',
    'G': '6',
    'B': '8',
}

# build bidirectional lookup: char -> set of possible substitutes (including itself)
_AMBIGUOUS_LOOKUP = {}
for a, b in AMBIGUOUS_CHARS.items():
    _AMBIGUOUS_LOOKUP.setdefault(a, {a}).add(b)
    _AMBIGUOUS_LOOKUP.setdefault(b, {b}).add(a)

@dataclass
class MatchedCard:
    c_js: dict
    catalog_entries: pd.DataFrame

class CSVConverter:

    def __init__(self, config: TCGConfig):
        self.config = config
        self.catalog = self._load_catalog(config.catalog)
        self.output: List[MatchedCard] = []

    def _load_catalog(self, path: str) -> pd.DataFrame:
        # load as str dtype to avoid type mismatches during comparison later
        df = pd.read_csv(path, dtype=str)
        return df.drop(columns=["scryfallId", "tcgplayerId"], errors="ignore")
    
    def _decompose_set_code(self, set_code: str):
        match = SET_CODE_PATTERN.match(set_code)
        if not match:
            return None

        return (
            match.group("expansion_id"),
            match.group("language_code"),
            match.group("collectors_number"),
        )
    
    def _find_in_catalog(self, c_js: dict, mode: str = "exact"):
        exp_id = c_js["exp_id"]
        cn = c_js["cn"]
        card_name = c_js["name"]
        if mode == "exact":
            remaining_entries = self.catalog[self.catalog["expansionCode"] == exp_id]

            if len(remaining_entries) > 1:
                remaining_entries = remaining_entries[remaining_entries["collectorNumber"] == cn]

                if len(remaining_entries) == 0:
                    self._find_in_catalog(c_js, mode="fuzzy")
                else:
                    self.output.append(MatchedCard(c_js=c_js, catalog_entries=remaining_entries))
            else:
                self._find_in_catalog(c_js, mode="fuzzy")

        if mode == "fuzzy":
            print("no exact match found")
            exp_id_permutations = self._generate_ambiguous_permutations(exp_id)
            print(f"permutations for {exp_id}: {str(exp_id_permutations)}")
            remaining_entries = self.catalog[self.catalog["expansionCode"].isin(exp_id_permutations)]

            filtered = remaining_entries[
                remaining_entries["name"].apply(lambda n: SequenceMatcher(None, card_name, n).ratio() > 0.6)
            ]

            if len(filtered) > 0:
                remaining_entries = filtered

            remaining_entries = remaining_entries[remaining_entries["collectorNumber"] == cn]
            self.output.append(MatchedCard(c_js=c_js, catalog_entries=remaining_entries))

    def _generate_ambiguous_permutations(self, exp_id: str) -> list[str]:
        char_options = [
            _AMBIGUOUS_LOOKUP.get(char, {char})
            for char in exp_id
        ]
        return ["".join(combo) for combo in product(*char_options)]
    
    def convert(self, card_jsons: List[dict]):
        for c_js in card_jsons:
            decom = self._decompose_set_code(c_js["set_code"])
            if decom is None:
                self.output.append(MatchedCard(c_js=c_js, catalog_entries=[]))
                continue

            exp_id, lang, cn = decom
            c_js.update({"exp_id": exp_id, "lang": lang, "cn": cn})
            self._find_in_catalog(c_js)
            with open("output.json", mode="w") as f:
                json.dump(
                    [
                        {
                            "c_js": m.c_js,
                            "catalog_entries": m.catalog_entries.to_dict(orient="records"),
                        }
                        for m in self.output
                    ],
                    f,
                )
