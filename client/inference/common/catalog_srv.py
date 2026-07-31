from difflib import SequenceMatcher
from typing import List
import pandas as pd
from shared.tcg_config import TCGConfig

class ProductCatalogService:

    def __init__(self, config: TCGConfig):
        self.config = config
        self.catalog = self._load_catalog(config.catalog)

    def _load_catalog(self, path: str) -> pd.DataFrame:
        # load as str dtype to avoid type mismatches during comparison later
        return pd.read_csv(path, dtype=str)
    
    def find_yugioh_card(self, card_name: str, ec_opts: List[str], cn_opts: List[str]):
        remaining_entries = self.catalog[self.catalog["expansionCode"].isin(ec_opts)]

        if len(remaining_entries) == 0:
            return remaining_entries

        remaining_entries = remaining_entries[remaining_entries["collectorNumber"].isin(cn_opts)]

        if len(remaining_entries) <= 1:
            return remaining_entries

        filtered = remaining_entries[
            remaining_entries["name"].apply(lambda n: SequenceMatcher(None, card_name, n).ratio() > 0.5)
        ]

        return filtered
