from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class OutputCSVRow:
    cardmarketId: str
    quantity: int
    name: str
    set: str
    cn: str
    condition: str
    language: str
    isFirstEd: Optional[str] = None     # "true" or empty

    isSigned = ""                     # very exceptional
    price: float = 1000.0               # fix default will be overwritten by auto pricing bot of TCG PowerTools
    comment: str = "Daily shipping"
    buyPrice = ""                     # not maintained

class YugiohCSVBuilder:

    def __init__(self, dest_path = "output/yugioh/"):
        self.dest_path = Path(dest_path)
        self.csv_data: List[OutputCSVRow] = []

    def append(self, product: dict):
        self.csv_data.append(OutputCSVRow(
            cardmarketId=product['cardmarketId'],
            quantity=1,
            name=product['name'],
            set=product['expansion'],
            cn=product['collectorNumber'],
            condition=product["x-condition"],
            language=product["x-language"],
            isFirstEd=product["x-isfirst"]
        ))

    def _new_trace_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d")
        return str(timestamp)

    def build(self):
        self.dest_path.mkdir(parents=True, exist_ok=True)

        file_path = self.dest_path / f"{self._new_trace_id()}.csv"
        df = pd.DataFrame([asdict(row) for row in self.csv_data])

        file_exists = file_path.exists()
        df.to_csv(
            str(file_path),
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )
