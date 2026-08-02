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
        self.ambiguous: List[dict] = []
        self.erroneous: List[dict] = []

    def append(self, det_progress: dict, products: pd.DataFrame):
        if products is None or len(products) == 0:
            self.erroneous.append(det_progress)
        elif len(products) != 1:
            self.ambiguous.append({"detection": det_progress, "products": products.to_dict(orient="records")})
        else:
            self.csv_data.append(OutputCSVRow(
                cardmarketId=products['cardmarketId'].iloc[0],
                quantity=1,
                name=products['name'].iloc[0],
                set=products['expansion'].iloc[0],
                cn=products['collectorNumber'].iloc[0],
                condition=det_progress["condition"],
                language=det_progress["language"],
                isFirstEd=det_progress["first_ed_0"][0] or det_progress["first_ed_1"][0]
            ))

    def _new_trace_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return str(timestamp)

    def build(self):
        out_path = self.dest_path / self._new_trace_id()
        out_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([asdict(row) for row in self.csv_data])
        df.to_csv(str(out_path / "bulklist.csv"), index=False)

        with open(out_path / "erroneous.json", "w") as f:
            json.dump(self.erroneous, f, indent=2)

        with open(out_path / "ambiguous.json", "w") as f:
            json.dump(self.ambiguous, f, indent=2) 
