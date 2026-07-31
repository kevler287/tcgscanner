from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
from typing import List, Optional
from client.inference.common.catalog_srv import ProductCatalogService
from client.inference.yugioh.setcode_resolver import resolve_setcode
from shared.tcg_config import TCGConfig
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

@dataclass
class AmbiguousDetection:
    det_json: dict
    setcode_error: bool = False
    product_opts: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        return {
            "det_json": self.det_json,
            "setcode_error": self.setcode_error,
            "product_opts": (
                self.product_opts.to_dict(orient="records")
                if self.product_opts is not None
                else None
            ),
        }

class YugiohCSVBuilder:

    def __init__(self, dest_path = "output/yugioh/"):
        self.dest_path = Path(dest_path)
        self.config = TCGConfig.load("shared/yugioh.json")
        self.catalog_srv = ProductCatalogService(config=self.config)
        self.csv_data: List[OutputCSVRow] = []
        self.ambiguous_data: List[AmbiguousDetection] = []
    
    def detections_to_csv(self, ygo_dets: List[dict]):
        for det_json in ygo_dets:
            set_code = det_json["set_code"][0]
            name = det_json["name"][0]
            first_ed_0 = det_json["first_ed_0"][0]
            first_ed_1 = det_json["first_ed_1"][0]

            if first_ed_0 and first_ed_1:
                self.ambiguous_data.append(AmbiguousDetection(det_json=det_json))
                continue

            ec_opts, language, cn_opts = resolve_setcode(setcode=set_code)
            det_json.update({"ec_opts": ec_opts, "language": language, "cn_opts": cn_opts})

            if any(x is None for x in [ec_opts, language, cn_opts]):
                self.ambiguous_data.append(AmbiguousDetection(det_json=det_json, setcode_error=True))
                continue

            products = self.catalog_srv.find_yugioh_card(card_name=name, ec_opts=ec_opts, cn_opts=cn_opts)
            if len(products) != 1:
                self.ambiguous_data.append(AmbiguousDetection(det_json=det_json, product_opts=products))
                continue

            self.csv_data.append(OutputCSVRow(
                cardmarketId=products['cardmarketId'].iloc[0],
                quantity=1,
                name=products['name'].iloc[0],
                set=products['expansion'].iloc[0],
                cn=products['collectorNumber'].iloc[0],
                condition="NM", #TODO set static via arg
                language=language,
                isFirstEd=first_ed_0 or first_ed_1
            ))

        self._build(ygo_dets=ygo_dets)

    def _new_trace_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return str(timestamp)

    def _build(self, ygo_dets: List[dict]):
        out_path = self.dest_path / self._new_trace_id()
        out_path.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([asdict(row) for row in self.csv_data])
        df.to_csv(str(out_path / "bulklist.csv"), index=False)

        with open(out_path / "detections.json", "w") as f:
            json.dump([d for d in ygo_dets], f, indent=2)

        with open(out_path / "ambiguous.json", "w") as f:
            json.dump([x.to_dict() for x in self.ambiguous_data], f, indent=2) 
