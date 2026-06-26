from dataclasses import dataclass

@dataclass(frozen=True)
class TCGBucket:
    name: str = "tcg-image-data"
    ygo_prefix: str = "raw/ygo_cards/"
    background_prefix: str = "raw/backgrounds/"
    seg_models_prefix: str = "models/segmentation/"

@dataclass(frozen=True)
class ModelResultDataset:
    name: str = "model_results"
    model_runs_table: str = "model_runs"
    traing_epoch_table: str = "training_epochs"

@dataclass(frozen=True)
class ETLConfig:
    random_seed: 42
    bucket: TCGBucket = TCGBucket()
    model_results_dataset: ModelResultDataset = ModelResultDataset()

CONFIG = ETLConfig()