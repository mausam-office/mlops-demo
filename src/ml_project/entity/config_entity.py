from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataLoadingConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir    : Path
    status_file : Path
    schema_store: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    test_size: float
    schema_store: dict


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model_name: str
    model_store_type: str
    algorithm: str
    params: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_filepath: Path
    mlflow_uri: str
    ml_type: str
