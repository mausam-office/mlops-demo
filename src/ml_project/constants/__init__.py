from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path('schema.yaml')

STAGES = [
    'data_ingestion', 'data_loading', 'data_validation', 'data_transformation',
    'model_training', 'model_evaluation'
]