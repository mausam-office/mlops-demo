import pandas as pd

from src.ml_project import logger
from src.ml_project.entity.config_entity import DataLoadingConfig

class DataLoading:
    def __init__(self, config: DataLoadingConfig) -> None:
        self.config = config

    def load_from_file(self):
        try:
            df = pd.read_csv(self.config.data_path)
            return df
        except Exception as e:
            logger.error(f"Failed to load data due to {str(e)}")
            raise e

