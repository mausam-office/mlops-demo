import pandas as pd
from src.ml_project import logger

from src.ml_project.config.configuration import ConfigurationManager
from src.ml_project.components.data_validation import DataValidation


class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        self.stage_name = "Data Validation Stage"

    def execute(self, df: pd.DataFrame) -> bool:
        config = ConfigurationManager()
        data_validation = DataValidation(config=config.get_data_validation_config())

        data_validation.validate(df)

        return data_validation.validation_status