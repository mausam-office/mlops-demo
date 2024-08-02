import pandas as pd
from src.ml_project import logger

from src.ml_project.config.configuration import ConfigurationManager
from src.ml_project.components.data_transformation import DataTransform


class DataTransformTrainingPipeline:
    def __init__(self) -> None:
        self.stage_name = "Data Transform Stage"

    def execute(self, df: pd.DataFrame):
        config = ConfigurationManager()
        data_transform = DataTransform(config.get_data_transformation_config())
        X_train, X_test, y_train, y_test = data_transform.split(df)
        return X_train, X_test, y_train, y_test