from src.ml_project import logger
from src.ml_project.components.data_loading import DataLoading
from src.ml_project.config.configuration import ConfigurationManager


class DataLoadingTrainingPipeline:
    def __init__(self) -> None:
        self.stage_name = "Data Loading Stage"
    
    def execute(self):
        config = ConfigurationManager()

        data_loading = DataLoading(config.get_data_loading_config())
        df = data_loading.load_from_file()

        return df