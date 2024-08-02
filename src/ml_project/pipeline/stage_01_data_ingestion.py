from src.ml_project import logger
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        self.stage_name = STAGE_NAME

    def execute(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()