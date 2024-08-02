from src.ml_project.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.ml_project.utils.common import read_yaml, create_directories
from src.ml_project.entity.config_entity import (DataIngestionConfig, DataLoadingConfig,
                                                DataValidationConfig, DataTransformationConfig,
                                                ModelTrainingConfig, ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
    
    def get_data_loading_config(self):
        config = self.config.data_loading

        create_directories([config.root_dir])

        return DataLoadingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )
    
    def get_data_validation_config(self):
        config = self.config.data_validation

        create_directories([config.root_dir])

        return DataValidationConfig(
            config.root_dir,
            config.status_file,
            self.schema
        )
    
    def get_data_transformation_config(self):
        config = self.config.data_transformation

        create_directories([config.root_dir])

        return DataTransformationConfig(
            config.root_dir,
            config.test_size,
            self.schema
        )
    
    def get_model_training_config(self):
        config = self.config.model_training

        create_directories([config.root_dir])

        return ModelTrainingConfig(
            config.root_dir,
            config.model_name,
            config.model_store_type,
            config.algorithm,
            self.params
        )
    
    def get_model_evaluation_config(self):
        config = self.config.model_evaluation
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            config.root_dir,
            config.metrics_filepath,
            config.mlflow_uri,
            self.config.model_training.ml_type
        )
