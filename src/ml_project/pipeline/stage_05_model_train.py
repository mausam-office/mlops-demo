from src.ml_project import logger

from src.ml_project.config.configuration import ConfigurationManager
from src.ml_project.components.model_train import ModelTraining


class ModelTrainingPipeline:
    def __init__(self) -> None:
        self.stage_name = "Model Training Stage"

    def execute(self, X_train, y_train):
        config = ConfigurationManager()
        model_training = ModelTraining(config.get_model_training_config())

        model_path = model_training.train(X_train, y_train)

        return model_path