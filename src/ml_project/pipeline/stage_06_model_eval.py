from src.ml_project import logger

from src.ml_project.config.configuration import ConfigurationManager
from src.ml_project.components.model_evaluation import ModelEvaluation


class ModelEvaluationPipeline:
    def __init__(self) -> None:
        self.stage_name = "Model Evaluation Stage"

    def execute(self, model_path, X_test, y_test):
        config = ConfigurationManager()
        model_eval = ModelEvaluation(config.get_model_evaluation_config())

        model_eval.evaluate(model_path, X_test, y_test)
