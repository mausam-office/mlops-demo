import dagshub
import json
import os
import numpy as np
import mlflow

# from mlflow.models.signature import infer_signature
from dagshub.mlflow import patch_mlflow
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.ml_project import logger
from src.ml_project.entity.config_entity import ModelEvaluationConfig
from src.ml_project.config.configuration import ConfigurationManager
from src.ml_project.utils.common import save_json, load_bin
from urllib.parse import urlparse


# patch_mlflow()

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        self.config = config

    def evaluate(self, model_path, X_test, y_test):
        algo, train_params = self.get_train_params()

        dagshub.init(repo_owner='dev.ai-rl', repo_name='mlops-demo', mlflow=True, patch_mlflow=True)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        print(f"{self.config.mlflow_uri=}")

        mlflow.set_experiment("ml-structure")
        experiment = mlflow.get_experiment_by_name("ml-structure")
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        run_count = int(self.get_run_count()) + 1

        # with mlflow.start_run(experiment_id=experiment.experiment_id ,run_name=f"run_{run_count}", nested=True):
        with mlflow.start_run(run_name=f"run_{run_count}", nested=True):
            model = load_bin(model_path)
            pred = self.predict(model, X_test)

            match self.config.ml_type:
                case 'regression':
                    scores = self.reg_metrics(y_test, pred)
                    save_json(
                        path=Path(self.config.metrics_filepath), 
                        data={run_count:scores}
                    )

                case 'classification':
                    self.clf_metrics(y_test, pred)

                case _:
                    raise Exception(
                        f"Not implemented for {self.config.ml_type}. \
                            Valid ml types are: `regression` and `classification`"
                        )
            
            # log into mlflow
            self.log_into_mlflow(train_params, algo, scores, model, tracking_url_type_store)
            
    
    def predict(self, model, X_test):
        return model.predict(X_test)


    def reg_metrics(self, actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        return {'rmse':rmse, 'mae':mae, 'r2':r2}
    

    def clf_metrics(self, actual, pred):
        ...


    def log_into_mlflow(self, train_params, algo, scores, model, tracking_url_type_store):
        mlflow.log_params(train_params)

        mlflow.log_metrics(scores)

        print(f"{tracking_url_type_store=}")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "artifacts", registered_model_name=algo)
        else:
            mlflow.sklearn.log_model(model, "artifacts")

    def get_run_count(self):
        if not os.path.exists(self.config.metrics_filepath):
            run_count = 0
        else:
            with open(self.config.metrics_filepath) as f:
                run_count = max(list(json.load(f).keys()))
        
        return run_count
    

    def get_train_params(self):
        train_config = ConfigurationManager().get_model_training_config()
        algo = train_config.algorithm
        train_params = train_config.params[algo]
        return algo, train_params