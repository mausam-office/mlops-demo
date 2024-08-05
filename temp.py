from urllib.parse import urlparse
import dagshub
from dagshub.mlflow import patch_mlflow
import mlflow

patch_mlflow()

dagshub.init(repo_owner='dev.ai-rl', repo_name='mlops-demo', mlflow=True)

mlflow.set_registry_uri('https://dagshub.com/dev.ai-rl/mlops-demo.mlflow')

tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

mlflow.set_experiment("ml-structure")
experiment = mlflow.get_experiment_by_name("ml-structure")

with mlflow.start_run(experiment_id=experiment.experiment_id , nested=True):

# with mlflow.start_run(nested=True):
    mlflow.log_params({'alpha':0.01,'l1_ratio':0.9,'random_state':0})
    mlflow.log_metrics({'rmse':0.01,'mae':0.9,'r2':0})