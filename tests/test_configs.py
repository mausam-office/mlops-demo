import os
import pytest
import requests

# import sys
# sys.path.insert(1, 'D:/Mausam/CI/ml-structured-project')

from src.ml_project.constants import (CONFIG_FILE_PATH, PARAMS_FILE_PATH, 
                                      SCHEMA_FILE_PATH, STAGES
                                    )
from src.ml_project.utils.common import read_yaml


def test_file_existance():
    assert os.path.exists(CONFIG_FILE_PATH)
    assert os.path.exists(PARAMS_FILE_PATH)
    assert os.path.exists(SCHEMA_FILE_PATH)


def test_stages_config():
    configs = read_yaml(CONFIG_FILE_PATH)

    stages_in_cfgs = list(configs.keys())

    for stage in STAGES:
        assert stage in stages_in_cfgs


def test_params_for_model():
    configs = read_yaml(CONFIG_FILE_PATH)
    params = read_yaml(PARAMS_FILE_PATH)

    algorithms = list(params.keys())
    algorithm = configs.model_training.algorithm
    assert algorithm in algorithms

  
def test_dagshub_url():
    configs = read_yaml(CONFIG_FILE_PATH)
    mlflow_uri = configs.model_evaluation.mlflow_uri
    response = requests.get(mlflow_uri)
    assert response.status_code == 200

  
def test_data_src_url():
    configs = read_yaml(CONFIG_FILE_PATH)
    source_url = configs.data_ingestion.source_url
    response = requests.get(source_url)
    assert response.status_code == 200


