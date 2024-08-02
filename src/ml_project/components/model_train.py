import os

from pathlib import Path
from src.ml_project.utils.common import save_bin
from src.ml_project import logger
from src.ml_project.entity.config_entity import ModelTrainingConfig
from sklearn.linear_model import ElasticNet


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config = config

    def train(self, X_train, y_train):
        match self.config.algorithm:
            case 'ElasticNet':
                return self.train_elasticnet(X_train, y_train)
    
    def train_elasticnet(self, X_train, y_train):
        hyp = self.config.params.ElasticNet
        alpha, l1_ratio = hyp.alpha, hyp.l1_ratio
        random_state = hyp.random_state
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)

        model.fit(X_train, y_train)

        filepath = Path(os.path.join(
            self.config.root_dir, 
            self.config.algorithm + '-' + self.config.model_name + '.' + self.config.model_store_type
        ))
        
        save_bin(model, filepath)

        return filepath
