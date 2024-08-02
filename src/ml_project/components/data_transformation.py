import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml_project import logger
from src.ml_project.entity.config_entity import DataTransformationConfig


class DataTransform:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame):
        X, y = self.__split_independent(df)
        return self.__split_train_test(X, y)
        

    def __split_independent(self, df: pd.DataFrame):
        label = self.config.schema_store.TARGET_COLUMN.name
        X = df.drop(label, axis=1)
        y = df[label]
        return X, y
    
    def __split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size)
        return X_train, X_test, y_train, y_test
        
