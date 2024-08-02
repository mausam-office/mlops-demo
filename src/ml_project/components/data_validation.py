import pandas as pd

from src.ml_project import logger
from src.ml_project.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config
        self.validation_status = True
    
    def validate(self, df: pd.DataFrame):
        columns = df.columns 
        schema_store = self.config.schema_store.COLUMNS.keys() 

        self.validate_columns_presence(columns, schema_store)
        
        self.write_status()

    def validate_columns_presence(self, columns, schema_store):
        for col in columns:
            if col not in schema_store:
                self.validation_status = False
                break

    def write_status(self):
        with open(self.config.status_file, 'w') as sts_file:
            sts_file.write(f"validation_status = {self.validation_status}")
