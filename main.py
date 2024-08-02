from src.ml_project import logger
from src.ml_project.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.ml_project.pipeline.stage_02_data_loading import DataLoadingTrainingPipeline
from src.ml_project.pipeline.stage_03_data_validation import DataValidationTrainingPipeline
from src.ml_project.pipeline.stage_04_data_transformation import DataTransformTrainingPipeline
from src.ml_project.pipeline.stage_05_model_train import ModelTrainingPipeline
from src.ml_project.pipeline.stage_06_model_eval import ModelEvaluationPipeline



try:
   data_ingestion = DataIngestionTrainingPipeline()
   logger.info(f">>>>>> stage {data_ingestion.stage_name} started <<<<<<") 
   data_ingestion.execute()
   logger.info(f">>>>>> stage {data_ingestion.stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


try:
   data_loading = DataLoadingTrainingPipeline()
   logger.info(f">>>>>> stage {data_loading.stage_name} started <<<<<<") 
   df = data_loading.execute()
   # print(df.head())
   logger.info(f">>>>>> stage {data_loading.stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e


try:
   data_validation = DataValidationTrainingPipeline()
   logger.info(f">>>>>> stage {data_validation.stage_name} started <<<<<<") 
   data_validation.execute(df)
   logger.info(f">>>>>> stage {data_validation.stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


try:
   data_transfomation = DataTransformTrainingPipeline()
   logger.info(f">>>>>> stage {data_transfomation.stage_name} started <<<<<<") 
   X_train, X_test, y_train, y_test = data_transfomation.execute(df)
   logger.info(f">>>>>> stage {data_transfomation.stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


try:
   model_training = ModelTrainingPipeline()
   logger.info(f">>>>>> stage {model_training.stage_name} started <<<<<<") 
   model_path = model_training.execute(X_train, y_train)
   logger.info(f">>>>>> stage {model_training.stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


try:
   model_evaluation = ModelEvaluationPipeline()
   logger.info(f">>>>>> stage {model_evaluation.stage_name} started <<<<<<") 
   model_evaluation.execute(model_path, X_test, y_test)
   logger.info(f">>>>>> stage {model_evaluation.stage_name} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

