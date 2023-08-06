from src.mlFlowProject import logger
from mlFlowProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlFlowProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlFlowProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from mlFlowProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

STAGE_NAME = "data ingestion stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "data validation stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    data_val = DataValidationTrainingPipeline()
    data_val.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "data transformation stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model training stage"
try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<")
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
except Exception as e:
    logger.exception(e)
    raise e