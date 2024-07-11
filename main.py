from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from textSummarizer.logging import logger


STAGE_NAME = "Data ingestion stage"
try:
    logger.info(f">>> Stage '{STAGE_NAME}' Started <<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>> Stage '{STAGE_NAME}' Completed <<<\n\nx=========================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data validation stage"
try:
    logger.info(f">>> Stage '{STAGE_NAME}' Started <<<")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f">>> Stage '{STAGE_NAME}' Completed <<<\n\nx=========================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data transformation stage"
try:
    logger.info(f">>> Stage '{STAGE_NAME}' Started <<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>> Stage '{STAGE_NAME}' Completed <<<\n\nx=========================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model training stage"
try:
    logger.info(f">>> Stage '{STAGE_NAME}' Started <<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.main()
    logger.info(f">>> Stage '{STAGE_NAME}' Completed <<<\n\nx=========================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model evaluation stage"
try:
    logger.info(f">>> Stage '{STAGE_NAME}' Started <<<")
    model_trainer = ModelEvaluationPipeline()
    model_trainer.main()
    logger.info(f">>> Stage '{STAGE_NAME}' Completed <<<\n\nx=========================x")
except Exception as e:
    logger.exception(e)
    raise e