from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import (DataLoaderConfig,
                                   DataIngestionConfig,
                                   DataValidationConfig,
                                   DataTransformationConfig,
                                   ModelTrainerConfig,
                                   ModelEvaluationConfig)

from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        params_file_path: Path = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        
        create_directories([Path(self.config.artifacts_root)])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([Path(config.root_dir)])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        
        return data_ingestion_config
    
    
    def get_data_loader_config(self) -> DataLoaderConfig:
        config = self.config.data_loader
        
        create_directories([Path(config.root_dir)])
        
        data_loader_config = DataLoaderConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            local_data_path=Path(config.local_data_path)
        )
        
        return data_loader_config
        
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        
        create_directories([Path(config.root_dir)])
        
        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        
        config = self.config.data_transformation
        
        create_directories([Path(config.root_dir)])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            tokenizer_name=config.tokenizer_name
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            base_model=config.base_model,
            fine_tuned_model=config.fine_tuned_model,
            use_fine_tuned_model=config.use_fine_tuned_model,
            out_dir=config.out_dir,
            dataset_text_field=params.dataset_text_field,
            num_train_epochs=params.num_train_epochs,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            learning_rate=params.learning_rate,
            lr_scheduler_type=params.lr_scheduler_type,
            logging_steps=params.logging_steps,
            max_seq_length=params.max_seq_length,
            lora_r=params.lora_r,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            lora_target_modules=params.lora_target_modules
        )
        
        return model_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([Path(config.root_dir)])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            metric_file_name=Path(config.metric_file_name)
        )
        
        return model_evaluation_config