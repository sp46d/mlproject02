import os
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size
from pathlib import Path
from textSummarizer.entity import DataLoaderConfig
from datasets import load_dataset


class DataLoader:
    
    def __init__(self, config: DataLoaderConfig):
        self.config = config
        
    def load_dataset(self):
        dataset_name = self.config.dataset_name
        local_data_path = self.config.local_data_path
        
        if not os.path.exists(local_data_path):
            dataset = load_dataset(dataset_name)
            dataset.save_to_disk(local_data_path)
            
            logger.info(f"Dataset {dataset_name} downloaded from HuggingFace hub")
        else:
            logger.info(f"{dataset_name} already exists of size {get_size(Path(local_data_path))}")