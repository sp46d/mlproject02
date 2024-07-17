from textSummarizer.logging import logger
from textSummarizer.components.data_loader import DataLoader
from textSummarizer.config.configuration import ConfigurationManager


class DataLoaderTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_loader = DataLoader(config.get_data_loader_config())
        data_loader.load_dataset()