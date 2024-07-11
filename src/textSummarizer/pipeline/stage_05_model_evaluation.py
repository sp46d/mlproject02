from textSummarizer.logging import logger
from textSummarizer.components.model_evaluation import ModelEvalution
from textSummarizer.config.configuration import ConfigurationManager


class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(model_evaluation_config)
        model_evaluation.evaluate()