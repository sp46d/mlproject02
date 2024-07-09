import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads a YAML file and returns a ConfigBox object
    
    Args:
        path_to_yaml (Path): path to the YAML file
        
    Raises:
        ValueError: if the YAML file is empty or does not exist
        e: empty file
        
    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"The YAML file at {path_to_yaml} is empty or does not exist.")
    except Exception as e:
        logger.error(f"Error parsing YAML file at {path_to_yaml}: {e}")
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool=True) -> None:
    """
    creates list of directories
    
    Args:
        path_to_directories (list): list of directories to be created
        verbose (bool, optional): whether to print directory creation messages. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")
            

@ensure_annotations
def get_size(path: Path) -> str:
    """
    get size in KB
    
    Args:
        path (Path): path to the file
        
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"{size_in_kb} KB"