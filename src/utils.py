import yaml
from loguru import logger


def load_yaml_config(file_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(e)
        raise 
