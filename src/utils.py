import yaml
import re
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


def write_to_jsonl(data_dict, output_file_path):
    """
    Write a Python dictionary containing Vietnamese content to a JSONL file.
    """    
    try:
        with open(output_file_path, 'a', encoding='utf-8') as file:
            json_line = json.dumps(data_dict, ensure_ascii=False)
            file.write(json_line + '\n')
        return True
    except Exception as e:
        print(f"Error writing to JSONL file: {e}")
        return False


def clean_text(input_text):
    cleaned_text = re.sub(r'\n+', '\n', input_text)
    return cleaned_text.strip()
