import os
import json
import torch
from tqdm import tqdm
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    classification_report
)

from src.utils import write_to_jsonl, clean_text
from src.utils import load_yaml_config


config = load_yaml_config('./configs/evaluation.yaml')
model_id = config['model']['id']
device = config['model']['device']
dtype = config['model']['dtype']


quantization_config = BitsAndBytesConfig(load_in_8bit=config['model']['quantization_8bit'])

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, quantization_config=quantization_config)


def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=config['moderation']['max_new_tokens'], pad_token_id=config['moderation']['pad_token_id'])
    prompt_len = input_ids.shape[-1]
    return str(tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)).strip()


def extract_file_name(path):
    filename = os.path.basename(path)
    name, extension = os.path.splitext(filename)
    return name.split(" ")[0]


def evaluation_pipeline(input_file_path: str = config['paths']['input_file'], output_dir: str = config['paths']['output_dir']):
    if input_file_path == 'None':
        assert "input_file_path must be provided."
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
    
    base_filename = extract_file_name(input_file_path)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = os.path.join(output_dir, f"{base_filename}_{timestamp}.jsonl")
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, desc="Processing data"):
            data = json.loads(line)
            data['prompt'] = clean_text(data['prompt'])
            data['response'] = clean_text(data['response'])

            prompt_chat_format = [{"role": "user", "content": data['prompt']}]
            response_chat_format = [{"role": "user", "content": data['response']}]

            prompt_score = moderate(prompt_chat_format)
            response_score = moderate(response_chat_format)

            data['prompt_score'] = prompt_score
            data['response_score'] = response_score

            write_to_jsonl(data, file_path)


def evaluate_prompt_response_scores(file_path):
    """
    Evaluates the binary classification performance based on the structure of prompt_score and response_score fields 
    in a JSONL file.
    """
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)

    y_true = []
    y_predict = []

    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Processing lines"):
            data = json.loads(line)
            question_score = data['prompt_score'].split('\n')
            response_score = data['response_score'].split('\n')

            y_true.append(1 if len(question_score) == 2 else 0)

            if len(question_score) == 1 and len(response_score) == 1:
                y_predict.append(0)
            elif (len(question_score) == 1 and len(response_score) == 2) or \
                 (len(question_score) == 2 and len(response_score) == 1):
                y_predict.append(1)
            else:
                y_predict.append(0)

    results = {
        "precision": precision_score(y_true, y_predict),
        "recall": recall_score(y_true, y_predict),
        "f1_score": f1_score(y_true, y_predict),
        "accuracy": accuracy_score(y_true, y_predict),
        "classification_report": classification_report(y_true, y_predict)
    }

    return results



