import os
import json
import torch
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from huggingface_hub import login

from src.utils import write_to_jsonl, clean_text
from src.utils import load_yaml_config

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

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

