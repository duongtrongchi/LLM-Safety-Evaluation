import os
import json
import argparse
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from datasets import load_dataset

# from provider.unsloth import UnslothModel
from provider.transformers import chat
load_dotenv()

logger.add("logs/data.txt", rotation="5 MB", retention="7 days", level="INFO")

DATASET_ID = os.getenv("HUGGINGFACE_DATASET_ID")
HUGGINGFACE_TOKEN=os.getenv("HUGGINGFACE_TOKEN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark on dataset")
    parser.add_argument('--model_id', type=str, default=None, help='Name of the model')
    parser.add_argument('--dataset_id', type=str, default="d-llm/detoxic_benchmark", help='Path to the dataset file')
    args = parser.parse_args()


    if args.model_id == None:
        raise Exception("model_id parameter is required!")


    ds = load_dataset(DATASET_ID, token=HUGGINGFACE_TOKEN, split="train") #.select(range(10))
    # model = UnslothModel(model_id=args.model_id, max_seq_length=2048, device="cuda")
    file_name = args.model_id.split('/')[-1] + " " + datetime.now().strftime("%d-%m-%Y %H:%M:%S")


    logger.info("Inference Processing...")
    for i in tqdm(ds, desc="Processing dataset"):
        try:
            response = chat(i['prompt'])
            with open(f'./result/{file_name}.jsonl', 'a', encoding='utf8') as f:
                f.write(json.dumps({
                    "prompt": i['prompt'],
                    "response": response,
                }, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(e)

