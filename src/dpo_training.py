import os
import argparse
from dotenv import load_dotenv
from loguru import logger

import comet_ml
from comet_ml import Experiment

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig


from src.processing.data import get_datasets, apply_chat_template
from src.utils import load_yaml_config

from huggingface_hub import login


def dpo_pipeline(config_file_path: str):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    load_dotenv()
    config = load_yaml_config(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID
    
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="llm-safety",
        # workspace="YOUR_WORKSPACE"
    )

    if not HUGGINGFACE_TOKEN:
        raise EnvironmentError("HUGGINGFACE_TOKEN not found in environment variables. Please set it in your .env file.")

    login(HUGGINGFACE_TOKEN)


    modes_selected = sum([
        int(config['model']['load_in_4bit']),
        int(config['model']['load_in_8bit']),
        int(config['model']['full_finetuning']),
    ])
    if modes_selected != 1:
        raise ValueError("Invalid configuration: Exactly one of [load_in_4bit, load_in_8bit, full_finetuning] must be True.")


    # For full-finetuning - set full_finetuning = True  and 8-bit finetuning - set load_in_8bit = True 
    if config['model']['load_in_4bit']:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model']['name'],
            max_seq_length = config['model']['max_seq_length'],
            dtype = config['model']['dtype'],
            load_in_4bit = config['model']['load_in_4bit'],
            load_in_8bit = config['model']['load_in_8bit'], 
            token = HUGGINGFACE_TOKEN
        )
    elif config['model']['load_in_8bit']:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model']['name'],
            max_seq_length = config['model']['max_seq_length'],
            dtype = config['model']['dtype'],
            load_in_4bit = config['model']['load_in_4bit'],
            load_in_8bit = config['model']['load_in_8bit'],
            token = HUGGINGFACE_TOKEN
        )
    elif config['model']['full_finetuning']:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model']['name'],
            max_seq_length = config['model']['max_seq_length'],
            dtype = config['model']['dtype'],
            full_finetuning = config['model']['full_finetuning'],
            token = HUGGINGFACE_TOKEN
        )


    raw_datasets = get_datasets(
        config['datasets']['sources'], 
        splits = config['datasets']['splits'],
    )
    column_names = list(raw_datasets["train"].features)


    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"},
        num_proc = config['datasets']['preprocessing']['num_proc'],
        remove_columns = column_names,
        desc = "Formatting comparisons with prompt template",
    )

    # # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in config['datasets']['splits']:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )


    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora']['r'],
        target_modules = config['lora']['target_modules'],
        lora_alpha = config['lora']['lora_alpha'],
        lora_dropout = config['lora']['lora_dropout'], # Supports any, but = 0 is optimized
        bias = config['lora']['bias'],    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = config['lora']['use_gradient_checkpointing'], # True or "unsloth" for very long context
        random_state = config['lora']['random_state'],
        max_seq_length = config['model']['max_seq_length'],
    )

    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = DPOConfig(
            per_device_train_batch_size = config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
            warmup_ratio = config['training']['warmup_ratio'],
            num_train_epochs = config['training']['num_train_epochs'],
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = config['training']['logging_steps'],
            optim = config['training']['optim'],
            seed = config['training']['seed'],
            output_dir = config['training']['output_dir'],
            beta = config['dpo']['beta'],
            max_length = config['dpo']['max_length'],
            max_prompt_length = config['dpo']['max_prompt_length'],
            learning_rate=float(config['dpo']['learning_rate']),
            report_to="comet_ml"
        ),
        train_dataset = raw_datasets['train'],
        # eval_dataset = YOUR_DATASET_HERE,
        tokenizer = tokenizer,

    )

 
    dpo_trainer.train()

    experiment.log_model("final_model", config['training']['output_dir'])

    model.save_pretrained_merged(config['artifacts']['trained_model_path'], tokenizer, save_method = "merged_16bit",)
    model.push_to_hub_merged(config['dpo']['huggingface_model_id'], tokenizer, save_method = "merged_16bit")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model with RLHF config:")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    logger.info("Start the dpo training process...")
    dpo_pipeline(args.config)
