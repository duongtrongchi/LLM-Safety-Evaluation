import os
import argparse
from dotenv import load_dotenv
from loguru import logger

from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()

from src.processing.data import get_datasets, apply_chat_template
from src.utils import load_yaml_config

from huggingface_hub import login


def dpo_pipeline(config_file_path: str):
    load_dotenv()
    config = load_yaml_config(config_file_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    login(HUGGINGFACE_TOKEN)


    # For full-finetuning - set full_finetuning = True  and 8-bit finetuning - set load_in_8bit = True 
    if config['model']['load_in_4bit']:
        if config['model']['load_in_8bit'] and config['model']['full_finetuning']:
            raise Exception(
                "Invalid configuration: You cannot enable both 8-bit loading and full finetuning when 4-bit loading is enabled. "
                "Choose only one method: set 'full_finetuning = True' for full finetuning, or 'load_in_8bit = True' for 8-bit finetuning."
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model']['name'],
            max_seq_length = config['model']['max_seq_length'],
            dtype = config['model']['dtype'],
            load_in_4bit = config['model']['load_in_4bit'],
            token = HUGGINGFACE_TOKEN
        )
    elif config['model']['load_in_8bit']:
        if config['model']['load_in_4bit'] and config['model']['full_finetuning']:
            raise Exception(
                "Invalid configuration: You cannot enable both 4-bit loading and full finetuning when 8-bit loading is enabled. "
                "Choose only one method: set 'full_finetuning = True' for full finetuning, or 'load_in_4bit = True' for 4-bit finetuning."
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model']['name'],
            max_seq_length = config['model']['max_seq_length'],
            dtype = config['model']['dtype'],
            load_in_8bit = config['model']['load_in_8bit'],
            token = HUGGINGFACE_TOKEN
        )
    elif config['model']['full_finetuning']:
        if config['model']['load_in_8bit'] and config['model']['load_in_4bit']:
            raise Exception(
                "Invalid configuration: You cannot enable both 8-bit loading and 4-bit when full finetuning loading is enabled. "
                "Choose only one method: set 'load_in_4bit = True' for 4-bit finetuning, or 'load_in_8bit = True' for 8-bit finetuning."
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = config['model']['name'],
            max_seq_length = config['model']['max_seq_length'],
            dtype = config['model']['dtype'],
            full_finetuning = config['model']['load_in_8bit'],
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
        ),
        train_dataset = raw_datasets['train'],
        # eval_dataset = YOUR_DATASET_HERE,
        tokenizer = tokenizer,

    )

 
    dpo_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model with RLHF config:")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    logger.info("Start the dpo training process...")
    dpo_pipeline(args.config)