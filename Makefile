INFERENCE_SCRIPT=src/inference.py
DPO_SCRIPT=src/dpo_training.py

# BASE MODEL
run-vinallama-base:
	poetry run python $(INFERENCE_SCRIPT) --model_id vilm/vinallama-2.7b-chat

run-qwen-base:
	poetry run python $(INFERENCE_SCRIPT) --model_id Qwen/Qwen2.5-1.5B-Instruct

run-sailor-base:
	poetry run python $(INFERENCE_SCRIPT) --model_id sail/Sailor-1.8B-Chat

# SFT
run-vinallama-sft:
	poetry run python $(INFERENCE_SCRIPT) --model_id d-llm/vinallama-2.7b-chat-sft

run-qwen-sft:
	poetry run python $(INFERENCE_SCRIPT) --model_id d-llm/Qwen2-1.5B-Instruct-sft

run-sailor-sft:
	poetry run python $(INFERENCE_SCRIPT) --model_id d-llm/sailor-1.8B-chat-sft

#ORPO
run-vinallama-orpo:
	poetry run python $(INFERENCE_SCRIPT) --model_id d-llm/vinallama-2.7b-chat-orpo

run-qwen-orpo:
	poetry run python $(INFERENCE_SCRIPT) --model_id d-llm/Qwen2-1.5B-Instruct-orpo

run-sailor-orpo:
	poetry run python $(INFERENCE_SCRIPT) --model_id d-llm/sailor-1.8b-orpo


# DPO
run-qwen-dpo:
	poetry run python $(DPO_SCRIPT) --config configs/qwen_dpo_config.yaml

