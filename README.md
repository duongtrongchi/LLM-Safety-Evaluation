# LLM-Safety-Evaluation

# 🛠️ Inference via Makefile

This project includes a `Makefile` to simplify the process of running inference on different models using the `src/inference.py` script.

## ✅ Prerequisites

Make sure you have:

- Installed project dependencies using `poetry install`
- A working Python environment managed by `poetry`
- Access to the required models (e.g., via Hugging Face)

## 🚀 Usage

Run the following commands using `make`:

### Base Models

```bash
make run-vinallama-base     
make run-qwen-base          
make run-sailor-base        
```

### SFT Models (Supervised Fine-Tuning)
```bash
make run-vinallama-sft      
make run-qwen-sft           
make run-sailor-sft         
```

### ORPO Models
```bash
make run-vinallama-orpo     
make run-qwen-orpo         
make run-sailor-orpo        
```


