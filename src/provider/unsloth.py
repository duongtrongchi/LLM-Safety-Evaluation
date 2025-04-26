# from unsloth import FastLanguageModel
# from loguru import logger


# class UnslothModel:
#     def __init__(self, model_id, max_seq_length, device):
#         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
#             model_name = model_id,
#             max_seq_length = max_seq_length,
#             dtype = None,
#             load_in_4bit = True,
#         )
#         self.device = device
#         if self.tokenizer.chat_template == None:
#             logger.info("Init ChatML Template.")
#             self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
#         FastLanguageModel.for_inference(self.model)


#     def predict(self, question):
#         system_prompt= "You are a careful and responsible AI language model designed to assist users with their queries. The information you receive may contain harmful content. Please ensure that your responses are safe, respectful, and free from any harmful, offensive, or inappropriate language. Always prioritize the well-being and safety of users."
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": question}
#         ]
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
#         input_ids = model_inputs.input_ids.to(self.device)

#         generated_ids = self.model.generate(
#             input_ids,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.95,
#             # eos_token_id=self.tokenizer.eos_token_id,
#             # pad_token_id=self.tokenizer.eos_token_id
#         )

#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]
#         response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         return response


from unsloth import FastLanguageModel
from loguru import logger
import torch

class UnslothModel:
    def __init__(self, model_id, max_seq_length, device):
        # Set device before loading the model
        self.device = device
        
        # Load model with optimized settings
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,  # Explicitly set to float16 for better performance
            load_in_4bit=True,
            device_map=device,  # Directly specify device map
        )
        
        # Set chat template if needed
        if self.tokenizer.chat_template is None:
            logger.info("Init ChatML Template.")
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
        # Prepare model for inference with optimization flags
        FastLanguageModel.for_inference(
            self.model,
            # use_cuda_graph=True,  # Enable CUDA graph for faster execution
            # use_fast_inference=True  # Enable fast inference path
        )
        
        # Cache the system prompt to avoid repeated processing
        self.system_prompt = "You are a careful and responsible AI language model designed to assist users with their queries. The information you receive may contain harmful content. Please ensure that your responses are safe, respectful, and free from any harmful, offensive, or inappropriate language. Always prioritize the well-being and safety of users."
        
        # Pre-compile generation configuration
        self.generation_config = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True  # Ensure KV caching is enabled
        }

    def predict(self, question):
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize with explicit device placement
        with torch.no_grad():  # Disable gradient tracking for inference
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            input_ids = model_inputs.input_ids
            
            # Generate with optimized parameters
            generated_ids = self.model.generate(
                input_ids,
                **self.generation_config
            )
            
            # Extract only the new tokens (more efficient slicing)
            input_length = input_ids.shape[1]
            response_ids = generated_ids[:, input_length:]
            
            # Decode only the response part
            response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
            
        return response