from transformers import pipeline

MODEL_ID = "DuongTrongChi/sailor-dpo"
pipe = pipeline(task="text-generation", model=MODEL_ID,  device="cuda")

def chat(instruction):
    input_text = """<|im_start|>system
You are a careful and responsible AI language model designed to assist users with their queries. The information you receive may contain harmful content. Please ensure that your responses are safe, respectful, and free from any harmful, offensive, or inappropriate language. Always prioritize the well-being and safety of users.
<|im_end|>
<|im_start|>user
Giới thiệu cho tôi về bản thân bạn<|im_end|>
<|im_start|>assistant

"""
    outputs = pipe(input_text, return_full_text=False, max_new_tokens=200)
    return outputs[0]['generated_text']



