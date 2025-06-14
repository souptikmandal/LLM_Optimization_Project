from transformers.models import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype="auto", device_map="auto")

inputs = tokenizer("Explain the concept of large language models in simple terms.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
