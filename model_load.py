from transformers.models import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Magistral-Small-2506", torch_dtype="auto", device_map="auto")
