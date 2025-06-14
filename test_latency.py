import time
import torch
import psutil
import os

from transformers import AutoTokenizer, AutoModelForCausalLM

# === CONFIGURATION ===
model_name = "microsoft/phi-2"
prompt = "Explain the concept of transformers in deep learning."
max_new_tokens = 100
use_gpu = torch.cuda.is_available()

# === LOAD MODEL AND TOKENIZER ===
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16 if use_gpu else torch.float32)
model.eval()

# === MEMORY USAGE BEFORE ===
def get_memory_usage():
    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
    else:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  # in MB

mem_before = get_memory_usage()

# === PREPARE INPUT ===
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# === INFERENCE TIMING ===
print("Running inference...")
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
end_time = time.time()

# === MEMORY USAGE AFTER ===
mem_after = get_memory_usage()

# === DECODE OUTPUT ===
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]

# === METRICS ===
latency = end_time - start_time
throughput = tokens_generated / latency if latency > 0 else 0
mem_used = mem_after - mem_before

# === RESULTS ===
print("\n=== Benchmark Results ===")
print(f"Latency:       {latency:.2f} seconds")
print(f"Throughput:    {throughput:.2f} tokens/sec")
print(f"Memory used:   {mem_used:.2f} MB")
print(f"Tokens output: {tokens_generated}")
print("\nGenerated text preview:")
print(generated)

