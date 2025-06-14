from vllm import LLM, SamplingParams
import torch

prompts = ["Explain the concept of large language models in simple terms."]

llm = LLM("mistralai/Mistral-7B-v0.1")

sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=100)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}, Generated Text: {generated_text}")
