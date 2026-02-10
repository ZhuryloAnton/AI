import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm

BASE_MODEL = "microsoft/phi-4-mini-instruct"
ADAPTER_MODEL = "rmtlabs/phi-4-mini-adapter-v1"

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_MODEL,
    trust_remote_code=True
)

base_model = AutoModeForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype = torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL
)

prompt = """
You extract structured CV information.
Return ONLY valid JSON.

CV:
Name: Anton Polisko
Skills: Java, Python
Education: Harvard
Experience: 10 years Java
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
print("Parse Json")
data = json.loads(result)

