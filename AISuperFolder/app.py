# ---- Phi-4 compatibility patch (MUST be first) ----
import transformers.modeling_flash_attention_utils as fa_utils

if not hasattr(fa_utils, "FlashAttentionKwargs"):
    class FlashAttentionKwargs(dict):
        pass
    fa_utils.FlashAttentionKwargs = FlashAttentionKwargs
# --------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/phi-4-mini-instruct"
ADAPTER_MODEL = "rmtlabs/phi-4-mini-adapter-v1"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

print("Loading adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL
)

model.eval()

cv_text = """
Name: Anton Polisko
Skills: Java, Python, JavaScript, C#
Education: Harvard University
Experience: 10 years of Java Backend development
"""

messages = [
    {
        "role": "system",
        "content": "You extract structured information and return valid JSON only."
    },
    {
        "role": "user",
        "content": f"""
Convert this CV into JSON with fields:
name, skills, education, experience.

CV:
{cv_text}
"""
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== MODEL OUTPUT ===\n")
print(result)
