import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------
# CONFIG
# ---------------------
BASE_MODEL = "microsoft/phi-4-mini-instruct"
ADAPTER = "rmtlabs/phi-4-mini-adapter-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(">>> Device:", device, flush=True)

# ---------------------
# TOKENIZER
# ---------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------
# MODEL
# ---------------------
print(">>> Loading model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
).to(device)

model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
model.config.pad_token_id = tokenizer.eos_token_id

print(">>> Model loaded", flush=True)

# ---------------------
# LONG INPUT
# ---------------------
cv_text = """
ARTEM ANTONENKO
Head of Engineering | CTO | Engineering Executive
18+ years experience. Led 100+ engineers. Built AI systems.
""" * 20   # <- artificially long prompt

prompt = f"""
Read the CV below and answer ONE question.

Question: What is the candidate's seniority level?

CV:
{cv_text}

Answer:
"""

# ---------------------
# TOKENIZE
# ---------------------
print(">>> Tokenizing input...", flush=True)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print(">>> Input tokens:", inputs["input_ids"].shape[1], flush=True)

# ---------------------
# GENERATE
# ---------------------
print(">>> Starting generation...", flush=True)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

print(">>> Generation finished", flush=True)

# ---------------------
# DECODE
# ---------------------
generated_tokens = output[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n=== MODEL OUTPUT ===")
print(result)
