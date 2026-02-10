import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =====================
# MODELS
# =====================
BASE_MODEL = "microsoft/phi-4-mini-instruct"
ADAPTER = "rmtlabs/phi-4-mini-adapter-v1"


# =====================
# DEVICE
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# TOKENIZER
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Safety for generation
tokenizer.pad_token = tokenizer.eos_token


# =====================
# MODEL
# =====================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16
).to(device)

model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

model.config.pad_token_id = tokenizer.eos_token_id


# =====================
# INPUT DATA
# =====================
cv_text = """
ARTEM ANTONENKO
Head of Engineering | CTO | Engineering Executive
+352 661 367 192 | antonenko.artem@gmail.com | linkedin.com/in/artem-antonenko-al | artem-antonenko.com
Luxembourg | EU Citizen | Open to relocation (EU / Global)

EXECUTIVE SUMMARY
Head of Engineering with 18+ years of experience scaling engineering organizations, modernizing delivery, and
driving AI-first transformation across large enterprise portfolios. Led 100+ engineers and delivered platforms
serving 50M+ daily users. Spearheaded AI adoption across 12 companies, enabling 500+ developers with agentic
coding assistants, RAG knowledge systems, and AI-augmented SDLC practices, achieving 80% coding-time
reduction and 2,000% ROI.

CORE SKILLS
Leadership, Distributed Systems, Cloud Architecture, AI-enabled SDLC
"""


prompt = f"""
Analyze the CV below and return ONLY valid JSON.
Do not add explanations or extra text.

JSON format:
{{
  "skills": [],
  "experience": "",
  "education": "",
  "seniority": ""
}}

CV:
{cv_text}

JSON:
"""


# =====================
# TOKENIZE
# =====================
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}


# =====================
# GENERATE
# =====================
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )


# =====================
# DECODE
# =====================
generated_tokens = output[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(result)
