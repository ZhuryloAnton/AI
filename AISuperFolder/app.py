import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================================================
# CONFIG
# =====================================================
base_model_id = "microsoft/Phi-4-mini-instruct"
adapter_id = "rmtlabs/phi-4-mini-adapter-v1"

# =====================================================
# LOAD TOKENIZER
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    use_fast=False
)

# =====================================================
# LOAD BASE MODEL
# =====================================================
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)

# =====================================================
# ATTACH ADAPTER
# =====================================================
model = PeftModel.from_pretrained(
    base_model,
    adapter_id
)

model.eval()
model_device = next(model.parameters()).device

print("✅ Model + Adapter loaded successfully")

# =====================================================
# SAMPLE CV
# =====================================================
cv_text = """
John Smith
Email: john.smith@email.com
Phone: +1 555-123-4567

Skills:
Python, Machine Learning, SQL, AWS

Experience:
Software Engineer at TechCorp (2020-2023)
- Built ML pipelines
- Developed REST APIs

Education:
BSc Computer Science, University of California, 2020
"""

prompt = f"""
Extract information from the CV and return ONLY valid JSON.

Format:
{{
  "name": "",
  "email": "",
  "phone": "",
  "skills": [],
  "experience": [],
  "education": []
}}

CV:
{cv_text}
"""

messages = [
    {"role": "system", "content": "You are a CV parser. Return ONLY valid JSON."},
    {"role": "user", "content": prompt}
]

chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    chat_text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

inputs = {k: v.to(model_device) for k, v in inputs.items()}

# =====================================================
# 🔥 BYPASS PEFT GENERATE
# =====================================================
with torch.no_grad():
    output = base_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=400,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=True)

start = decoded.find("{")
end = decoded.rfind("}") + 1

if start != -1 and end != -1:
    result = decoded[start:end]
else:
    result = decoded

print("\n=== MODEL OUTPUT ===\n")
print(result)

try:
    parsed = json.loads(result)
    print("\n✅ Valid JSON")
    print(json.dumps(parsed, indent=2))
except Exception as e:
    print("\n❌ Not valid JSON")
    print("Error:", e)
