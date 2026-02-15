import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "microsoft/Phi-4-mini-instruct"
adapter_id = "rmtlabs/phi-4-mini-adapter-v1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    use_fast=False   # VERY IMPORTANT
)

# Load base model FIRST
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# THEN attach adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_id
)

model.eval()

print("✅ Model + Adapter loaded successfully")


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
You are a CV parser.

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

JSON:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.1,
        do_sample=False
    )

full_output = tokenizer.decode(output[0], skip_special_tokens=True)
result = full_output.split("JSON:")[-1].strip()

print("\n=== MODEL OUTPUT ===\n")
print(result)

try:
    parsed = json.loads(result)
    print("\n✅ Valid JSON")
except:
    print("\n❌ Not valid JSON")