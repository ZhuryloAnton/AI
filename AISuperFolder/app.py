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
# ❌ common mistake: loading tokenizer from adapter
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


# =====================
# BASE MODEL
# ❌ common mistake: model.to(device) + device_map="auto" together
# =====================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)


# =====================
# ADAPTER
# ❌ common mistake: thinking this is a merge (it is NOT)
# =====================
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


# =====================
# INPUT
# =====================
cv_text = """
Имя: Иван Петров
Опыт: 3 года Python backend
Навыки: Python, FastAPI, PostgreSQL, Docker
Образование: Бакалавр ИТ
"""

prompt = f"""
Проанализируй CV и дай краткую оценку кандидата в JSON формате.

{{
  "skills": [],
  "experience": "",
  "education": ""
}}

CV:
{cv_text}

Ответ:
"""


# =====================
# TOKENIZE
# ❌ common mistake: inputs on CPU while model on GPU
# =====================
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}


# =====================
# GENERATE
# =====================
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )


# =====================
# DECODE
# ❌ common mistake: printing full prompt + answer
# =====================
generated_tokens = output[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(result)
