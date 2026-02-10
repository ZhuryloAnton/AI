import torch, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-4-mini-instruct"
ADAPTER_MODEL = "rmtlabs/phi-4-mini-adapter-v1"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.eval()

messages = [
    {"role": "system", "content": "Extract CV data and return ONLY valid JSON."},
    {"role": "user", "content": """
Name: Anton Polisko
Skills: Java, Python
Education: Harvard
Experience: 10 years Java
"""}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
json_text = re.search(r"\{.*\}", text, re.S).group()
data = json.loads(json_text)

print(data)
