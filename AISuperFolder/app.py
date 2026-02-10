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
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL
)

model.eval()
text = """
    CV
Name: Anton Polisko
I know Java, Python, Java Scritp and C#
I studied in Harvard University
I have 10 years experience in Java BackEnd development
"""


prompt = (f"""Take a look on CV, convert all information in understendable json format, like
          {
            "skills":"skill"
            "experience":"experience"
            "education":"education"
          }
          CV
          f{text}
"""
          )


inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== MODEL OUTPUT ===\n")
print(result)
