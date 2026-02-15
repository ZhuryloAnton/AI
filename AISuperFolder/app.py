import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "microsoft/Phi-4-mini-instruct"
adapter_id = "rmtlabs/phi-4-mini-adapter-v1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True
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
