import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model IDs
BASE_MODEL = "microsoft/Phi-4-mini-instruct"
ADAPTER_ID = "rmtlabs/phi-4-mini-adapter-v1"

# Device (use "cuda" if you have a GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load tokenizer (from base model)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# 2. Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,  # or torch.float16
    trust_remote_code=True,
    device_map="auto",
)

# 3. Load PEFT adapter on top of base model
model = PeftModel.from_pretrained(model, ADAPTER_ID)

# Optional: merge adapter into base for faster inference (no need to keep LoRA separate)
# model = model.merge_and_unload()

model.eval()

# 4. Simple chat-style generation
messages = [
    {"role": "user", "content": "What is 84 * 3 / 2?"}
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)