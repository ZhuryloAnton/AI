import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/phi-4-mini-instruct"
ADAPTER_MODEL = "rmtlabs/phi-4-mini-adapter-v1"

DATASET_PATH = "cv_dataset.json"  # list of {"text": ..., "label": {...}}

# ------------------ Load model ------------------

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model.eval()

# ------------------ Helpers ------------------

def normalize(value):
    if isinstance(value, list):
        return sorted(v.strip().lower() for v in value)
    if isinstance(value, str):
        return value.strip().lower()
    return value

def safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        return None

# ------------------ Evaluation ------------------

with open(DATASET_PATH) as f:
    dataset = json.load(f)

metrics = {
    "total": 0,
    "valid_json": 0,
    "exact_match": 0,
    "field_f1": {k: [] for k in ["name", "skills", "education", "experience"]},
}

for sample in tqdm(dataset):
    prompt = f"""
You extract structured CV information.

Return ONLY valid JSON with keys:
name, skills, education, experience.

CV:
{sample['text']}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(output[0], skip_special_tokens=True)
    pred = safe_json_load(raw)

    metrics["total"] += 1
    if pred is None:
        continue

    metrics["valid_json"] += 1

    gt = sample["label"]

    # exact match
    if all(normalize(pred.get(k)) == normalize(gt.get(k)) for k in gt):
        metrics["exact_match"] += 1

    # field-level F1
    for field in gt:
        p = normalize(pred.get(field))
        t = normalize(gt.get(field))

        if isinstance(t, list):
            p, t = set(p or []), set(t)
            tp = len(p & t)
            precision = tp / len(p) if p else 0
            recall = tp / len(t) if t else 0
        else:
            precision = recall = float(p == t)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        metrics["field_f1"][field].append(f1)

# ------------------ Report ------------------

print("\n=== EVALUATION RESULTS ===")
print(f"Samples: {metrics['total']}")
print(f"Valid JSON rate: {metrics['valid_json'] / metrics['total']:.2%}")
print(f"Exact match rate: {metrics['exact_match'] / metrics['total']:.2%}")

for field, scores in metrics["field_f1"].items():
    avg = sum(scores) / len(scores) if scores else 0
    print(f"{field} F1: {avg:.3f}")
