"""
Test script: Phi-4-mini-instruct + rmtlabs/phi-4-mini-adapter-v1 with a simple CV parsing example.
Usage: python test_cv_parser.py
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Config ---
BASE_MODEL = "microsoft/Phi-4-mini-instruct"
ADAPTER_ID = "rmtlabs/phi-4-mini-adapter-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3

# --- Sample CV ---
SAMPLE_CV = """
John Smith
Email: john.smith@email.com | Phone: +1 555-123-4567 | Location: San Francisco, CA

PROFESSIONAL SUMMARY
Software engineer with 5+ years of experience in Python, ML pipelines, and cloud services. Led migration to microservices at TechCorp.

EDUCATION
- B.S. Computer Science, Stanford University, 2018
- M.S. Machine Learning, MIT, 2020

EXPERIENCE
- Senior Software Engineer, TechCorp (2021–Present)
  Built recommendation APIs, reduced latency by 40%. Tech: Python, AWS, Kubernetes.
- Software Engineer, StartupXYZ (2018–2021)
  Backend services and data pipelines. Tech: Python, PostgreSQL, Redis.

SKILLS
Python, Java, SQL, AWS, Docker, Kubernetes, TensorFlow, PyTorch, REST APIs, Agile.

CERTIFICATIONS
AWS Solutions Architect Associate (2022)
"""


def build_cv_prompt(cv_text: str) -> str:
    return (
        "You are a CV/resume parser. Extract information from the CV below and respond with valid JSON only. "
        "Use this structure (use empty arrays/strings if not found):\n"
        '{"name":"","email":"","phone":"","location":"","summary":"","skills":[],'
        '"experience":[{"title":"","company":"","period":"","description":""}],'
        '"education":[{"degree":"","institution":"","year":""}],"certifications":[]}\n\n'
        "CV:\n---\n"
        + cv_text.strip()
        + "\n---\nRespond with only the JSON object, no other text."
    )


def extract_json(text: str) -> str:
    text = text.strip()

    # Unwrap markdown code block if present
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find the first { (start of JSON object)
    start = text.find("{")
    if start == -1:
        return text

    # Find the matching } by counting braces
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model.eval()

    prompt = build_cv_prompt(SAMPLE_CV)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    json_str = extract_json(response)

    try:
        result = json.loads(json_str)
        print("\n--- Parsed JSON ---\n")
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print("\n--- Raw response (JSON parse failed) ---\n")
        print(response)


if __name__ == "__main__":
    main()