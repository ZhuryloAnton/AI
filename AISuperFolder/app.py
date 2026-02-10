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
    torch_dtype=torch.float32
).to(device)


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
ARTEM ANTONENKO
Head of Engineering | CTO | Engineering Executive
+352 661 367 192 | antonenko.artem@gmail.com | linkedin.com/in/artem-antonenko-al | artem-antonenko.com
Luxembourg | EU Citizen | Open to relocation (EU / Global)

EXECUTIVE SUMMARY
Head of Engineering with 18+ years of experience scaling engineering organizations, modernizing delivery, and
driving AI-first transformation across large enterprise portfolios. Led 100+ engineers and delivered platforms
serving 50M+ daily users. Spearheaded AI adoption across 12 companies, enabling 500+ developers with agentic
coding assistants, RAG knowledge systems, and AI-augmented SDLC practices, achieving 80% coding-time
reduction and 2,000% ROI. Expert in engineering operations, distributed systems, cloud architecture, AI-enabled
SDLC, and quality engineering. Builds resilient, data-driven delivery organizations and partners with executives to
accelerate product velocity, improve reliability, and maximize AI investment impact.

CORE SKILLS
Leadership & Management: Team Scaling (6→100+ engineers), Cross-functional Leadership, Stakeholder
Management, Agile Transformation, Delivery Culture Building, Performance Management, Technical Strategy,
Architecture Governance, Budget Ownership, Security & Compliance Oversight (OWASP, GDPR)
Programming & Development: Java, PHP, Python, C++, JavaScript; Microservices, Domain-Driven Design
(DDD), REST APIs, Distributed systems; Symfony, Spring Boot, Laravel, Phalcon, FastAPI, React, Angular
Infrastructure & DevOps: AWS, Azure, DigitalOcean; Docker, Kubernetes, Helm, Jenkins, GitLab CI, CircleCI,
Trunk-based Development; ELK Stack, Grafana, Performance Tuning
Data & Messaging: MongoDB, MySQL, Redis, PostgreSQL; Kafka, RabbitMQ; ETL, Vector DBs
AI & Transformation: RAG-based AI, Agentic AI, AI Governance; Windsurf, Cursor, Claude Code, GitHub Copilot,
Codex, Gemini CLI; Training Programs, ROI Optimization, Productivity Enhancement
Quality & Testing: TDD, ATDD, BDD, Shift-left Testing; Test Framework Development, Quality Engineering
Business & Strategy: ACSPO; Digital Transformation, GDPR Compliance; Brain-Based Conversation Skills®

EXECUTIVE EXPERIENCE
Head of Engineering & AI Transformation - Byborg Enterprises – Luxembourg (Jul 2019 - Dec 2025)
Byborg Enterprises is a multinational technology group of 12 companies operating in media, entertainment, and
digital platforms, leveraging AI to enhance productivity and power global live-streaming and creator ecosystems.
● Delivered enterprise-wide AI transformation across 12 companies, enabling 500+ engineers to use agentic
  coding assistants, RAG systems, and LLM-augmented SDLC workflows; drove 86% adoption, 80%
  coding-time reduction, 5–8× delivery acceleration, and 2,000% ROI.
● Standardized AI-driven engineering workflows (AI code review, architecture generation, automated
  documentation), boosting throughput and reducing defects by 30–40%.
● Launched RAG-powered internal knowledge hubs, cutting requirement-preparation time by 20% and
  increasing consistency in solution design across engineering.
● Implemented AI workflow automation (face/object recognition, text-to-SQL analytics), reducing moderation
  workload by 60% and BI workload by 40%, accelerating operational decision-making.
● Led architecture and AI design for Jasmin.ai, mentoring teams on RAG and prompt engineering, resulting
  in 1,000+ active users in the first month.
● Scaled engineering from 6 → 100+, establishing hiring frameworks, onboarding systems, and performance
  processes that improved retention by 15% and tripled successful-hire rate.
● Strengthened delivery operations via data-driven governance and iterative planning, improving release
  predictability by 25% and enabling higher cross-team velocity.
● Built a leadership pipeline of 18 emerging leaders, reducing dependencies on senior management and
  improving distributed ownership.
● Implemented architecture–roadmap governance, aligning innovation, technical debt, and product
  milestones to ensure continuous engineering output.
● Delivered a monolith → microservices migration, improving page-load performance by 40%, enabling 30+
  parallel projects, and achieving 90% estimation accuracy using ML-based forecasting.
● Developed fractal architecture, enabling feature extraction in <5 days and accelerating capability delivery.
● Established TDD, ATDD, and shift-left QA, achieving a sustained zero-bug backlog and eliminating
  recurring production incidents.
● Architected high-scale systems for products serving 50M+ daily users, ensuring 400–500 RPS reliability
  across 30+ microservices.

Engineering Manager – BBH Stockholm (formerly REMIT), Stockholm, Sweden (Oct 2017 - Jul 2019)
● Architected Panini’s eCommerce platform (Java, PHP, Kafka, AWS) for 30+ restaurants
● Served as Lead Architect for MyVolvo
● Delivered site-creation pipeline for Hilton and Sheraton
● Led distributed backend teams across 9 countries

Senior Web Developer – Oracle, Ukraine (May 2016 - Nov 2017)
● Architected and optimized Oracle Field Service Cloud applications
● Embedded TDD and built custom API-testing frameworks

EDUCATION & CERTIFICATIONS
AI Strategy, MIT Sloan Executive Education - 2025
Agentic AI, DeepLearning.AI (Andrew Ng), RAG - 2025
Software Design Masterclass (Eric Evans, DDD Academy) - 2024
MSc, Aircraft Engineering – National Aerospace University (Ukraine)

LANGUAGES
English (Proficient) | Ukrainian, Russian (Native) | French, Luxembourgish (Basic)
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
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id
    )


# =====================
# DECODE
# ❌ common mistake: printing full prompt + answer
# =====================
generated_tokens = output[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(result)
