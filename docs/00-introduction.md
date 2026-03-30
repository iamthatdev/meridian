# IIAS (Intelligent Item Authoring System) - Complete Guide

**Version:** 1.0 (MVP)
**Last Updated:** 2026-03-27
**Status:** Draft

---

## Table of Contents

1. [What is IIAS?](#what-is-iias)
2. [Why Do We Need IIAS?](#why-do-we-need-iias)
3. [System Overview](#system-overview)
4. [Core Concepts](#core-concepts)
5. [Component Architecture](#component-architecture)
6. [Technology Stack](#technology-stack)
7. [Development Workflow](#development-workflow)
8. [Next Steps](#next-steps)

---

## What is IIAS?

**IIAS (Intelligent Item Authoring System)** is a machine learning platform that automatically generates, validates, and manages SAT-style test questions at scale.

### The Problem IIAS Solves

Creating high-quality test questions is **expensive and slow**:

- A single psychometrician-authored SAT question takes **2-8 hours** to write
- Each question requires: domain expertise, alignment to College Board standards, bias review, field testing, and statistical calibration
- Scaling to thousands of questions for adaptive testing is **economically challenging** with human-only authoring

### The IIAS Solution

IIAS uses **fine-tuned large language models (LLMs)** to:

1. **Generate** SAT Reading & Writing and Math questions in JSON format
2. **Validate** each question through automated quality checks
3. **Store** questions in a database with lifecycle tracking
4. **Enable** human psychometricians to review and approve questions efficiently

**Key insight:** IIAS doesn't replace psychometricians — it shifts their work from **authoring** to **reviewing and curating**, a much higher-leverage activity.

---

## Why Do We Need IIAS?

### Traditional Test Item Development

```
Human Psychometrician writes item
        ↓
   Internal review (1-2 colleagues)
        ↓
   Bias and sensitivity review
        ↓
   Field testing (administer to students)
        ↓
   Statistical analysis (IRT calibration)
        ↓
   Operational use
```

**Timeline:** 4-8 weeks per item
**Cost:** $500-2000 per item

### IIAS-Powered Development

```
Training data (existing SAT items)
        ↓
   Fine-tune LLM (one-time, 2-3 days)
        ↓
   Generate 100 candidate items (minutes)
        ↓
   Auto-QA filters (seconds)
        ↓
   Human review 50-100 items (hours)
        ↓
   Field testing (same as traditional)
        ↓
   Operational use
```

**Timeline:** 1-2 weeks per batch of 100 items
**Cost:** $50-200 per item (after initial model training)

### The Economic Case

**Traditional approach:**
- 1,000 items × $1,000/item = **$1,000,000**

**IIAS approach:**
- Model training: $50,000 (one-time)
- Compute for inference: $5,000
- Human review: 1,000 items × $50/item = $50,000
- **Total: $105,000** (90% cost reduction)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IIAS MVP System                          │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ Training Data    │
    │ (4,101 existing  │
    │  SAT items)      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Generation       │
    │ Service          │
    │                  │
    │ • RW Model       │
    │ • Math Model     │
    └────────┬─────────┘
             │
             │ (generates items)
             ▼
    ┌──────────────────┐
    │ Auto-QA          │
    │ Service          │
    │                  │
    │ • Schema Check   │
    │ • Readability    │
    │ • Quality Rules  │
    └────────┬─────────┘
             │
             │ (validates items)
             ▼
    ┌──────────────────┐
    │ Item Bank        │
    │ (PostgreSQL)     │
    │                  │
    │ • draft          │
    │ • pretesting     │
    │ • operational    │
    │ • retired        │
    └──────────────────┘
```

### What IIAS Produces

**Output:** JSON-formatted SAT questions

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "section": "math",
  "domain": "algebra.quadratic_equations",
  "difficulty": "hard",
  "content_json": {
    "passage": null,
    "question": "If $x^2 - 5x + 6 = 0$, which of the following gives all possible values of $x$?",
    "math_format": "latex",
    "choices": [
      {"label": "A", "text": "$x = 2$ only"},
      {"label": "B", "text": "$x = 3$ only"},
      {"label": "C", "text": "$x = 2$ and $x = 3$"},
      {"label": "D", "text": "$x = -2$ and $x = -3$"}
    ],
    "correct_answer": "C",
    "correct_answer_text": "$x = 2$ and $x = 3$",
    "rationale": "Factoring gives $(x-2)(x-3)=0$, so $x=2$ or $x=3$.",
    "solution_steps": "1. Write the equation: $x^2 - 5x + 6 = 0$\\n2. Factor: $(x-2)(x-3) = 0$\\n3. Set each factor to zero: $x-2=0$ or $x-3=0$\\n4. Solve: $x=2$ or $x=3$"
  }
}
```

---

## Core Concepts

### 1. Fine-Tuning

**Definition:** Taking a pre-trained LLM and training it further on a specific dataset to specialize its behavior.

**Why we need it:** Base LLMs (GPT-4, Claude, etc.) are general-purpose. They can generate SAT questions, but not consistently in the exact format, difficulty level, and domain alignment we need. Fine-tuning teaches the model to follow SAT-specific patterns.

**Example:**
```
Base model: "Write a math question"
→ Could produce anything (arithmetic, calculus, word problems, etc.)

Fine-tuned model: "Generate a SAT Math item, domain: algebra.quadratic_equations, difficulty: hard"
→ Produces JSON output matching SAT format with LaTeX math
```

### 2. LoRA (Low-Rank Adaptation)

**Definition:** A parameter-efficient fine-tuning method that trains small adapter matrices on top of frozen base model weights.

**Why we use it:**
- **Efficient:** Trains 1-3% of parameters instead of 100%
- **Fast:** Reduces training time from days to hours
- **Safe:** Easy to rollback if something goes wrong
- **Cost-effective:** Runs on fewer GPUs

**Technical explanation:**
```
Base model: 7 billion parameters (Qwen2.5-7B)
Full fine-tuning: Train all 7B parameters
LoRA fine-tuning: Train ~200M parameters (rank 32 adapters)

Result: 98% reduction in trainable parameters
```

### 3. Item Response Theory (IRT)

**Definition:** A psychometric framework that models the relationship between:
- **Student ability** (θ, theta) — latent skill level
- **Item difficulty** (b parameter) — how hard the question is
- **Item discrimination** (a parameter) — how well it distinguishes ability levels
- **Guessing parameter** (c) — probability of random correct guess

**Why it matters:**
- Without IRT: "This feels like a hard question"
- With IRT: "This question has b=1.2, meaning only 12% of students (θ=0) get it right"

**For MVP:** We use **seeded placeholder values** (irt_a=1.0, irt_b=0.0, irt_c=0.25) because we don't have real student response data yet.

**Future (Phase 2+):** After pretesting items to real students, we'll calculate empirical IRT parameters.

### 4. Item Lifecycle States

**Items move through defined stages:**

```
draft → pretesting → operational → retired
```

**Explanation:**

- **`draft`**: Initial state after generation. Item has passed auto-QA but not yet reviewed by humans. Cannot be administered to students.

- **`pretesting`**: Item has been approved by a psychometrician and is eligible for field testing in non-scored positions. We collect student responses to calibrate IRT parameters.

- **`operational`**: Item has been field tested, has calibrated IRT parameters, and is ready for use in actual scored test forms. This is the "gold standard" state.

- **`retired`**: Item is removed from use. Reasons: failed quality gates, detected bias, showed IRT parameter drift, or psychometrician flagged it.

**State transitions are guarded:**
- `draft → pretesting`: Requires human approval
- `pretesting → operational`: Requires N ≥ 500 responses AND passed calibration
- `operational → retired`: Quality gate failure OR manual retirement

### 5. Schema Validation

**Definition:** Checking that generated JSON conforms to the expected structure and data types.

**Why it's critical:** LLMs can generate invalid JSON (mismatched braces, wrong types, missing fields). Schema validation catches these before they corrupt the database.

**Example of schema violations:**
```json
// ❌ WRONG: Missing required field
{
  "section": "math",
  "domain": "algebra"
  // Missing: difficulty, content_json
}

// ❌ WRONG: Wrong enum value
{
  "section": "math",
  "difficulty": "extremely_hard"  // Should be: easy, medium, hard
}

// ❌ WRONG: Mismatched braces
{
  "choices": [
    {"label": "A", "text": "..."}
  // Missing closing bracket for array
}
```

### 6. Auto-QA (Automated Quality Assurance)

**Definition:** A pipeline of automated checks that assess item quality before human review.

**What it checks:**
1. **Schema validity:** Is the JSON well-formed and complete?
2. **Readability:** Is the passage at appropriate grade level (9-12)?
3. **Basic quality:** Are there duplicate choices? Trivial distractors? Too-short explanations?
4. **Math verification:** (Future) Does the math check out symbolically?

**What it produces:**
```json
{
  "auto_qa_passed": true,
  "qa_score": 0.92,
  "qa_flags": []
}
```

**Items that fail auto-QA are rejected immediately** — humans never see them. This saves reviewer time.

---

## Component Architecture

### Component 1: Generation Service

**Purpose:** Train and host fine-tuned models that generate SAT questions

**Two separate models:**
1. **RW Model:** Qwen2.5-7B-Instruct (fine-tuned for Reading & Writing)
2. **Math Model:** Phi-4 (fine-tuned for Math with LaTeX)

**Inputs:**
- Training data (existing SAT items in chat template format)
- Generation prompts (section, domain, difficulty, topic)

**Outputs:**
- JSON-formatted SAT questions
- Model checkpoints (saved LoRA adapters)

**Key technologies:**
- PyTorch, HuggingFace Transformers, PEFT, TRL
- QLoRA (4-bit quantization for efficient training)
- vLLM (for fast inference)

**See:** [`docs/01-generation-service.md`](./01-generation-service.md) for complete details.

---

### Component 2: Auto-QA Service

**Purpose:** Validate generated items through automated quality checks

**Three validation stages:**
1. **Schema validation:** Hard gate — rejects malformed JSON
2. **Readability check:** Flesch-Kincaid grade level for RW passages
3. **Basic quality rules:** Duplicate detection, length checks, trivial distractor detection

**Inputs:**
- Generated items (from Generation Service)

**Outputs:**
- Validation results (pass/fail, score, flags)
- Passed items → Item Bank
- Failed items → Rejection log (for retraining signal)

**Key technologies:**
- Pydantic (schema validation)
- Flesch-Kincaid (readability scoring)
- Custom rule engine

**See:** [`docs/02-auto-qa-service.md`](./02-auto-qa-service.md) for complete details.

---

### Component 3: Item Bank

**Purpose:** PostgreSQL database storing all items with lifecycle tracking

**Key tables:**
- `items` — Main item records
- `domains` — Controlled taxonomy vocabulary
- `calibration_log` — IRT calibration history (future)
- `review_records` — Human review decisions

**Item lifecycle:**
```
draft → pretesting → operational → retired
```

**Inputs:**
- Validated items (from Auto-QA Service)
- Human review decisions
- Calibration results (future)

**Outputs:**
- Queries for item retrieval
- Exports for test assembly
- Analytics for domain coverage

**Key technologies:**
- PostgreSQL (relational database)
- JSONB (JSON storage in Postgres)
- SQL indexes for fast queries

**See:** [`docs/03-item-bank.md`](./03-item-bank.md) for complete details.

---

## Technology Stack

### Programming Languages

- **Python 3.11+** — Primary language for all components
- **SQL** — Database queries and schema management

### Core Libraries

**For Generation Service:**
- `transformers` (HuggingFace) — Model loading and training
- `peft` — LoRA/QLoRA fine-tuning
- `trl` — Reinforcement learning utilities
- `torch` — PyTorch deep learning framework
- `vllm` — Fast inference serving (future)

**For Auto-QA Service:**
- `pydantic` — Schema validation
- `textstat` — Readability scoring (Flesch-Kincaid)

**For Item Bank:**
- `psycopg2` — PostgreSQL adapter for Python
- `sqlalchemy` — ORM and query building (optional)
- `alembic` — Database migrations (optional for MVP)

**For All Components:**
- `pydantic` — Data validation and settings management
- `loguru` — Structured logging
- `pytest` — Testing framework
- `python-dotenv` — Environment configuration

### Infrastructure

**Training:**
- **Local:** Apple Silicon M4 with MLX (for pipeline validation)
- **Production:** vast.ai GPU instances (A100 80GB or 40GB)

**Storage:**
- **PostgreSQL 15+** — Item bank and all metadata
- **Filesystem** — Model checkpoints, training logs

**Compute:**
- **Training:** CUDA GPUs (NVIDIA A100, RTX 4090, etc.)
- **Inference:** CUDA GPUs or CPU (slower)

---

## Development Workflow

### Phase 1: Foundation (Current MVP)

**Week 1-2: Data Preparation**
- Convert ItemBank data to training format
- Create training/validation splits
- Set up Pydantic schemas

**Week 3-4: Generation Service**
- Implement training pipeline
- Fine-tune RW model (Qwen2.5-7B)
- Fine-tune Math model (Phi-4)
- Test generation quality

**Week 5-6: Auto-QA Service**
- Implement schema validation
- Add readability checks
- Add basic quality rules
- Test on generated items

**Week 7-8: Item Bank**
- Design PostgreSQL schema
- Implement database layer
- Create CLI scripts for review workflow
- End-to-end testing

**Deliverables:**
- ✅ Two trained models (RW + Math)
- ✅ Auto-QA pipeline
- ✅ PostgreSQL Item Bank
- ✅ CLI scripts for generation and review
- ✅ Documentation (this guide)

---

### Phase 2: Production Integration (Future)

**Additions:**
- REST API (`/v1/generate_item`, etc.)
- Review UI (web interface for psychometricians)
- Enhanced Auto-QA (similarity detection, bias screening)
- IRT predictor (predict difficulty from content)
- Observability (metrics, dashboards, alerting)

---

### Phase 3: IRT Calibration (Future)

**Prerequisites:**
- Pretesting pipeline (administer items to students)
- Response data collection (N ≥ 500 per item)
- Anchor items for equating

**Additions:**
- Calibration engine (MAP/MMLE estimation)
- Equating framework (common theta scale)
- DIF analysis (differential item functioning)
- Scale stability monitoring

---

## Next Steps

### For Developers

1. **Read the component documentation:**
   - [`docs/01-generation-service.md`](./01-generation-service.md)
   - [`docs/02-auto-qa-service.md`](./02-auto-qa-service.md)
   - [`docs/03-item-bank.md`](./03-item-bank.md)

2. **Set up the development environment:**
   - Install Python 3.11+
   - Install dependencies: `pip install -r requirements.txt`
   - Set up PostgreSQL locally
   - Configure `.env` file

3. **Run the training pipeline:**
   - Prepare training data
   - Train RW model
   - Train Math model
   - Test generation

4. **Implement Auto-QA:**
   - Build schema validator
   - Add readability checks
   - Test on generated items

5. **Set up Item Bank:**
   - Create database schema
   - Implement database layer
   - Test CRUD operations

### For Project Managers

1. **Review the architecture:**
   - Understand the three components
   - Review data flow diagram
   - Validate against requirements

2. **Plan resources:**
   - GPU compute for training
   - PostgreSQL hosting
   - Development timeline

3. **Define success metrics:**
   - JSON validity rate ≥ 90%
   - Auto-QA pass rate ≥ 60%
   - Human approval rate ≥ 40%

---

## Glossary

**Auto-QA:** Automated Quality Assurance — pipeline of automated checks on generated items.

**Draft:** Initial lifecycle state for items that have passed auto-QA but not yet human-reviewed.

**Fine-tuning:** Training a pre-trained LLM on a specific dataset to specialize its behavior.

**IRT:** Item Response Theory — psychometric framework for modeling student ability and item parameters.

**Item Bank:** Database storing all generated items with lifecycle tracking.

**LoRA:** Low-Rank Adaptation — parameter-efficient fine-tuning method.

**MVP:** Minimum Viable Product — initial version with core functionality.

**Operational:** Lifecycle state for items that are calibrated and ready for scored test use.

**Pretesting:** Lifecycle state for items approved for field testing in non-scored positions.

**QLoRA:** Quantized LoRA — LoRA applied to 4-bit quantized base models.

**Retired:** Terminal lifecycle state for items removed from use.

**RW:** Reading & Writing section of the SAT.

**Schema:** Structure and data type definitions for item JSON.

---

## Appendix: Quick Reference

### File Structure

```
meridian/
├── docs/                          # This documentation
│   ├── 00-introduction.md         # This file
│   ├── 01-generation-service.md   # Generation Service guide
│   ├── 02-auto-qa-service.md      # Auto-QA Service guide
│   └── 03-item-bank.md            # Item Bank guide
├── src/
│   ├── generation/                # Generation Service code
│   ├── auto_qa/                   # Auto-QA Service code
│   └── item_bank/                 # Item Bank code
├── scripts/                       # CLI scripts
│   ├── train_model.py             # Train models
│   ├── generate_items.py          # Generate items
│   ├── review_items.py            # Review draft items
│   └── approve_item.py            # Approve/reject items
├── data/
│   ├── raw/                       # Source data (ItemBank)
│   ├── training/                  # Training JSONL files
│   └── generated/                 # Generated items (pre-QA)
├── checkpoints/                   # Saved model checkpoints
└── configs/                       # Configuration files
    ├── local.yaml                 # Local environment config
    └── production.yaml            # Production environment config
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/meridian

# Model Configuration
RW_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
MATH_MODEL_ID=microsoft/phi-4

# Training
LEARNING_RATE=2e-5
BATCH_SIZE=32
NUM_EPOCHS=3

# Logging
LOG_LEVEL=INFO
```

### Common Commands

```bash
# Train models
python scripts/train_model.py --section rw

# Generate items
python scripts/generate_items.py --section math --domain algebra --count 10

# Review items
python scripts/review_items.py --status draft

# Approve item
python scripts/approve_item.py --item-id uuid-123 --approve

# Export items
python scripts/export_items.py --status operational --output items.json
```

---

**Document version:** 1.0
**Last updated:** 2026-03-27
**Author:** IIAS Development Team
**License:** Internal Use Only
