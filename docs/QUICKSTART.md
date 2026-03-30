# IIAS Quick Start Guide

**Version:** 1.0 (MVP)
**Last Updated:** 2026-03-27

---

## Welcome to IIAS

This guide will help you get started with the **Intelligent Item Authoring System (IIAS)** — a platform for generating, validating, and managing SAT test questions using fine-tuned language models.

**What you'll build:**
1. **Generation Service** — Two fine-tuned models (RW + Math) that generate SAT questions
2. **Auto-QA Service** — Automated quality checks that validate items
3. **Item Bank** — PostgreSQL database storing all items with lifecycle tracking

**Time to complete:** 2-3 days (assuming familiarity with Python and ML)

---

## Prerequisites

### Hardware

**For training (local validation):**
- Apple Silicon M4 (or better) with 8GB+ unified memory
  - Uses MLX for fast local validation
  - Or: NVIDIA GPU (RTX 3090/4090 or A100)

**For production training:**
- NVIDIA GPU with 40GB+ VRAM (A100 40GB or 80GB recommended)
  - Or: Use vast.ai GPU instances (~$1-2/hr)

### Software

**Required:**
- Python 3.11+
- PostgreSQL 15+
- Git
- CUDA 11.8+ (for NVIDIA GPUs)

**Python libraries:**
```bash
pip install torch transformers peft trl
pip install psycopg2-binary pydantic loguru
pip install textstat  # For readability checks
```

---

## Setup (30 minutes)

### 1. Clone Repository

```bash
cd ~/projects
git clone https://github.com/your-org/meridian.git
cd meridian
```

### 2. Set Up Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL

**On macOS:**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**On Ubuntu:**
```bash
sudo apt update
sudo apt install postgresql-15
sudo systemctl start postgresql
```

**Create database:**
```bash
sudo -u postgres psql
CREATE DATABASE meridian;
CREATE USER meridian_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE meridian TO meridian_user;
\q
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

**`.env` file:**
```bash
# Database
DATABASE_URL=postgresql://meridian_user:your_password@localhost:5432/meridian

# Models
RW_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
MATH_MODEL_ID=microsoft/phi-4

# Environment
APP_ENV=local  # or 'production'

# Logging
LOG_LEVEL=INFO
```

### 5. Initialize Database

```bash
python scripts/init_db.py
```

This creates all tables, indexes, and triggers.

---

## Phase 1: Data Preparation (1-2 hours)

### 1.1 Download Training Data

Your training data is the **ItemBank** from the `sat_synthetic_generator` project:

```bash
cp /Users/pradeep/projects/sat_synthetic_generator/data/itembank_questions_complete.json \
   data/raw/itembank_questions_complete.json
```

### 1.2 Convert to Training Format

```bash
python scripts/convert_itembank.py \
  --source data/raw/itembank_questions_complete.json \
  --output data/training/
```

This creates:
- `data/training/rw_train.jsonl`
- `data/training/rw_val.jsonl`
- `data/training/math_train.jsonl`
- `data/training/math_val.jsonl`

### 1.3 Validate Training Data

```bash
python scripts/validate_training_data.py data/training/
```

Expected output:
```
✅ rw_train.jsonl: 1,700 examples, 100% valid
✅ rw_val.jsonl: 300 examples, 100% valid
✅ math_train.jsonl: 1,800 examples, 100% valid
✅ math_val.jsonl: 315 examples, 100% valid
```

---

## Phase 2: Train Models (2-4 hours)

### 2.1 Train RW Model (Qwen2.5-7B)

**Local (MLX on M4):**
```bash
APP_ENV=local python scripts/train_model.py \
  --section reading_writing \
  --data data/training/rw_train.jsonl \
  --val-data data/training/rw_val.jsonl \
  --epochs 3 \
  --output checkpoints/rw-sft-v1.0
```

**Production (CUDA on A100):**
```bash
APP_ENV=production python scripts/train_model.py \
  --section reading_writing \
  --data data/training/rw_train.jsonl \
  --val-data data/training/rw_val.jsonl \
  --epochs 3 \
  --output checkpoints/rw-sft-v1.0
```

**Expected duration:**
- Local (MLX): ~3 hours
- Production (CUDA A100): ~2 hours

**Monitor training:**
```bash
tail -f outputs/logs/train_2026-03-27.log
```

Look for:
- JSON validity rate increasing (should reach ≥90%)
- Loss decreasing
- No NaN values

### 2.2 Train Math Model (Phi-4)

```bash
APP_ENV=production python scripts/train_model.py \
  --section math \
  --data data/training/math_train.jsonl \
  --val-data data/training/math_val.jsonl \
  --epochs 3 \
  --output checkpoints/math-sft-v1.0
```

**Expected duration:**
- Production (CUDA A100): ~4.5 hours

### 2.3 Verify Checkpoints

```bash
python scripts/verify_checkpoint.py \
  --checkpoint checkpoints/rw-sft-v1.0 \
  --section reading_writing

python scripts/verify_checkpoint.py \
  --checkpoint checkpoints/math-sft-v1.0 \
  --section math
```

Expected output:
```
✅ Checkpoint valid
✅ Adapter weights loaded
✅ Config loaded
✅ Test generation successful
```

---

## Phase 3: Generate Items (10 minutes)

### 3.1 Generate Sample Items

**Generate 10 RW items:**
```bash
python scripts/generate_items.py \
  --section reading_writing \
  --domain information_and_ideas.central_ideas_and_details \
  --difficulty medium \
  --count 10 \
  --output data/generated/rw_sample.jsonl
```

**Generate 10 Math items:**
```bash
python scripts/generate_items.py \
  --section math \
  --domain algebra.quadratic_equations \
  --difficulty medium \
  --count 10 \
  --output data/generated/math_sample.jsonl
```

### 3.2 Inspect Generated Items

```bash
cat data/generated/rw_sample.jsonl | jq '.content_json.question' | head -5
```

You should see properly formatted SAT questions.

---

## Phase 4: Auto-QA Validation (5 minutes)

### 4.1 Run Auto-QA on Generated Items

```bash
python scripts/validate_items.py \
  --input data/generated/rw_sample.jsonl \
  --output data/validated/
```

Expected output:
```
Processing 10 items...
✅ Item 1: PASSED (score: 1.0)
✅ Item 2: PASSED (score: 1.0)
⚠️  Item 3: PASSED WITH WARNINGS (score: 1.0, flags: READABILITY_BELOW_RANGE)
❌ Item 4: FAILED (score: 0.0, errors: SCHEMA_VALIDATION_FAILED)
...
Summary:
- Total: 10
- Passed: 8
- Passed with warnings: 1
- Failed: 1
```

### 4.2 Review Validation Results

```bash
cat data/validated/validation_report.json | jq .
```

---

## Phase 5: Store in Item Bank (5 minutes)

### 5.1 Load Validated Items into Database

```bash
python scripts/load_items.py \
  --input data/validated/validated_items.jsonl \
  --status draft
```

### 5.2 Verify Items in Database

```bash
python scripts/query_items.py \
  --status draft \
  --section reading_writing \
  --count 5
```

Expected output:
```
Draft items (Reading & Writing):
1. [uuid-1] medium - information_and_ideas.central_ideas_and_details
2. [uuid-2] medium - information_and_ideas.central_ideas_and_details
...
```

---

## Phase 6: Review Workflow (Manual)

### 6.1 Review Draft Items

```bash
python scripts/review_items.py --status draft --section reading_writing
```

This displays items one by one for review.

### 6.2 Approve/Reject Items

**Approve an item:**
```bash
python scripts/approve_item.py \
  --item-id uuid-123 \
  --approve \
  --reviewer-id user-uuid
```

**Reject an item:**
```bash
python scripts/approve_item.py \
  --item-id uuid-456 \
  --reject \
  --reasons INCORRECT_ANSWER,POOR_DISTRACTOR_QUALITY \
  --notes "The correct answer is B, not C"
```

### 6.3 Check Item Status

```bash
python scripts/query_items.py --item-id uuid-123
```

---

## Testing the Full Pipeline

### End-to-End Test

```bash
# 1. Generate items
python scripts/generate_items.py \
  --section math \
  --domain algebra.linear_equations \
  --difficulty medium \
  --count 5

# 2. Validate items
python scripts/validate_items.py \
  --input data/generated/math_algebra_linear_equations_medium.jsonl

# 3. Load into database
python scripts/load_items.py \
  --input data/validated/validated_items.jsonl

# 4. Review items
python scripts/review_items.py --status draft

# 5. Approve items
python scripts/approve_item.py --item-id uuid-xxx --approve

# 6. Export approved items
python scripts/export_items.py \
  --status pretesting \
  --output exports/pretesting_items.json
```

---

## Common Workflows

### Generate Items for a Specific Domain

```bash
python scripts/generate_items.py \
  --section math \
  --domain geometry.trigonometry \
  --difficulty hard \
  --count 20
```

### Check Domain Coverage

```bash
python scripts/analytics.py --domain-coverage --section math
```

Output:
```
Domain Coverage (Math):
├── algebra.linear_equations_one_variable: 145 items (target: 150) ✅
├── algebra.linear_equations_two_variables: 132 items (target: 150) ⚠️
├── algebra.linear_functions: 89 items (target: 150) ❌ DEFICIT
...
```

### Export Items for Test Assembly

```bash
python scripts/export_items.py \
  --status operational \
  --section math \
  --domains algebra.linear_equations,geometry.trigonometry \
  --difficulty medium \
  --output exports/math_form_1.json
```

---

## Troubleshooting

### Problem: Training Fails with OOM Error

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Use QLoRA (4-bit quantization)
export LOAD_IN_4BIT=true

# Or reduce batch size
python scripts/train_model.py ... --batch-size 4
```

### Problem: Low JSON Validity Rate

**Symptom:** JSON validity rate < 50%

**Solutions:**
1. Check training data quality
2. Increase training epochs (try 5 instead of 3)
3. Lower temperature during generation (try 0.5 instead of 0.7)

### Problem: Database Connection Error

**Error:** `psycopg2.OperationalError: could not connect to server`

**Solution:**
```bash
# Check PostgreSQL is running
sudo service postgresql status

# Start PostgreSQL
sudo service postgresql start
```

---

## What's Next?

### Phase 2+ Features (Future)

- **REST API** — HTTP endpoints for generation and validation
- **Review UI** — Web interface for psychometricians
- **IRT Predictor** — Predict difficulty from content
- **Enhanced Auto-QA** — Similarity detection, bias screening
- **Pretesting Pipeline** — Administer items to students
- **IRT Calibration** — Empirical parameter estimation

### Learn More

- **Introduction:** [`docs/00-introduction.md`](./00-introduction.md)
- **Generation Service:** [`docs/01-generation-service.md`](./01-generation-service.md)
- **Auto-QA Service:** [`docs/02-auto-qa-service.md`](./02-auto-qa-service.md)
- **Item Bank:** [`docs/03-item-bank.md`](./03-item-bank.md)

---

## Summary

You've built:
- ✅ Two fine-tuned models (RW + Math)
- ✅ Auto-QA pipeline
- ✅ PostgreSQL Item Bank
- ✅ End-to-end generation workflow

**Time to next item:** ~30 seconds
**Cost per item:** ~$0.001 (self-hosted)
**Quality:** ~60-80% approval rate (improves with retraining)

**Congratulations! You now have a working SAT item generation platform.**

---

**Need help?**
- Check the documentation in `docs/`
- Review logs in `outputs/logs/`
- Open an issue on GitHub

**Document version:** 1.0
**Last updated:** 2026-03-27
