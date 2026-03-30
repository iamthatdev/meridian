# IIAS MVP Deployment Guide

**Last Updated:** 2026-03-28
**Status:** MVP Complete (13/13 tasks)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Training and Testing](#local-training-and-testing)
3. [Production Deployment](#production-deployment)
4. [Data Requirements](#data-requirements)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and set APP_ENV=local
```

### Verify Installation

```bash
# Run tests
pytest tests/test_config.py -v

# Expected: 14 tests PASSED
```

---

## Local Training and Testing

### Step 1: Initialize Database

```bash
# Start PostgreSQL
brew services start postgresql  # macOS
# or: sudo systemctl start postgresql  # Linux

# Create database
createdb meridian

# Run migrations
python scripts/init_db.py

# Expected: ✅ Database initialized successfully
```

### Step 2: Prepare Training Data

**IMPORTANT:** You need training data first (see [Data Requirements](#data-requirements)).

Once you have data in `data/raw/itembank_questions_complete.json`:

```bash
# Convert ItemBank format to training format
python scripts/convert_itembank.py \
  data/raw/itembank_questions_complete.json \
  data/training/

# Expected output:
# Converting data/raw/itembank_questions_complete.json...
# Loaded X items from ItemBank
# Separated Y RW items, Z Math items
# Split: RW train=A, val=B
# Split: Math train=C, val=D
# ✅ Conversion complete:
#   RW: A train, B val
#   Math: C train, D val
```

**Output files created:**
```
data/training/
├── rw_train.jsonl
├── rw_val.jsonl
├── math_train.jsonl
└── math_val.jsonl
```

### Step 3: Train Models Locally

**⚠️ IMPORTANT:** Local training uses a **proxy model** (Qwen2.5-1.5B-4bit) for **pipeline validation only**. It will NOT produce production-quality models.

```bash
# Train Reading & Writing model
APP_ENV=local python scripts/train_model.py \
  --section reading_writing \
  --data-path data/training \
  --output-dir checkpoints/reading_writing

# Train Math model
APP_ENV=local python scripts/train_model.py \
  --section math \
  --data-path data/training \
  --output-dir checkpoints/math
```

**Training output:**
```
Loading reading_writing model...
Loading dataset from data/training
Dataset: X train examples, Y val examples
Starting training...
Epoch 1 | Step 10/100 | Loss: 0.5234
Epoch 1 | Step 20/100 | Loss: 0.4891
...
Running validation...
Average validation loss: 0.3891
Saved best checkpoint (val_loss: 0.3891)
✅ Training complete
```

**Checkpoint structure:**
```
checkpoints/
├── reading_writing/
│   ├── 20260328_154039/
│   │   ├── checkpoint-step-100/
│   │   ├── epoch-1/
│   │   ├── epoch-2/
│   │   ├── best/          ← Best validation loss
│   │   ├── final/         ← End of training
│   │   └── training_metadata.json
│   └── latest -> 20260328_154039/
└── math/
    └── [same structure]
```

### Step 4: Test Model Generation

```bash
# Generate sample Math items
python scripts/generate_items.py \
  --checkpoint checkpoints/math/best \
  --section math \
  --domain algebra.linear_equations_one_variable \
  --difficulty medium \
  --items-per-domain 5 \
  --output data/generated/test_items.jsonl

# Expected:
# Loading model for math from checkpoints/math/best
# Model loaded for math
# Generating 5 math items: algebra.linear_equations_one_variable, medium
# ✅ Generated 5 items
# ✅ Saved 5 items to data/generated/test_items.jsonl
```

### Step 5: Validate Generated Items

```bash
# Run Auto-QA validation
python scripts/validate_items.py \
  --input data/generated/test_items.jsonl \
  --output data/validated/validated_items.jsonl \
  --verbose

# Expected:
# ============================================================
# Validation Summary
# ============================================================
# Total items:    5
# Passed:         4 (80.0%)
# Failed:         1
# ============================================================
# Top failure reasons:
#   QUESTION_TOO_SHORT: 1
# ============================================================
```

### Step 6: Load Items into Database

```bash
# Load validated items
python scripts/load_items.py \
  --input data/validated/validated_items.jsonl \
  --section math

# Expected:
# Loading items from data/validated/validated_items.jsonl
# Running Auto-QA validation...
# ✅ Loaded X items into Item Bank
```

### Step 7: Query and Review Items

```bash
# Query draft items
python scripts/query_items.py --status draft --limit 10

# Approve an item
python scripts/approve_item.py \
  --item-id <uuid-from-query> \
  --reviewer-id user-123 \
  --approve

# Query operational items
python scripts/query_items.py --status operational --section math
```

---

## Production Deployment

### Infrastructure Requirements

#### 1. GPU Server

**Minimum Specifications:**
- **GPU:** A100 40GB or RTX 4090 24GB
- **RAM:** 64GB+
- **Storage:** 100GB+ SSD
- **OS:** Ubuntu 22.04 LTS

**Recommended:**
- **GPU:** A100 80GB
- **RAM:** 128GB+
- **Storage:** 200GB+ NVMe SSD

**Cloud Options:**
- vast.ai (most cost-effective)
- AWS (p3/p4 instances)
- GCP (A2 instances)
- Lambda Labs

#### 2. Production Database

**Options:**
- AWS RDS for PostgreSQL
- Google Cloud SQL
- DigitalDB
- Self-hosted with replication

**Configuration:**
Update `configs/production.yaml`:
```yaml
database:
  url: postgresql://user:password@prod-db-host:5432/meridian
  pool_size: 10
  max_overflow: 20
```

#### 3. Model Storage

**Options:**
- AWS S3 + CloudFront
- Google Cloud Storage
- Azure Blob Storage
- Self-hosted MinIO

### Production Training Steps

```bash
# On production GPU server:

# 1. Set environment variables
export APP_ENV=production
export HF_TOKEN=your_huggingface_token  # For gated models
export DATABASE_URL=postgresql://user:pass@host:5432/meridian

# 2. Clone repository and install dependencies
git clone <repo-url>
cd meridian
pip install -r requirements.txt

# 3. Initialize production database
python scripts/init_db.py

# 4. Convert training data
python scripts/convert_itembank.py \
  /data/raw/itembank_questions_complete.json \
  /data/training/

# 5. Train production models
python scripts/train_model.py \
  --section math \
  --data-path /data/training \
  --output-dir /checkpoints/math

# 6. Validate checkpoints
# (Check training_metadata.json for validation metrics)

# 7. Upload checkpoints to model storage
# (Use AWS CLI, gsutil, or rclone)

# 8. Deploy generation API
# (See API deployment section below)
```

**Expected training time (A100 80GB):**
- Math (phi-4 14B): ~4-6 hours
- Reading & Writing (Qwen 7B): ~2-3 hours

### Production Deployment Checklist

#### Phase 1: Infrastructure Setup
- [ ] GPU server provisioned and accessible
- [ ] PostgreSQL database created and configured
- [ ] Model storage (S3/GCS) provisioned
- [ ] Firewall rules configured (allow necessary ports)
- [ ] DNS configured (if using custom domain)

#### Phase 2: Configuration
- [ ] Update `configs/production.yaml` with real values
- [ ] Set environment variables (`HF_TOKEN`, `DATABASE_URL`)
- [ ] Adjust batch sizes based on GPU VRAM
- [ ] Configure logging and monitoring

#### Phase 3: Data Preparation
- [ ] Obtain full ItemBank dataset
- [ ] Convert data to training format
- [ ] Verify data quality and balance
- [ ] Split into train/val/test sets

#### Phase 4: Model Training
- [ ] Train Reading & Writing model (production)
- [ ] Train Math model (production)
- [ ] Validate checkpoints on test set
- [ ] Upload checkpoints to storage
- [ ] Document training metrics

#### Phase 5: Deployment
- [ ] Deploy generation API (FastAPI/FastAPI)
- [ ] Set up load balancer (nginx/ALB)
- [ ] Configure SSL certificates
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation (ELK/Loki)

#### Phase 6: Testing
- [ ] Load test generation endpoints
- [ ] Validate Auto-QA pipeline
- [ ] Test database CRUD operations
- [ ] Run end-to-end integration tests
- [ ] Performance testing (latency, throughput)

#### Phase 7: Operations
- [ ] Set up automated backups
- [ ] Configure alerting (PagerDuty/Slack)
- [ ] Document runbooks
- [ ] Train operations team
- [ ] Set up CI/CD pipeline

---

## Data Requirements

### Data Format

The system expects training data in **ItemBank format**:

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "section": "math",
    "domain": "algebra.linear_equations_one_variable",
    "difficulty_tier": "medium",
    "content_json": {
      "passage": null,
      "question": "If 2x + 5 = 13, what is the value of x?",
      "math_format": "plain",
      "choices": [
        {"label": "A", "text": "4"},
        {"label": "B", "text": "8"},
        {"label": "C", "text": "9"},
        {"label": "D", "text": "18"}
      ],
      "correct_answer": "A",
      "correct_answer_text": "4",
      "rationale": "Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4."
    }
  }
]
```

### Per Section Requirements

**Minimum Viable Data:**
- **Per section:** 500 items
- **Total:** 1,000 items (500 RW + 500 Math)

**Recommended for Production:**
- **Per section:** 2,000 items
- **Total:** 4,000 items (2,000 RW + 2,000 Math)

**Ideal for Best Performance:**
- **Per section:** 5,000+ items
- **Total:** 10,000+ items

### Distribution Guidelines

**By Difficulty:**
- Easy: 20% (100 items per 500)
- Medium: 60% (300 items per 500)
- Hard: 20% (100 items per 500)

**By Domain (Reading & Writing - 11 domains):**
- Aim for ~180 items per domain (for 2,000 items)
- Minimum: 45 items per domain (for 500 items)

**By Domain (Math - 16 domains):**
- Aim for ~125 items per domain (for 2,000 items)
- Minimum: 30 items per domain (for 500 items)

**Example breakdown for 2,000 items per section:**

| Section | Easy | Medium | Hard | Total |
|---------|------|--------|------|-------|
| Reading & Writing | 400 | 1,200 | 400 | 2,000 |
| Math | 400 | 1,200 | 400 | 2,000 |
| **Total** | **800** | **2,400** | **800** | **4,000** |

### Data Sources

**Option 1: Existing SAT Item Bank (Recommended)**
- College Board Official SAT Practice Tests
- Licensed test prep content
- Internal item banks

**Option 2: Synthetic Data Generation**
- Use GPT-4/Claude to generate items
- Quality control with Auto-QA pipeline
- Human review of generated items

**Option 3: Public Datasets**
- Kaggle SAT datasets
- Open-source educational content
- Academic research datasets

**Option 4: Hybrid Approach**
- Start with 500 synthetic items
- Fine-tune initial model
- Generate more items with fine-tuned model
- Iteratively improve quality

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Error

```
Error: could not connect to server: Connection refused
```

**Solution:**
```bash
# Check PostgreSQL is running
brew services list | grep postgresql

# Start if not running
brew services start postgresql

# Verify connection
psql -h localhost -U your_user -d meridian
```

#### 2. Out of Memory During Training

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size in `configs/local.yaml` or `configs/production.yaml`
- Enable gradient checkpointing
- Use 4-bit quantization (already enabled by default)

```yaml
training:
  batch_size: 2  # Reduce from 4 or 8
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

#### 3. Model Loading Error

```
Error: Model requires HF_TOKEN
```

**Solution:**
```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Or add to .env file
echo "HF_TOKEN=your_token_here" >> .env
```

#### 4. Auto-QA Validation Failures

**Issue:** All items failing validation

**Solution:**
- Check data format matches ItemBank schema
- Verify all required fields are present
- Run with `--verbose` flag to see specific failures:
  ```bash
  python scripts/validate_items.py --input items.jsonl --verbose
  ```

#### 5. Generation Produces Invalid JSON

**Issue:** Generated items have malformed JSON

**Solution:**
- Checkpoint may be undertrained
- Generate with lower temperature:
  ```bash
  python scripts/generate_items.py \
    --checkpoint checkpoints/math/best \
    --section math \
    --domain algebra.linear_equations \
    --difficulty medium \
    --temperature 0.7  # Lower = more deterministic
  ```

### Debug Mode

Enable verbose logging:

```bash
# Set log level in .env
LOG_LEVEL=DEBUG

# Or pass directly
LOG_LEVEL=DEBUG python scripts/train_model.py --section math
```

### Getting Help

1. Check logs: `outputs/logs/`
2. Review training metadata: `checkpoints/*/training_metadata.json`
3. Run tests: `pytest tests/ -v`
4. Check documentation: `docs/CODE_WALKTHROUGH.md`

---

## Appendix

### File Structure Reference

```
meridian/
├── configs/
│   ├── local.yaml              # Local development config
│   └── production.yaml         # Production config
├── data/
│   ├── raw/                    # Source ItemBank data (gitignored)
│   │   └── itembank_questions_complete.json
│   ├── training/               # Training JSONL files
│   │   ├── rw_train.jsonl
│   │   ├── rw_val.jsonl
│   │   ├── math_train.jsonl
│   │   └── math_val.jsonl
│   ├── generated/              # Generated items (pre-validation)
│   └── validated/              # Validated items
├── checkpoints/                # Model checkpoints (gitignored)
│   ├── reading_writing/
│   │   └── 20260328_154039/
│   │       ├── best/
│   │       ├── final/
│   │       └── training_metadata.json
│   └── math/
│       └── [same structure]
├── outputs/
│   ├── logs/                   # Training logs
│   └── evals/                  # Evaluation results
└── scripts/
    ├── init_db.py              # Database initialization
    ├── convert_itembank.py     # Data conversion
    ├── train_model.py          # Training script
    ├── generate_items.py       # Generation script
    ├── validate_items.py       # Validation script
    ├── load_items.py           # Load to database
    ├── query_items.py          # Query database
    └── approve_item.py         # Approve/reject items
```

### Environment Variables

```bash
# Required
APP_ENV=local|production
DATABASE_URL=postgresql://user:pass@host:5432/meridian

# Optional (production)
HF_TOKEN=your_huggingface_token
LOG_LEVEL=INFO|DEBUG|WARNING
```

### Model IDs

**Local (Proxy):**
- Not applicable - uses whatever model is specified in config

**Production:**
- **Reading & Writing:** `Qwen/Qwen2.5-7B-Instruct`
- **Math:** `microsoft/phi-4`
- **Fallback:** `meta-llama/Llama-3.1-8B-Instruct`

---

**Document Version:** 1.0
**Last Updated:** 2026-03-28
