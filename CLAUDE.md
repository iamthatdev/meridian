# CLAUDE.md — Meridian

This file is the single source of truth for Claude when working in this codebase.
Read it fully before making any changes. Do not hallucinate file paths, configs, or
model names — everything canonical is defined here.

---

## Project Overview

**Meridian** is a fine-tuning pipeline for training SAT-prep tutoring models. It targets
two distinct skill domains — Reading & Writing (RW) and Math — each with its own base
model, data schema, and evaluation rubric.

The pipeline runs in two environments:

| Environment | Backend | Purpose |
|---|---|---|
| `local` | Apple Silicon (MLX) | Pipeline validation on a small proxy model |
| `production` | RunPod (CUDA) | Full-scale fine-tuning on production models |

The `APP_ENV` environment variable (set in `.env`) controls which config, model, and
training backend is active at any given time. **Never hardcode model names or paths.**
Always read them from the active config.

**Note:** MLT automation for RunPod deployment is in `~/mlt_automation/`, not in this repository.

---

## Directory Structure

```
meridian/
├── configs/
│   ├── production.yaml         # RunPod config (production models, CUDA)
│   └── models/                 # Model-specific configs
│
├── data/
│   ├── raw/                    # Untouched source data (gitignored)
│   ├── generated/              # LLM-generated examples pre-validation (gitignored)
│   ├── validated/              # Examples that passed auto-QA
│   └── splits/                 # train / val / test JSONL splits
│
├── src/
│   ├── data/
│   │   ├── generator.py        # Prompt construction + LLM call for data gen
│   │   ├── validator.py        # Auto-QA: schema check, rubric scoring, dedup
│   │   └── pipeline.py         # Orchestrates generator → validator → splits
│   │
│   ├── models/
│   │   ├── base.py             # ModelConfig dataclass, config loader
│   │   ├── rw.py               # RW model class (inherits base)
│   │   └── math.py             # Math model class (inherits base)
│   │
│   ├── training/
│   │   ├── trainer.py          # Abstract base trainer + factory function
│   │   ├── mlx_trainer.py      # MLX implementation (local only)
│   │   └── cuda_trainer.py     # CUDA / HuggingFace Trainer implementation
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, F1, rubric score aggregation
│   │   └── auto_qa.py          # Per-example QA checks (called during data gen)
│   │
│   └── api/
│       ├── server.py           # FastAPI app factory
│       └── routes.py           # Inference endpoints
│
├── scripts/
│   ├── generate_data.py        # CLI: generate + validate a dataset
│   ├── train_model.py          # Legacy training script (has bugs)
│   ├── train_huggingface.py    # HuggingFace Trainer script (recommended)
│   ├── evaluate.py             # CLI: run eval against a checkpoint
│   └── deploy.py               # CLI: push checkpoint to serving endpoint
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_evaluation.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── outputs/                    # Gitignored
│   ├── checkpoints/
│   ├── logs/
│   └── evals/
│
├── .env                        # Never committed
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── CLAUDE.md                   # This file
```

---

## Environment Setup

### Local (M4 MacBook Air)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set APP_ENV=local in .env
```

Required packages: `mlx`, `mlx-lm`, `transformers`, `datasets`, `pydantic`,
`fastapi`, `uvicorn`, `loguru`, `ruff`, `mypy`, `pytest`

### Production (RunPod)

```bash
# On the RunPod instance
pip install -r requirements.txt
# Set APP_ENV=production and HF_TOKEN in .env
```

**RunPod Deployment:** Use the MLT automation at `~/mlt_automation/` for GPU discovery, instance creation, and training deployment. See MLT automation docs for details.

Required packages: same as local minus `mlx` and `mlx-lm`, plus `torch`,
`bitsandbytes`, `trl`, `peft`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `APP_ENV` | ✅ | `local` or `production` |
| `HF_TOKEN` | ✅ (prod) | HuggingFace token for gated models |
| `RUNPOD_API_KEY` | ✅ (prod) | RunPod API key (set in `~/mlt_automation/.env`) |
| `LOG_LEVEL` | ❌ | Defaults to `INFO` |
| `FORCE_MODEL` | ❌ | Override model ID for one-off debugging only |

Never commit `.env`. All secrets live there and nowhere else.

---

## Config Schema

Configs use YAML. `configs/production.yaml` contains the production configuration:

```yaml
app_env: production

models:
  rw_model_id: <hf model id>
  math_model_id: <hf model id>
  fallback_model_id: <hf model id>

lora:
  r: int
  alpha: int
  dropout: float

quantization:
  load_in_4bit: bool
  bnb_4bit_quant_type: str
  bnb_4bit_compute_dtype: str
  gradient_checkpointing: bool

training:
  learning_rate: float
  batch_size: int
  gradient_accumulation_steps: int
  num_epochs: int
  max_seq_length_rw: int
  max_seq_length_math: int
  warmup_ratio: float

paths:
  data_dir: /root/meridian/data
  training_dir: /root/meridian/data/training
  generated_dir: /root/meridian/data/generated
  validated_dir: /root/meridian/data/validated
  checkpoint_dir: /root/meridian/checkpoints
  log_dir: /root/meridian/logs
```

Config is loaded via `src/config.py:load_config()`. Never read YAML directly in feature code.

---

## Model Choices

This section documents *why* each model was chosen, not just which model is active.
When evaluating a swap or upgrade, consult the criteria below before touching any config.

---

### Decision Criteria

Models were selected against four axes:

1. **Instruction following** — the model must reliably produce structured JSON output
   without post-processing hacks. Non-negotiable for the data pipeline.
2. **Domain reasoning** — RW requires nuanced rhetorical and grammatical judgment;
   Math requires reliable multi-step symbolic reasoning at SAT hard difficulty.
3. **VRAM budget** — production runs on a single vast.ai GPU. Models above 14B at
   full precision are out of scope unless quantized to 4-bit with no accuracy loss.
4. **License** — must permit fine-tuning and commercial deployment.

---

### Local Proxy Model

```
mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

**Why this model:**
- Fits entirely in 8GB unified memory on an M4 MacBook Air with headroom for the OS.
- The MLX community quantized version is the only reliable 4-bit Qwen2.5 build that
  runs without modification under `mlx-lm`.
- Qwen2.5 at any size has strong instruction-following relative to its parameter count,
  which means pipeline and schema enforcement tests are meaningful even at 1.5B — bad
  outputs are more likely a pipeline bug than a model limitation.

**What this model is NOT for:**
- Output quality. A 1.5B model will produce weak explanations and occasionally wrong
  answers. This is expected and irrelevant.
- Benchmarking. Never compare proxy model eval results to production targets.

**When to swap:**
- If `mlx-lm` drops support for this quantization format.
- If a `Qwen2.5-3B-Instruct-4bit` build becomes available and still fits in 8GB.

---

### Production: Reading & Writing

**Primary:** `Qwen/Qwen2.5-7B-Instruct`

**Why this model:**
- Qwen2.5-7B consistently ranks at the top of its weight class on structured output
  and instruction-following benchmarks (IFEval, MT-Bench).
- RW tasks are fundamentally about following complex rhetorical rules and producing
  well-reasoned prose explanations — capabilities where Qwen2.5 outperforms same-size
  Llama and Mistral variants.
- 7B is the sweet spot for LoRA fine-tuning on a 40GB A100: full batch sizes, fast
  iteration, no need for gradient checkpointing.
- Apache 2.0 license — no fine-tuning or deployment restrictions.

**LoRA target modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

**When to consider upgrading:**
- If SAT RW accuracy on hard difficulty plateaus below 82% after full fine-tuning.
- Candidate upgrade: `Qwen/Qwen2.5-14B-Instruct` (requires 80GB VRAM or 4-bit quant).

---

### Production: Math

**Primary:** `microsoft/phi-4`

**Why this model:**
- phi-4 (14B) was specifically trained on high-quality synthetic math and reasoning
  data. On MATH and GSM8K benchmarks it outperforms models significantly larger than
  itself.
- SAT hard math involves multi-step algebraic reasoning, geometry, and data analysis.
  phi-4 handles these more reliably than Qwen2.5-7B at SAT hard difficulty in
  internal testing.
- Microsoft Research license permits fine-tuning for non-commercial and research use.
  **Verify license terms before any commercial deployment.**

**LoRA target modules:** `q_proj, k_proj, v_proj, o_proj, fc1, fc2`

**VRAM requirement:** ~28GB at bf16, ~16GB at 4-bit. Use 4-bit QLoRA on instances
below 40GB. Recommended instance: A100 80GB (full precision) or A100 40GB (4-bit).

**When to consider upgrading:**
- If SAT Math accuracy on hard difficulty plateaus below 85% after full fine-tuning.
- Candidate upgrade: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (strong math
  reasoning, same weight class, worth A/B testing).

---

### Fallback Model (both sections)

**Fallback:** `meta-llama/Llama-3.1-8B-Instruct`

**Why this model:**
- Widely available on all hosting providers, well-supported by `transformers` and `trl`.
- Acts as a known-good baseline if a primary model is unavailable, rate-limited,
  or produces unexpected behavior after a HuggingFace update.
- Use the fallback for unblocking work only — do not treat it as a production target.

**Activating the fallback:**
```bash
FORCE_MODEL=meta-llama/Llama-3.1-8B-Instruct python scripts/train.py --section math
```

---

### How to Swap a Model

When asked to swap a model, Claude must follow these five steps in order and
confirm each one before proceeding:

1. Update the relevant model ID in `configs/local.yaml` or `configs/production.yaml`.
2. Update the **LoRA target modules** for the new architecture in `configs/models/rw.yaml`
   or `configs/models/math.yaml`. Wrong target modules silently reduce fine-tuning
   effectiveness.
3. Verify the tokenizer's chat template is compatible with the prompt format in
   `src/data/generator.py`. Qwen, Llama, and Phi all use different template formats.
4. Run a short smoke-test training run locally (`--num_epochs 1 --max_steps 10`)
   before committing the change.
5. Update the model name and rationale in this section of `CLAUDE.md`.

---

## Data Schema

This section is the **single source of truth for the IIAS item schema**.

**IIAS (Intelligent Item Authoring System)** generates, validates, and manages SAT test questions.
The schema below defines the structure for items in the Item Bank.

**Implementation:** The canonical schema is implemented in `src/auto_qa/schema.py` as Pydantic v2 models.

**To change the schema:** edit this section first, then update `src/auto_qa/schema.py` to match.

---

### Item Schema Version

```
SCHEMA_VERSION = "2.0.0"  # IIAS item schema (distinct from legacy Meridian training schema)
```

---

### Canonical Field Definitions

**Top-Level Item Fields:**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `id` | `string` | ✅ | UUID4 format | Unique item identifier |
| `section` | `enum` | ✅ | `reading_writing` or `math` | SAT test section |
| `domain` | `string` | ✅ | See domain lists below | Subcategory within section |
| `difficulty` | `enum` | ✅ | `easy`, `medium`, `hard` | SAT difficulty tier |
| `content_json` | `object` | ✅ | Nested structure | Item content (see below) |
| `model_version` | `string` | ❌ | — | Model that generated this item |

**content_json Nested Structure:**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `passage` | `string` | ❌ | Max 2000 chars | Optional passage for RW items |
| `question` | `string` | ✅ | Min 20, max 1200 chars | Question stem |
| `math_format` | `enum` | ❌ | `plain` or `latex` | Math formatting (Math items only) |
| `choices` | `array` | ✅ | Exactly 4 objects | Multiple choice options |
| `choices[].label` | `enum` | ✅ | `A`, `B`, `C`, `D` | Choice label (must be in order) |
| `choices[].text` | `string` | ✅ | Min 1 char | Choice text content |
| `correct_answer` | `enum` | ✅ | `A`, `B`, `C`, `D` | Label of correct choice |
| `correct_answer_text` | `string` | ✅ | Min 1 char | Text content of correct answer |
| `rationale` | `string` | ✅ | Min 60, max 500 chars | Explanation for correct answer |
| `solution_steps` | `string` | ❌ | — | Optional step-by-step solution |

**Database-Only Fields** (not in Pydantic schema, managed by Item Bank):

| Field | Type | Database |
|---|---|---|
| `status` | `enum` | `draft`, `pretesting`, `operational`, `retired` |
| `created_at`, `updated_at` | `timestamp` | Automatic |
| `irt_a`, `irt_b`, `irt_c` | `float` | IRT parameters (seeded: 1.0, 0.0, 0.25) |
| `irt_source` | `enum` | `seeded`, `predicted`, `calibrated` |
| `auto_qa_passed` | `boolean` | Auto-QA validation result |
| `qa_score` | `float` | Auto-QA score (0.0 - 1.0) |
| `qa_flags` | `array` | Auto-QA failure reasons |
| `reviewed_at`, `reviewer_id` | — | Review metadata |

---

### Domain Lists

These are the valid values for the `domain` field. See `src/item_bank/migrations/init.sql` for complete list.

**Reading & Writing (`section = reading_writing`)**

- `information_and_ideas.central_ideas_and_details`
- `information_and_ideas.command_of_evidence_textual`
- `information_and_ideas.inferences`
- `information_and_ideas.words_in_context`
- `craft_and_structure.text_structure_and_purpose`
- `craft_and_structure.cross_text_connections`
- `expression_of_ideas.rhetorical_synthesis`
- `expression_of_ideas.transitions`
- `standard_english_conventions.boundaries`
- `standard_english_conventions.form_structure_sense`
- `standard_english_conventions.standard_english`

**Math (`section = math`)**

- `algebra.linear_equations_one_variable`
- `algebra.linear_equations_two_variables`
- `algebra.linear_functions`
- `algebra.systems_of_linear_equations`
- `advanced_math.nonlinear_functions`
- `advanced_math.nonlinear_equations`
- `advanced_math.equivalent_expressions`
- `problem_solving_and_data_analysis.ratios_rates_proportions`
- `problem_solving_and_data_analysis.percentages`
- `problem_solving_and_data_analysis.one_variable_data`
- `problem_solving_and_data_analysis.two_variable_data`
- `problem_solving_and_data_analysis.probability`
- `problem_solving_and_data_analysis.inference_from_samples`
- `geometry_and_trigonometry.area_volume`
- `geometry_and_trigonometry.lines_angles_triangles`
- `geometry_and_trigonometry.right_triangles_trigonometry`
- `geometry_and_trigonometry.circles`

---

### Example (valid, complete)

```json
{
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "section": "math",
  "domain": "algebra.quadratic_equations",
  "difficulty": "medium",
  "content_json": {
    "passage": null,
    "question": "If x² - 5x + 6 = 0, which of the following gives all possible values of x?",
    "math_format": "latex",
    "choices": [
      {"label": "A", "text": "x = 2 only"},
      {"label": "B", "text": "x = -2 and x = -3"},
      {"label": "C", "text": "x = 2 and x = 3"},
      {"label": "D", "text": "x = -2 and x = 3"}
    ],
    "correct_answer": "C",
    "correct_answer_text": "x = 2 and x = 3",
    "rationale": "Factoring gives (x-2)(x-3)=0, so x=2 or x=3. Choice A and B are incomplete. Choice D has wrong signs.",
    "solution_steps": null
  },
  "model_version": "rw-sft-v1.0"
}
```

---

### Validation Rules

**Schema validation (hard gate):**
- UUID4 format for `id`
- `section` must be `reading_writing` or `math`
- `difficulty` must be `easy`, `medium`, or `hard`
- Exactly 4 choices with labels A, B, C, D in order
- `correct_answer` must match one of the choice labels
- `correct_answer_text` must equal the text of `correct_answer`

**Auto-QA validation:**
- Question length: 20-1200 characters
- Rationale length: 60-500 characters
- Passage length: max 2000 characters (if present)
- Math items should use `latex` format when expressions present
5. Claude warns you if the change is breaking (i.e. removes or renames a required
   field) and asks whether existing `data/validated/` files should be migrated
   or discarded.

---

## Data Pipeline

### Flow

```
generate_data.py
  └── pipeline.py
        ├── generator.py   → raw JSONL written to data/generated/
        ├── validator.py   → schema check + rubric auto-QA
        └── splits         → train/val/test JSONL written to data/splits/
```

### Running the Pipeline

```bash
# Generate and validate 500 hard RW examples
python scripts/generate_data.py --section rw --count 500 --difficulty hard

# Generate a balanced set across all difficulties
python scripts/generate_data.py --section math --count 300 --difficulty easy
python scripts/generate_data.py --section math --count 300 --difficulty medium
python scripts/generate_data.py --section math --count 300 --difficulty hard
```

The pipeline will refuse to proceed to splits if the total validated example count
is below `data.min_examples` in the active config. This threshold exists to prevent
training on dangerously small datasets.

---

## Training

### Recommended: HuggingFace SFTTrainer

**Use `scripts/train_huggingface.py` for all training.** This script uses `trl.SFTTrainer` which correctly implements loss masking for supervised fine-tuning.

```bash
# Train math model on RunPod
python scripts/train_huggingface.py --section math --env production

# Train reading/writing model
python scripts/train_huggingface.py --section reading_writing --env production

# Resume from checkpoint
python scripts/train_huggingface.py --section math --resume-from checkpoints/math/latest
```

**Why SFTTrainer:**

- Correctly implements loss masking (only computes loss on assistant tokens)
- Handles all chat template formats automatically
- Industry-standard library for SFT training
- Zero data preprocessing required

### Deprecated: Custom Dataset

**The `SFTDataset` class in `src/training/dataset.py` is deprecated.** It has incorrect loss masking implementation. Use `trl.SFTTrainer` instead.

See migration guide in the `SFTDataset` docstring.

### Legacy Script (Buggy)

**`scripts/train_model.py` has known bugs** - use `train_huggingface.py` instead. The legacy script has:
- Import statement issues
- Duplicate imports inside functions
- Incorrect config parameter usage

### Checkpointing

Checkpoints are saved to `checkpoints/{section}/{timestamp}/`.
Each checkpoint directory contains:

- `adapter_model.safetensors` — LoRA adapter weights
- `adapter_config.json` — LoRA configuration
- `tokenizer_config.json` — Tokenizer configuration
- Training logs and metrics

---

## Evaluation

```bash
python scripts/evaluate.py --section math --checkpoint outputs/checkpoints/math/latest
python scripts/evaluate.py --section rw --checkpoint outputs/checkpoints/rw/latest
```

Metrics reported per run:

- **Accuracy** — percentage correct on the held-out test split
- **Rubric score** — average auto-QA score of model-generated explanations
- **Difficulty breakdown** — accuracy split by easy / medium / hard
- **Domain breakdown** — per-domain accuracy for both RW and Math

Results are written to `outputs/evals/{section}_{timestamp}.json`.

---

## API

```bash
uvicorn src.api.server:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/answer` | Submit a question, get answer + explanation |
| `GET` | `/v1/health` | Health check and active model info |
| `GET` | `/v1/config` | Returns the active config (no secrets) |

---

## Coding Conventions

### General

- Python 3.11+. Use type hints everywhere, including return types.
- Use `pydantic` v2 models for all data structures that cross module boundaries.
- Use `pathlib.Path` — never `os.path`.
- Use `loguru` for all logging — never `print()` in library code.
- Config is always loaded via `src/models/base.py:load_config()` — never read YAML
  directly in feature code.
- All CLI scripts in `scripts/` use `argparse`. No click, no typer.

### Error Handling

Raise specific exceptions, never bare `Exception`:

- Data validation failures → `ValidationError` (pydantic)
- Config issues → `ConfigError` (define in `src/models/base.py`)
- Training failures → `TrainingError` (define in `src/training/trainer.py`)
- Schema version mismatch → `SchemaVersionError` (define in `src/data/validator.py`)

### Testing

- Every public function in `src/` needs at least one test.
- Use `pytest`. All shared fixtures go in `tests/conftest.py`.
- Mock all LLM calls in tests — never make real API calls in the test suite.
- Tests must pass before any commit that touches `src/`.
- Run with: `pytest tests/ -v`

### File Naming

- Snake case for all Python files.
- Kebab case for YAML configs.
- UUID4 for all generated data example IDs.

---

## What Claude Should Never Do

- Do not hardcode model IDs anywhere outside of YAML config files.
- Do not use `scripts/train_model.py` - it has known bugs. Use `scripts/train_huggingface.py`.
- Do not commit `.env`, `data/raw/`, `data/generated/`, or `outputs/`.
- Do not add new top-level dependencies without updating `requirements.txt`.
- Do not write examples to `data/splits/` that have not passed validation.
- Do not infer schema fields from code — always read the field table in this file.
- Do not use `print()` in any file under `src/` — use `loguru`.
- Do not modify `SCHEMA_VERSION` without also updating this file's schema section.
- Do not assume the `passage` field is populated — always check for null before use.
- Do not create custom training loops — use `transformers.Trainer` from HuggingFace.

---

## Useful Commands

```bash
# Training (recommended - HuggingFace)
python scripts/train_huggingface.py --section math --env production
python scripts/train_huggingface.py --section reading_writing --env production

# Training with resume
python scripts/train_huggingface.py --section math --resume-from checkpoints/math/latest

# Check validated example counts
find data/validated -name "*.jsonl" | xargs wc -l

# Run full test suite
pytest tests/ -v

# Run a single test file
pytest tests/test_data.py -v

# Lint and type check
ruff check src/ && mypy src/

# Tail training logs
tail -f logs/training_*.log
```

### RunPod Deployment Commands

```bash
# Discover available GPUs
cd ~/mlt_automation
./mlt_train.sh discover A100-80GB 80

# Create a training instance
./mlt_train.sh create --gpu A100-80GB --pricing spot

# Check instance status
./mlt_train.sh status
```

---

## RunPod Deployment (MLT Automation)

The MLT (Machine Learning Training) automation system is located at `~/mlt_automation/` (outside this repository).

**Features:**
- GPU discovery with pricing (spot and on-demand)
- Automatic instance creation on RunPod
- 5-second graceful shutdown for checkpoint saving
- SSH key management
- Cost tracking and estimation

**Quick Start:**
```bash
cd ~/mlt_automation
./mlt_train.sh discover A100-80GB 80
./mlt_train.sh create --gpu A100-80GB --pricing spot
./mlt_train.sh status
```

**Environment Setup:**
- API key: `~/mlt_automation/.env` (contains `RUNPOD_API_KEY`)
- Config: `~/mlt_automation/config/mlt.conf`
- State: `~/mlt_automation/state/`

**Important:** The MLT automation uses RunPod, not vast.ai. All production deployment should use this system.

---

## Current Status

> Keep this section up to date as the project progresses. Claude reads this to
> understand what has already been built before generating new code.

- [x] Config loader implemented (`src/config.py`)
- [x] Pydantic schema + validator implemented (`src/auto_qa/schema.py`)
- [x] HuggingFace training script (`scripts/train_huggingface.py`)
- [x] Production config updated for RunPod deployment
- [x] MLT automation for RunPod GPU management (`~/mlt_automation/`)
- [ ] Data generator implementation incomplete
- [ ] Data pipeline orchestrator incomplete
- [ ] Local MLX training not implemented
- [ ] Evaluation pipeline incomplete
- [ ] FastAPI server incomplete

**Production-ready:**
- Use `scripts/train_huggingface.py` for training
- Use `~/mlt_automation/` for RunPod deployment
- Config schema matches `configs/production.yaml`

**Known Issues:**
- `scripts/train_model.py` has bugs (use `train_huggingface.py` instead)
- `src/training/trainer.py` and related files are empty/incomplete
```
