# Generation Service — Complete Guide

**Component:** Generation Service
**Version:** 1.0 (MVP)
**Last Updated:** 2026-03-27

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is Fine-Tuning?](#what-is-fine-tuning)
3. [Model Architecture](#model-architecture)
4. [Training Data Format](#training-data-format)
5. [Training Process](#training-process)
6. [Inference (Generation)](#inference-generation)
7. [Implementation Details](#implementation-details)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is the Generation Service?

The **Generation Service** is the component of IIAS responsible for **training and hosting fine-tuned language models** that generate SAT-style test questions.

**Key responsibilities:**
1. **Train models** on existing SAT items (supervised fine-tuning)
2. **Host models** for inference (generation)
3. **Generate new items** on demand (given constraints)
4. **Manage model checkpoints** (versioning, rollback)

### Two Separate Models

IIAS uses **two distinct models**, each specialized for a different SAT section:

| Model | Base Architecture | Purpose | Training Data |
|-------|------------------|---------|---------------|
| **RW Model** | Qwen/Qwen2.5-7B-Instruct | Generate Reading & Writing items | SAT RW questions with passages |
| **Math Model** | microsoft/phi-4 | Generate Math items with LaTeX | SAT Math questions |

**Why two models?**

Reading & Writing and Math are fundamentally different tasks:

**RW challenges:**
- Passage comprehension (300-500 word texts)
- Rhetorical analysis
- Grammar and usage
- Text structure and organization

**Math challenges:**
- Mathematical reasoning (multi-step problem solving)
- Symbolic notation (LaTeX formatting)
- Algebraic manipulation
- Geometric visualization

A single model struggles to excel at both. Specialized models perform better.

---

## What is Fine-Tuning?

### Definition

**Fine-tuning** is the process of taking a **pre-trained language model** (that has learned general language patterns from vast text corpora) and training it further on a **specific, smaller dataset** to specialize its behavior for a particular task.

### The Fine-Tuning Process

```
Step 1: Pre-training (done by model creators)
┌─────────────────────────────────────────────────────────────┐
│ Base Model (e.g., Qwen2.5-7B-Instruct)                      │
│ Trained on: internet text, books, code, etc.                │
│ Learned: general language patterns, reasoning, formatting   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
Step 2: Fine-tuning (what we do)
┌─────────────────────────────────────────────────────────────┐
│ Training Data: 4,101 SAT items in chat template format      │
│ Task: Learn to generate SAT questions in JSON format        │
│ Method: LoRA (Low-Rank Adaptation)                          │
│ Result: Specialized model for SAT item generation           │
└─────────────────────────────────────────────────────────────┘
```

### Why Fine-Tune? Why Not Just Use GPT-4?

**Base models (GPT-4, Claude, etc.) can generate SAT questions, BUT:**

1. **Inconsistent format:** Sometimes JSON, sometimes markdown, sometimes plain text
2. **Domain drift:** May generate questions outside SAT scope
3. **Difficulty misalignment:** "Hard" questions aren't consistently hard
4. **No schema adherence:** May miss required fields or use wrong enum values
5. **Expensive:** $0.01-0.10 per generation via API
6. **Rate limits:** Can't generate thousands of items quickly
7. **No control:** Can't update model behavior without re-prompting

**Fine-tuned models:**

1. **Consistent JSON output:** 98%+ schema validity rate
2. **Domain-aligned:** Trained specifically on SAT content
3. **Difficulty-calibrated:** Learns SAT difficulty patterns
4. **Schema-enforced:** Pydantic validation during training
5. **Cost-effective:** $0.0001-0.001 per generation (self-hosted)
6. **Unlimited throughput:** Generate as fast as GPU allows
7. **Controllable:** Retrain model to improve behavior

### What Fine-Tuning Teaches the Model

**Before fine-tuning, a base model prompted with:**

```
"Generate a SAT Math question about quadratic equations"
```

**Might produce:**
```
"What is the quadratic formula?"
(Too simple, not in SAT format, no choices)
```

**After fine-tuning on 500 SAT quadratic equation items, the model learns:**

```
{
  "section": "math",
  "domain": "algebra.quadratic_equations",
  "difficulty": "medium",
  "content_json": {
    "question": "If x² - 5x + 6 = 0, what are the roots?",
    "choices": [
      {"label": "A", "text": "x = 2, 3"},
      {"label": "B", "text": "x = -2, -3"},
      {"label": "C", "text": "x = 2, -3"},
      {"label": "D", "text": "x = -2, 3"}
    ],
    "correct_answer": "A",
    "rationale": "..."
  }
}
```

**The model learned:**
- SAT Math format (4 multiple choice options, A-D)
- LaTeX for math expressions (x², not x^2)
- Appropriate difficulty for SAT
- JSON structure with required fields
- SAT-style distractors (wrong answers that seem plausible)

---

## Model Architecture

### Base Models

### RW Model: Qwen2.5-7B-Instruct

**Why Qwen2.5-7B?**

1. **Superior structured output:** Qwen2.5 consistently ranks #1 among 7B models for JSON generation
2. **Strong instruction following:** Excels at following complex formatting constraints
3. **Good tokenizer:** Handles punctuation and non-ASCII characters well (important for passages)
4. **Open license:** Apache 2.0 — allows commercial use and fine-tuning
5. **7B parameter sweet spot:** Large enough for quality, small enough for cost-effective training

**Model specifications:**
```
Parameters: 7 billion
Context window: 32,768 tokens
Architecture: Decoder-only transformer
Training data: Multilingual web text, code, books
License: Apache 2.0
```

### Math Model: Phi-4

**Why Phi-4?**

1. **Math-specialized:** Trained specifically on high-quality math reasoning data
2. **Superior reasoning:** Outperforms larger models on MATH and GSM8K benchmarks
3. **14B parameters:** More capacity for multi-step mathematical reasoning
4. **LaTeX support:** Generates properly formatted mathematical notation
5. **Microsoft Research license:** Allows research and non-commercial use

**Model specifications:**
```
Parameters: 14 billion
Context window: 16,384 tokens (varies by quantization)
Architecture: Decoder-only transformer
Training data: Math-focused synthetic and curated data
License: Microsoft Research License (verify before commercial use)
```

### LoRA Adapters

**What are LoRA adapters?**

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method. Instead of updating all 7B or 14B parameters in the base model, we train **small adapter matrices** that modify the model's behavior.

**How it works:**

```
Base model weight matrix: W (7B × 7B = 49 billion parameters)

Traditional fine-tuning: Update W → W + ΔW (all 49B parameters)

LoRA fine-tuning: Decompose ΔW into two low-rank matrices:
  ΔW = A × B
  Where: A is (7B × r), B is (r × 7B), and r is rank (e.g., 32)

Parameters to train: 7B × 32 + 32 × 7B = 448 million (1% of base model!)
```

**LoRA configuration for IIAS:**

```yaml
lora_r: 32              # Rank — higher = more capacity, more parameters
lora_alpha: 64          # Scaling factor — controls adapter influence
lora_dropout: 0.05      # Dropout for regularization
target_modules:         # Which layers to attach adapters to
  - q_proj              # Query projection in attention
  - k_proj              # Key projection
  - v_proj              # Value projection
  - o_proj              # Output projection
  - gate_proj           # Gating in feed-forward layers
  - up_proj             # Upscaling in feed-forward layers
  - down_proj           # Downscaling in feed-forward layers
bias: none              # Don't train bias terms
task_type: CAUSAL_LM    # Causal language modeling (standard for LLMs)
```

**Why rank 32?**

- **Rank 16:** Adequate for conversational tasks, but struggles with strict JSON formatting
- **Rank 32:** Sweet spot for structured output — enough capacity to learn schema constraints
- **Rank 64+:** Diminishing returns, more parameters, slower training

**Why these target modules?**

These modules cover **all linear transformations** in the transformer architecture:
- **Attention layers:** q_proj, k_proj, v_proj, o_proj (query/key/value/output projections)
- **Feed-forward layers:** gate_proj, up_proj, down_proj (gating and transformations)

**Result:** LoRA adapters modify the model's behavior at every layer, enabling fine-grained adaptation.

### QLoRA: Quantized LoRA

**QLoRA** is LoRA applied to a **4-bit quantized base model**. This dramatically reduces memory usage, enabling training on smaller GPUs.

**How quantization works:**

```
Standard precision (FP16 or BF16): 16 bits per parameter
4-bit quantization (NF4): 4 bits per parameter

Memory reduction: 16/4 = 4x less memory

Example: Qwen2.5-7B
FP16: 7B × 2 bytes = 14 GB VRAM
4-bit: 7B × 0.5 bytes = 3.5 GB VRAM
```

**QLoRA configuration:**

```yaml
load_in_4bit: true                    # Load base model in 4-bit
bnb_4bit_quant_type: nf4              # NormalFloat 4-bit (optimal for weights)
bnb_4bit_compute_dtype: bfloat16      # Use BF16 for computations (stability)
bnb_4bit_use_double_quant: true       # Quantize quantization constants (saves more memory)
```

**Trade-offs:**

- **Pro:** Train on smaller GPUs (RTX 3090, 4090 instead of A100)
- **Pro:** Faster training due to memory efficiency
- **Con:** Slight quality degradation (usually <1% on benchmarks)
- **Con:** Slightly slower inference (dequantization overhead)

**For IIAS:** We use QLoRA for training efficiency. Quality impact is negligible for our task.

---

## Training Data Format

### The Chat Template Approach

**Critical design decision:** IIAS uses the **chat template format** for training, not raw input/output pairs.

**Why chat templates matter:**

Modern instruction-tuned models (Qwen, Llama, Phi) are trained to follow **conversational patterns**. Their capabilities (instruction following, formatting, refusal behavior) are encoded around their specific chat message format.

**Bypassing the chat template** (using raw input/output) degrades model behavior:
- Loses instruction-following ability
- May repeat or paraphrase instructions in output
- Fails to leverage the model's formatting training

### Chat Template Structure

**Training example format:**

```json
{
  "dataset_version": "rw-sft-v1.0",
  "schema_version": "item-schema-v1",
  "section": "reading_writing",
  "domain": "information_and_ideas.central_ideas_and_details",
  "difficulty_tier": "medium",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert SAT item writer. Generate items in strict JSON matching the provided schema. Output JSON only, with no preamble, explanation, or markdown formatting."
    },
    {
      "role": "user",
      "content": "Generate a single SAT Reading & Writing item.\n\nConstraints:\n- section: reading_writing\n- domain: information_and_ideas.central_ideas_and_details\n- difficulty_tier: medium\n- topic: effects of urbanization on bird populations\n- passage_length_words: 300-400\n- question_type: main_idea"
    },
    {
      "role": "assistant",
      "content": "{\"section\": \"reading_writing\", \"domain\": \"information_and_ideas.central_ideas_and_details\", ...}"
    }
  ]
}
```

**Message roles explained:**

**System message:**
- **Purpose:** Sets the behavior and format expectations
- **Content:** "You are an expert SAT item writer. Generate items in strict JSON..."
- **Model learns:** "I'm supposed to output JSON only, no conversational filler"

**User message:**
- **Purpose:** Provides the generation constraints
- **Content:** Section, domain, difficulty, topic, format requirements
- **Model learns:** "Given these constraints, generate an appropriate item"

**Assistant message:**
- **Purpose:** The target output — what the model should generate
- **Content:** The complete item in JSON format
- **Model learns:** "This is what a correct SAT item looks like"

### Loss Masking

**Critical:** Loss is computed **only on the assistant turn tokens**. System and user tokens are masked (set to -100) and do not contribute to the loss.

**Why?**

If we compute loss on instruction tokens:
- Model wastes capacity learning to predict the instructions
- May develop tendency to repeat instructions in output
- Diffuses gradient signal from what we actually care about (the item)

**Loss masking in practice:**

```
Input sequence:
[System: "You are an expert...", User: "Generate a SAT item...", Assistant: "{...item JSON...}"]

Loss computation:
- System tokens: masked (loss = 0)
- User tokens: masked (loss = 0)
- Assistant tokens: compute loss (train on these)

Result: Model learns to generate items conditioned on instructions, not to predict instructions.
```

### Complete Training Example: Reading & Writing

```json
{
  "dataset_version": "rw-sft-v1.0",
  "schema_version": "item-schema-v1",
  "section": "reading_writing",
  "domain": "information_and_ideas.central_ideas_and_details",
  "difficulty_tier": "medium",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert SAT item writer. Generate items in strict JSON matching the provided schema. Output JSON only, with no preamble, explanation, or markdown formatting."
    },
    {
      "role": "user",
      "content": "Generate a single SAT Reading & Writing item.\n\nConstraints:\n- section: reading_writing\n- domain: information_and_ideas.central_ideas_and_details\n- difficulty_tier: medium\n- topic: effects of urbanization on bird populations\n- passage_length_words: 300-400\n- question_type: main_idea"
    },
    {
      "role": "assistant",
      "content": "{\n  \"section\": \"reading_writing\",\n  \"domain\": \"information_and_ideas.central_ideas_and_details\",\n  \"difficulty_tier\": \"medium\",\n  \"content_json\": {\n    \"passage\": \"As cities expand outward, the forests and grasslands that once bordered them are replaced by roads, parking lots, and residential developments. For many bird species, this transformation is catastrophic. Forest-interior specialists—birds that require large, unbroken tracts of habitat for nesting and foraging—find themselves stranded in shrinking patches of green space surrounded by hostile urban matrix. Research conducted in several mid-Atlantic metropolitan regions has documented declines of 40 to 60 percent in the populations of wood thrushes, ovenbirds, and other area-sensitive species over the past three decades. The mechanism is not simply habitat loss but fragmentation: even when the total acreage of forested land remains roughly constant, breaking it into smaller, isolated parcels dramatically increases the ratio of edge to interior habitat. Edge zones expose nests to higher rates of predation by raccoons and blue jays, and they allow brown-headed cowbirds—brood parasites that thrive in disturbed landscapes—to penetrate deeper into forest patches and lay their eggs in the nests of other species. The net result is a landscape that appears green on a satellite map but functions, for area-sensitive species, as a biological desert.\",\n    \"question\": \"According to the passage, what is the central claim about urban expansion and bird populations?\",\n    \"math_format\": \"plain\",\n    \"choices\": [\n      {\"label\": \"A\", \"text\": \"Urban expansion increases the diversity of bird species that can survive in metropolitan areas.\"},\n      {\"label\": \"B\", \"text\": \"Habitat fragmentation, more than total habitat loss, is responsible for declines in area-sensitive bird species.\"},\n      {\"label\": \"C\", \"text\": \"Predation by raccoons is the primary driver of declining forest bird populations near cities.\"},\n      {\"label\": \"D\", \"text\": \"Cities that preserve total forest acreage successfully maintain healthy populations of interior forest birds.\"}\n    ],\n    \"correct_answer\": \"B\",\n    \"correct_answer_text\": \"Habitat fragmentation, more than total habitat loss, is responsible for declines in area-sensitive bird species.\",\n    \"rationale\": \"The passage explicitly states that 'the mechanism is not simply habitat loss but fragmentation' and explains how breaking forests into smaller parcels increases edge habitat and predation pressure. Choice B directly captures this central argument. Choice A contradicts the passage. Choice C identifies one factor but the passage presents it as one consequence of fragmentation, not the primary driver. Choice D is directly contradicted by the final sentence.\"\n  }\n}"
    }
  ]
}
```

### Complete Training Example: Math

```json
{
  "dataset_version": "math-sft-v1.0",
  "schema_version": "item-schema-v1",
  "section": "math",
  "domain": "algebra.linear_equations_two_variables",
  "difficulty_tier": "medium",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert SAT Math item writer. Generate items in strict JSON with LaTeX for all mathematical expressions. Output JSON only, with no preamble, explanation, or markdown formatting."
    },
    {
      "role": "user",
      "content": "Generate a single SAT Math item.\n\nConstraints:\n- section: math\n- domain: algebra.linear_equations_two_variables\n- difficulty_tier: medium\n- skill: interpreting the meaning of a variable in a linear model\n- math_format: latex\n- context: real-world application (cost modeling)"
    },
    {
      "role": "assistant",
      "content": "{\n  \"section\": \"math\",\n  \"domain\": \"algebra.linear_equations_two_variables\",\n  \"difficulty_tier\": \"medium\",\n  \"content_json\": {\n    \"passage\": null,\n    \"question\": \"A catering company charges a flat setup fee plus a fixed price per guest. The total cost $C$, in dollars, for $n$ guests is modeled by $C = 12n + 350$. What does the value 350 represent in this context?\",\n    \"math_format\": \"latex\",\n    \"choices\": [\n      {\"label\": \"A\", \"text\": \"The cost per guest\"},\n      {\"label\": \"B\", \"text\": \"The total cost for 350 guests\"},\n      {\"label\": \"C\", \"text\": \"The flat setup fee charged regardless of the number of guests\"},\n      {\"label\": \"D\", \"text\": \"The number of guests the company can accommodate\"}\n    ],\n    \"correct_answer\": \"C\",\n    \"correct_answer_text\": \"The flat setup fee charged regardless of the number of guests\",\n    \"solution_steps\": \"In the linear model $C = 12n + 350$, the term $12n$ represents the variable cost that depends on $n$ (the number of guests), and 350 is the constant term. A constant term in a cost model represents a fixed charge that applies regardless of the quantity—here, the setup fee. When $n = 0$, $C = 350$, confirming that 350 is the cost before any guests are added.\",\n    \"rationale\": \"Choice C is correct. The 350 is the y-intercept of the linear model—the cost when $n = 0$—which represents the flat setup fee. Choice A confuses 350 with the slope (12), which is the per-guest cost. Choice B misreads 350 as a guest count rather than a dollar amount. Choice D is irrelevant to the model.\"\n  }\n}"
    }
  ]
}
```

**Key differences from RW:**
- No passage (Math items are standalone)
- `math_format: "latex"` (all expressions in LaTeX)
- `solution_steps` field included (step-by-step reasoning)
- Shorter question text (no long passage)

### Dataset Requirements

**Minimum examples per cell:** 150

Each `(section, domain, difficulty_tier)` combination must have at least 150 training examples.

**Why?**

Below 150 examples:
- Model overfits to the few examples
- Memorizes specific questions rather than learning patterns
- Poor generalization to new topics

At 150+ examples:
- Model learns domain patterns
- Can generate novel items (not just memorized training data)
- Stable training behavior

**Example:**

```
Good coverage:
- RW / central_ideas / easy: 180 examples ✅
- RW / central_ideas / medium: 210 examples ✅
- RW / central_ideas / hard: 150 examples ✅

Poor coverage:
- Math / quadratic_equations / hard: 45 examples ❌
→ Don't train on this cell yet, acquire more data
```

**Dataset versioning policy:**

Every training JSONL file is named with its version:
```
rw-sft-v1.0.jsonl  (initial version)
rw-sft-v1.1.jsonl  (minor: added examples, typo fixes)
rw-sft-v2.0.jsonl  (major: schema change, systematic relabeling)
```

**Versioning rules:**
- **Minor bump (v1.0 → v1.1):** New examples added, existing examples lightly corrected
  - Can continue fine-tuning from previous checkpoint
- **Major bump (v1 → v2):** Schema change, systematic relabeling, domain restructuring
  - Must retrain from base model (not from previous checkpoint)

---

## Training Process

### Step-by-Step Training Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Data Preparation                                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
    Load ItemBank data (4,101 questions)
                    │
                    ▼
    Convert to chat template format
                    │
                    ▼
    Split into train/validation (85%/15%)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Configuration                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
    Load training config (YAML)
                    │
                    ▼
    Set hyperparameters (LR, batch size, epochs)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Model Loading                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
    Load base model (Qwen2.5-7B or Phi-4)
                    │
                    ▼
    Apply 4-bit quantization (QLoRA)
                    │
                    ▼
    Attach LoRA adapters (rank 32)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Training Loop                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
    For each epoch:
      For each batch:
        Forward pass (compute loss)
        Backward pass (compute gradients)
        Update LoRA weights
        Log metrics (loss, learning rate)
        Validate on validation set
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Checkpoint Saving                                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
    Save LoRA adapter weights
                    │
                    ▼
    Save training configuration
                    │
                    ▼
    Save validation metrics
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Evaluation                                          │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
    Generate test items
                    │
                    ▼
    Compute JSON validity rate
                    │
                    ▼
    Human evaluation (optional)
                    │
                    ▼
    Promote to production if metrics pass
```

### Hyperparameters

**Complete configuration:**

```yaml
# LoRA adapter configuration
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
bias: none
task_type: CAUSAL_LM

# Quantization (QLoRA)
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

# Training efficiency
gradient_checkpointing: true    # Trade compute for memory
flash_attention_2: true          # Faster attention

# Hyperparameters
learning_rate: 2e-5             # Conservative LR for instruction-tuned models
effective_batch_size: 64        # Actual batch size across GPUs
gradient_accumulation_steps: 8   # Simulate larger batch
num_train_epochs: 3              # Full passes through training data
max_seq_length_rw: 4096         # Context window for RW (passage + JSON)
max_seq_length_math: 2048       # Context window for Math (no passage)

# Optimizer
optimizer: adamw_torch           # AdamW with PyTorch implementation
weight_decay: 0.01               # L2 regularization
lr_scheduler: cosine             # Cosine annealing schedule
warmup_ratio: 0.05              # 5% of steps for LR warmup

# Early stopping
early_stopping_metric: json_validity_rate_on_validation
early_stopping_patience: 2       # Stop if no improvement for 2 epochs
```

**Rationale for key choices:**

**Learning rate: 2e-5**
- Too high (>1e-4): Destroys model's formatting and refusal behavior
- Too low (<1e-6): Training never converges
- 2e-5: Conservative, maintains base model capabilities while adapting

**Batch size: 64 (effective)**
- Larger batches = more stable gradients
- Achieved via gradient accumulation (8 steps × 8 batch per GPU)
- Fits in GPU memory with QLoRA

**Max sequence length: 4096 (RW) vs 2048 (Math)**
- RW items: Passage (500 words) + JSON = ~3000 tokens
- Math items: No passage + JSON = ~800 tokens
- Using 4096 for everything wastes memory on Math
- Different lengths optimize GPU usage

**Early stopping on JSON validity rate**
- Validation loss is a poor proxy for structural quality
- Model can achieve low perplexity while generating invalid JSON
- Measure what you care about: schema adherence

### Training Duration

**Estimated training times (single A100 80GB):**

| Model | Training Examples | Per Epoch | Total (3 epochs) |
|-------|------------------|-----------|------------------|
| RW (Qwen2.5-7B) | 2,000 | ~45 minutes | ~2.25 hours |
| Math (Phi-4) | 2,100 | ~90 minutes | ~4.5 hours |

**With QLoRA on smaller GPUs (RTX 4090 24GB):**
- RW: ~3.5 hours total
- Math: ~7 hours total

**Cost (vast.ai A100 80GB @ $1/hr):**
- RW: ~$2.25 per training run
- Math: ~$4.50 per training run
- **Total: ~$7 per full retrain**

---

## Inference (Generation)

### Loading a Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model (4-bit quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "checkpoints/rw-sft-v1.0"  # Path to saved adapter
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

### Generating Items

```python
def generate_item(section: str, domain: str, difficulty: str, topic: str = None):
    """
    Generate a single SAT item.

    Args:
        section: "reading_writing" or "math"
        domain: e.g., "algebra.quadratic_equations"
        difficulty: "easy", "medium", or "hard"
        topic: Optional specific topic

    Returns:
        Generated item as JSON
    """

    # Construct prompt
    user_message = f"Generate a single SAT {section.replace('_', ' ')} item.\n\n"
    user_message += f"Constraints:\n"
    user_message += f"- section: {section}\n"
    user_message += f"- domain: {domain}\n"
    user_message += f"- difficulty_tier: {difficulty}\n"
    if topic:
        user_message += f"- topic: {topic}\n"

    # Apply chat template
    messages = [
        {"role": "system", "content": "You are an expert SAT item writer. Generate items in strict JSON matching the provided schema. Output JSON only, with no preamble, explanation, or markdown formatting."},
        {"role": "user", "content": user_message}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,  # Maximum length of generated item
        temperature=0.7,      # Sampling temperature (0.7 = balanced creativity)
        top_p=0.9,           # Nucleus sampling
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode output
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parse JSON (may need error handling)
    import json
    try:
        item = json.loads(generated_text)
        return item
    except json.JSONDecodeError:
        # Handle invalid JSON (retry or post-process)
        return None
```

### Generation Parameters

**Temperature (0.7)**
- Controls randomness in generation
- **Low (0.1-0.3):** Deterministic, conservative, similar to training data
- **Medium (0.5-0.8):** Balanced creativity and reliability
- **High (0.9-1.5):** Very creative, may produce unusual items
- **Recommendation:** 0.7 for SAT generation (consistent but novel)

**Top-p (0.9)**
- Nucleus sampling: only sample from tokens comprising top 90% probability mass
- Prevents model from generating very low-probability tokens
- **Recommendation:** 0.9 (standard for structured generation)

**Max new tokens (2048)**
- Maximum length of generated response
- RW items: ~1500-2000 tokens (passage + JSON)
- Math items: ~500-800 tokens (no passage)
- **Recommendation:** 2048 for RW, 1024 for Math

### Batch Generation

For efficiency, generate multiple items in one call:

```python
def generate_items_batch(section: str, domain: str, difficulty: str, count: int = 5):
    """Generate multiple items in one forward pass."""

    # Construct prompts for all items
    prompts = []
    for i in range(count):
        user_message = f"Generate a single SAT {section.replace('_', ' ')} item.\n\n"
        user_message += f"Constraints:\n- section: {section}\n- domain: {domain}\n- difficulty_tier: {difficulty}\n"

        messages = [
            {"role": "system", "content": "You are an expert SAT item writer..."},
            {"role": "user", "content": user_message}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Tokenize all prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode all outputs
    items = []
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        try:
            item = json.loads(generated_text)
            items.append(item)
        except json.JSONDecodeError:
            continue  # Skip invalid JSON

    return items
```

---

## Implementation Details

### Directory Structure

```
src/generation/
├── __init__.py
├── config.py              # Configuration dataclasses
├── dataset.py             # Training data loading and preprocessing
├── model.py               # Model loading and LoRA setup
├── trainer.py             # Training loop
├── generator.py           # Inference/generation
└── utils.py               # Helper functions

scripts/
├── train_model.py         # CLI script for training
└── generate_items.py      # CLI script for generation

data/
├── raw/                   # Source ItemBank data
├── training/              # Training JSONL files
│   ├── rw_train.jsonl
│   ├── rw_val.jsonl
│   ├── math_train.jsonl
│   └── math_val.jsonl
└── generated/             # Generated items (pre-QA)

checkpoints/
├── rw-sft-v1.0/          # Saved RW model checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.json
└── math-sft-v1.0/        # Saved Math model checkpoint
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── training_args.json
```

### Key Classes and Functions

#### Configuration (`config.py`)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    early_stopping_patience: int = 2

@dataclass
class ModelConfig:
    """Model configuration."""
    section: Literal["reading_writing", "math"]
    base_model_id: str
    checkpoint_dir: str
    lora: LoRAConfig
    training: TrainingConfig
```

#### Dataset (`dataset.py`)

```python
import json
from torch.utils.data import Dataset
from typing import Dict, List

class SATDataset(Dataset):
    """Dataset for SAT item generation."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 4096):
        self.data = []
        with open(jsonl_path) as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        example = self.data[idx]
        messages = example["messages"]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Mask loss on non-assistant tokens
        labels = encodings["input_ids"].clone()

        # Find assistant message start
        assistant_start = text.find("assistant")  # Simplified
        # ... (actual implementation more complex)

        # Set labels to -100 for non-assistant tokens
        # labels[:assistant_start] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
```

#### Trainer (`trainer.py`)

```python
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

class SATTrainer(Trainer):
    """Custom trainer for SAT item generation."""

    def __init__(self, *args, json_validity_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_validity_fn = json_validity_fn

    def compute_metrics(self, eval_pred):
        """Compute metrics including JSON validity rate."""
        predictions, labels = eval_pred

        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )

        # Compute JSON validity rate
        if self.json_validity_fn:
            validity_rate = self.json_validity_fn(decoded_preds)
            return {"json_validity_rate": validity_rate}

        return {}
```

---

## Best Practices

### 1. Always Use Chat Templates

**❌ WRONG:**
```python
prompt = f"Item: {item_json}\n\nGenerate a similar item."
# Bypasses chat template, loses instruction-following
```

**✅ RIGHT:**
```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Generate a SAT item..."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
```

### 2. Monitor JSON Validity Rate During Training

**Why:** Validation loss doesn't correlate with structural quality.

**How:**
```python
def compute_json_validity(predictions: list) -> float:
    valid_count = 0
    for pred in predictions:
        try:
            json.loads(pred)
            valid_count += 1
        except:
            pass
    return valid_count / len(predictions)
```

**Use as early stopping metric:**
```python
EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.01
)
# Metric: json_validity_rate
```

### 3. Save Checkpoints Frequently

**Strategy:**
- Save every 500 steps
- Keep best 3 checkpoints (by JSON validity rate)
- Save final checkpoint at end of training

**Why:** Training can diverge. Frequent checkpoints enable rollback.

### 4. Validate on Held-Out Set

**Never train on 100% of data.**

**Always split:**
- Train: 85%
- Validation: 15%

**Validation set purposes:**
- Early stopping
- Hyperparameter tuning
- Detect overfitting

### 5. Use Gradient Checkpointing for Long Contexts

**For RW items (4096 tokens):**

```python
training_args = TrainingArguments(
    ...,
    gradient_checkpointing=True,  # Trade compute for memory
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**Enables:**
- Training on longer sequences
- Larger batch sizes
- Training on smaller GPUs

**Trade-off:**
- ~20% slower training
- Enables training that would otherwise OOM (out of memory)

### 6. Set Random Seeds for Reproducibility

```python
import torch
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)  # Before training
```

**Why:** Same seed + same data = same checkpoint (within numerical tolerance)

---

## Troubleshooting

### Problem: Model Generates Invalid JSON

**Symptoms:**
- JSON validity rate < 50%
- Mismatched braces, missing quotes, truncated strings

**Solutions:**

1. **Check training data quality**
   ```bash
   python scripts/validate_training_data.py data/training/rw_train.jsonl
   ```

2. **Increase training epochs**
   - Model may need more exposure to learn JSON structure
   - Try 5 epochs instead of 3

3. **Increase sequence length**
   - Items may be getting truncated
   - Try `max_seq_length: 6144` for RW

4. **Add JSON repair post-processing**
   ```python
   import json
   import json_repair  # pip install json_repair

   def repair_json(json_string: str) -> dict:
       try:
           return json.loads(json_string)
       except:
           return json_repair.loads(json_string)
   ```

### Problem: Model Repeats Instructions in Output

**Symptoms:**
```
"Generate a SAT item about quadratic equations. Here's the item: {...}"
```

**Causes:**
- Not using chat template
- Computing loss on instruction tokens
- Learning rate too high

**Solutions:**

1. **Use chat template** (see Best Practices #1)

2. **Mask loss on instruction tokens**
   ```python
   # In dataset __getitem__
   labels[:instruction_end] = -100
   ```

3. **Lower learning rate**
   ```yaml
   learning_rate: 1e-5  # Instead of 2e-5
   ```

### Problem: Loss Diverges (NaN or Inf)

**Symptoms:**
- Training loss suddenly becomes NaN
- Gradients explode

**Causes:**
- Learning rate too high
- Gradient accumulation steps too high
- Batch size too small (high variance)

**Solutions:**

1. **Lower learning rate**
   ```yaml
   learning_rate: 1e-5  # Half the default
   ```

2. **Reduce gradient accumulation**
   ```yaml
   gradient_accumulation_steps: 4  # Instead of 8
   ```

3. **Increase batch size**
   ```yaml
   per_device_batch_size: 16  # Instead of 8
   ```

4. **Add gradient clipping**
   ```python
   training_args = TrainingArguments(
       ...,
       max_grad_norm: 1.0  # Clip gradients at norm 1.0
   )
   ```

### Problem: Out of Memory (OOM)

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes mid-epoch

**Solutions:**

1. **Use QLoRA** (4-bit quantization)
   ```yaml
   load_in_4bit: true
   ```

2. **Enable gradient checkpointing**
   ```yaml
   gradient_checkpointing: true
   ```

3. **Reduce batch size**
   ```yaml
   per_device_batch_size: 4  # Instead of 8
   ```

4. **Reduce sequence length**
   ```yaml
   max_seq_length: 2048  # Instead of 4096
   ```

5. **Use smaller model** (temporary)
   - Try Qwen2.5-3B instead of 7B for debugging

### Problem: Model Generates Items in Wrong Domain

**Symptoms:**
- Prompt: "algebra.quadratic_equations"
- Output: "geometry.trigonometry" item

**Causes:**
- Training data has domain labels
- Model not learning to follow domain constraint

**Solutions:**

1. **Check training data**
   - Ensure domain labels are consistent
   - Remove mislabeled examples

2. **Emphasize domain in prompt**
   ```python
   user_message = f"Generate a SAT Math item.\n\n"
   user_message += f"CRITICAL: Domain must be {domain}\n"
   user_message += f"Difficulty: {difficulty}\n"
   ```

3. **Add domain validation in post-processing**
   ```python
   def validate_domain(item: dict, expected_domain: str) -> bool:
       return item["domain"] == expected_domain
   ```

---

## Summary

The Generation Service is the core of IIAS — it trains and hosts the models that generate SAT questions.

**Key takeaways:**

1. **Two specialized models** (RW + Math) perform better than one general model
2. **Fine-tuning** teaches models SAT-specific patterns and JSON formatting
3. **LoRA/QLoRA** enables efficient training on smaller GPUs
4. **Chat templates** preserve model's instruction-following capabilities
5. **Loss masking** ensures model learns to generate items, not predict instructions
6. **JSON validity rate** is the right metric for early stopping
7. **Frequent checkpoints** enable rollback if training diverges

**Next steps:**
- Read [`docs/02-auto-qa-service.md`](./02-auto-qa-service.md) to learn how generated items are validated
- Read [`docs/03-item-bank.md`](./03-item-bank.md) to learn how items are stored

---

**Document version:** 1.0
**Last updated:** 2026-03-27
**Author:** IIAS Development Team
