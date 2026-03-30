# MLX Trainer Design

**Date:** 2026-03-30
**Status:** Draft
**Author:** Claude (Sonnet 4.6)

---

## Overview

Implement a minimal viable training loop for Apple Silicon using MLX that validates the fine-tuning infrastructure works end-to-end. The trainer loads the 1.5B Qwen proxy model, trains on mock IIAS data, and saves checkpoints in MLX format.

This is **not** for production-quality training — it's a smoke test to verify the pipeline works before spending money on vast.ai GPUs.

---

## Architecture

### Components

1. **`convert_item_to_messages()`** — Utility function that converts IIAS schema items to chat format. Takes an item with `content_json` (question, choices, correct_answer, rationale) and formats it as a conversation between user and assistant.

2. **`MLXDataset`** — Dataset class that:
   - Reads JSONL files with IIAS items
   - Converts items to chat messages on-the-fly using `convert_item_to_messages()`
   - Tokenizes with the MLX tokenizer
   - Returns batches as MLX arrays (`mx.array`)

3. **`MLXTrainer`** — Main training class that:
   - Loads the model using mlx-lm's `load()` function
   - Runs training loop: forward pass → loss → backward → optimizer step
   - Saves checkpoints in MLX native format
   - Minimal logging (epoch start/end, loss, checkpoint saves)

### Data Flow

```
IIAS items (JSONL)
    ↓
MLXDataset.__getitem__()
    ↓
convert_item_to_messages()
    ↓
{"role": "user", "content": "Generate SAT item..."}
{"role": "assistant", "content": "Here's the item..."}
    ↓
Tokenizer → MLX arrays
    ↓
Training loop
    ↓
MLX checkpoint
```

---

## Implementation Details

### `src/training/converters.py`

```python
def convert_item_to_messages(item: dict) -> list[dict]:
    """Convert IIAS item to chat messages format."""
    section = "Reading & Writing" if item["section"] == "reading_writing" else "Math"

    user_msg = f"""Generate a SAT {section} item with these constraints:
- Domain: {item["domain"]}
- Difficulty: {item["difficulty"]}

Output JSON only."""

    assistant_msg = json.dumps(item["content_json"])

    return [
        {"role": "system", "content": "You are an expert SAT item writer."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ]
```

### `src/training/mlx_dataset.py`

```python
class MLXDataset:
    def __init__(self, data_path: str, tokenizer, max_seq_length: int):
        self.items = [json.loads(line) for line in open(data_path)]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        messages = convert_item_to_messages(item)

        # Tokenize with chat template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self.tokenizer(text, return_tensors="mx", truncation=True,
                                max_length=self.max_seq_length)

        return tokens["input_ids"], tokens["attention_mask"]
```

### `src/training/mlx_trainer.py` (fill existing stub)

```python
class MLXTrainer:
    def __init__(self, model, tokenizer, train_path, val_path, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = MLXDataset(train_path, tokenizer, config.max_seq_length)
        self.val_dataset = MLXDataset(val_path, tokenizer, config.max_seq_length) if val_path else None
        self.optimizer = mx.optim.AdamW(learning_rate=config.learning_rate)

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training loop
            total_loss = 0
            for step, (input_ids, attention_mask) in enumerate(self.train_dataset):
                # Forward pass
                loss = self._compute_loss(input_ids, attention_mask)

                # Backward pass
                loss.backward()
                self.optimizer.step(self.model)
                mx.eval(self.model, self.optimizer)  # Update parameters

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataset)
            logger.info(f"Average loss: {avg_loss:.4f}")

            # Validation
            if self.val_dataset:
                val_loss = self._validate()
                logger.info(f"Validation loss: {val_loss:.4f}")

            # Save checkpoint
            self._save_checkpoint(epoch, avg_loss)
```

**Key MLX-specific patterns:**
- `mx.eval()` after optimizer step to actually update parameters
- No `.to(device)` needed — MLX handles this automatically
- Use mlx-lm's `load()` function to get model + tokenizer together

---

## File Structure

**New files:**
```
src/training/
├── mlx_dataset.py      # MLXDataset class
├── mlx_trainer.py      # MLXTrainer class (fill in the stub)
└── converters.py       # convert_item_to_messages()

scripts/
└── train_local.py      # New CLI script for local training
```

**Why separate `train_local.py` instead of modifying `train.py`:**
- `train.py` is for production PyTorch training (vast.ai)
- `train_local.py` is for local MLX training (M4 MacBook)
- Each uses different backends with different APIs
- Cleaner separation — local vs production paths

---

## CLI Interface

**`scripts/train_local.py`:**

```bash
# Train math section locally
python scripts/train_local.py --section math --num_epochs 2

# Train RW with custom batch size
python scripts/train_local.py --section reading_writing --batch_size 2 --num_epochs 1
```

**Arguments:**
- `--section`: `math` or `reading_writing`
- `--num_epochs`: Number of training epochs (default: 2)
- `--batch_size`: Override config batch size if needed
- `--checkpoint-dir`: Where to save checkpoints (default: `outputs/checkpoints/`)

**What it does:**
1. Load config (`APP_ENV=local`)
2. Load model via mlx-lm (override to use `mlx-community/Qwen2.5-1.5B-Instruct-4bit`)
3. Create MLXTrainer with train/val datasets
4. Call `trainer.train(num_epochs)`
5. Log summary at the end

---

## Error Handling

**Graceful failures:**

1. **Missing data files** — Clear error message if `train.jsonl` or `val.jsonl` don't exist, with expected paths

2. **Model loading failures** — Catch MLX errors and provide context (wrong model ID, corrupted weights, etc.)

3. **OOM handling** — MLX on 8GB unified memory might hit limits. Catch and suggest reducing batch size or sequence length

4. **Empty datasets** — Warn if loaded dataset has 0 examples after filtering

**Minimal but informative:**
```python
try:
    model, tokenizer = load(model_id)
except Exception as e:
    logger.error(f"Failed to load model {model_id}: {e}")
    logger.info("Check that the model ID is correct and available on HuggingFace")
    sys.exit(1)
```

---

## Testing Strategy

### Integration Test (Smoke Test)

```bash
# This is what you'll run to verify everything works
python scripts/create_mock_data.py --section math --count 5
python scripts/train_local.py --section math --num_epochs 1
```

**What the smoke test validates:**
- Mock data generates successfully
- MLX model loads without errors
- Dataset tokenizes items correctly
- Training loop completes 1 epoch
- Checkpoint saves to disk
- No crashes or OOM errors

### Unit Tests (Optional)

1. **`test_converters.py`** — Test `convert_item_to_messages()`:
   - Validates IIAS item → chat format conversion
   - Checks required fields are preserved
   - Tests both math and RW sections

2. **`test_mlx_dataset.py`** — Test `MLXDataset`:
   - Verifies JSONL loading
   - Checks tokenization produces valid MLX arrays
   - Validates sequence truncation works

---

## Success Criteria

**The MLX trainer is working when:**

✅ Loads the 1.5B Qwen model in 4-bit quantization
✅ Converts IIAS items to chat format correctly
✅ Runs 1-2 epochs without crashing
✅ Loss decreases (even slightly)
✅ Checkpoint files are created and valid
✅ Logs show clear progress: epoch start, loss, checkpoint saves

**Red flags that mean something is broken:**
- Loss is `NaN` or `inf`
- Loss doesn't change at all (flat line)
- OOM errors on 8GB memory
- Checkpoint files are 0 bytes or corrupted
- Model refuses to load

---

## Key Design Decisions

### No LoRA Initially

MLX's LoRA implementation differs from PEFT/PyTorch. For infrastructure validation, full fine-tuning the 1.5B model is acceptable. Can add LoRA later if needed.

### MLX Arrays Throughout

Unlike PyTorch which uses `.to(device)`, MLX arrays are automatically on the GPU/Unified Memory. Simpler code, no device management.

### Simple Optimizer

Use MLX's built-in `AdamW` or `SGD`. No learning rate scheduler for the smoke test — constant LR is fine.

### MLX Native Checkpoint Format

Save using MLX's `save_model()` which creates `.safetensors` files. Simple, and you're validating infrastructure — not actually transferring weights between environments.

---

## Edge Cases & Considerations

**Memory management:**
- 8GB unified memory is tight. Default batch size should be 1 or 2
- If OOM, suggest user reduces `max_seq_length` or batch size

**Model ID handling:**
- Local config has production model IDs (Qwen 7B, phi-4)
- For MLX, override to use `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- Add comment explaining why

**Data format mismatch:**
- If someone passes real training data (not mock), it might still be IIAS format
- MLXDataset should handle this gracefully
- Add clear error if data format is unexpected

---

## Implementation Order

1. **`converters.py`** — Simple utility, no dependencies
2. **`mlx_dataset.py`** — Depends on converters
3. **`mlx_trainer.py`** — Fill in the stub, depends on dataset
4. **`train_local.py`** — CLI script that wires it all together
5. **Test with mock data** — Run the smoke test

Each step builds on the previous one, and you can test incrementally.
