# Loss Masking Fix Design

**Date:** 2026-04-08
**Author:** Claude (with user approval)
**Status:** Approved Design

---

## Overview

Migrate from the custom `SFTDataset` class (which has broken loss masking) to HuggingFace's `trl.SFTTrainer`, which correctly implements loss masking for supervised fine-tuning (SFT). This fixes the critical issue where loss is computed on all tokens instead of just assistant responses.

## Problem Statement

The current `SFTDataset._create_labels()` method contains a placeholder implementation that returns unmasked labels, causing the training pipeline to compute loss on ALL tokens including:

- System messages
- User prompts
- Assistant responses

This is fundamentally broken because:
1. **Teaches the model to generate prompts instead of answers** - 70-80% of training compute is wasted on irrelevant tokens
2. **Violates SFT principles** - Supervised fine-tuning should only compute loss on model outputs (assistant responses)
3. **Produces poor quality models** - Models learn to repeat prompts rather than generate useful responses

### Current Broken Implementation

```python
def _create_labels(self, prompt: str, input_ids: torch.Tensor) -> torch.Tensor:
    # ... 80+ lines of comments explaining what should be done ...

    # Placeholder: return labels as-is (compute loss on all tokens)
    # This is not ideal but will work for basic training
    return labels  # ← BROKEN: No masking applied
```

### Correct Behavior Required

For a conversation like:
```
System: "You are a SAT tutor"
User: "Generate an algebra question"
Assistant: "If x² - 5x + 6 = 0, which of the following..."
```

Loss should be computed **only** on the assistant's response, not the system or user messages.

## Solution

Migrate to `trl.SFTTrainer`, which is the industry-standard library for SFT training and correctly implements loss masking out of the box.

### Why trl.SFTTrainer?

1. **Battle-tested** - Used by thousands of ML teams for SFT training
2. **Correct implementation** - Handles all edge cases (different chat templates, special tokens, multi-turn conversations)
3. **Zero data changes** - Natively supports the current JSONL format with "messages" field
4. **Less code to maintain** - Replaces custom dataset code with well-maintained library
5. **Actively maintained** - HuggingFace team continuously updates and fixes bugs

### How It Works for Digital SAT Questions

The training data format:
```json
{
  "messages": [
    {"role": "system", "content": "You are a SAT question generator..."},
    {"role": "user", "content": "Generate a hard algebra question..."},
    {"role": "assistant", "content": "If x² - 5x + 6 = 0, which of the following...\n\nChoices:\nA) ...\nB) ...\nC) ...\nD) ...\n\nCorrect Answer: C\n\nRationale: Factoring gives..."}
  ]
}
```

`trl.SFTTrainer` automatically:
- ✅ Masks the **system message** (no loss computed)
- ✅ Masks the **user prompt** (no loss computed)
- ✅ Computes loss on the **entire assistant response** (question + choices + answer + rationale)

This is correct for SAT question generation because:
- The model learns to generate the COMPLETE question package
- All parts of the assistant's response contribute to training
- The model learns: "Given this prompt → produce this full response"

## Architecture

### Components to Modify

**1. `scripts/train_huggingface.py`**
- Replace `transformers.Trainer` with `trl.SFTTrainer`
- Remove custom dataset creation logic (`create_dataset_from_jsonl`)
- Pass JSONL data directly to SFTTrainer with `dataset_text_field="messages"`
- Remove manual label masking logic

**2. `src/training/dataset.py`**
- Add deprecation warning to `SFTDataset` class
- Add docstring directing users to use `trl.SFTTrainer` instead
- Keep the class functional for backward compatibility

**3. `requirements.txt`**
- Add `trl` library if not already present

### Data Flow

**Before (broken):**
```
JSONL → SFTDataset → manual tokenization → broken masking → Trainer
```

**After (fixed):**
```
JSONL → SFTTrainer → automatic tokenization → correct masking → training
```

## Implementation Details

### Key Changes in `train_huggingface.py`

**Remove:**
```python
from transformers import Trainer
from datasets import Dataset

# Custom dataset creation (~50 lines)
def create_dataset_from_jsonl(data_path, tokenizer, max_length):
    # ... custom logic
    dataset = Dataset.from_dict({...})

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
```

**Add:**
```python
from trl import SFTTrainer
from datasets import load_dataset

# Load training data - no preprocessing needed
train_dataset = load_dataset(
    "json",
    data_files=str(train_file),
    split="train"
)

# Load validation data if exists
val_dataset = None
if val_file.exists():
    val_dataset = load_dataset(
        "json",
        data_files=str(val_file),
        split="train"
    )

# Verify tokenizer has chat template support
if not hasattr(tokenizer, 'apply_chat_template'):
    raise ValueError(
        f"Tokenizer for {model_name} doesn't support chat templates. "
        f"SFTTrainer requires a tokenizer with chat_template attribute."
    )

# SFTTrainer handles everything automatically
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Validation dataset
    dataset_text_field="messages",  # Use the 'messages' field from JSONL
    max_seq_length=max_length,
    tokenizer=tokenizer,
)
```

### Deprecation in `dataset.py`

Add warning and docstring to `SFTDataset`:

```python
class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning Dataset for chat models.

    [Existing documentation...]

    .. deprecated::
        This class has incorrect loss masking implementation.
        Use `trl.SFTTrainer` instead, which handles SFT training correctly.

        This class is kept for backward compatibility only and will be
        removed in a future version.

        See: https://huggingface.co/docs/trl/main/en/sft_trainer

    Examples:
        Use SFTTrainer instead:

        >>> from trl import SFTTrainer
        >>> from datasets import load_dataset
        >>>
        >>> dataset = load_dataset("json", data_files="data.jsonl", split="train")
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     train_dataset=dataset,
        ...     dataset_text_field="messages"
        ... )
    """

    def __init__(self, ...):
        logger.warning(
            "SFTDataset is deprecated due to incorrect loss masking. "
            "Use trl.SFTTrainer instead. "
            "See: https://huggingface.co/docs/trl/main/en/sft_trainer"
        )
        # ... rest of existing code
```

## Testing Strategy

### Pre-Training Verification

**Verify chat template compatibility:**
```bash
# Test that both models support required chat template
python -c "
from transformers import AutoTokenizer

# Test phi-4
print('Testing phi-4 tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-4')
test_msg = [{'role': 'user', 'content': 'test'}]
result = tokenizer.apply_chat_template(test_msg, tokenize=False)
print(f'✓ phi-4 supports chat templates')

# Test Qwen2.5
print('Testing Qwen2.5 tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
result = tokenizer.apply_chat_template(test_msg, tokenize=False)
print(f'✓ Qwen2.5 supports chat templates')
"
```

**Verify data format:**
```python
# scripts/verify_training_data_format.py
import json
import sys
from pathlib import Path

def verify_data_format(data_file: Path):
    """Verify training data has correct format for SFTTrainer."""
    with open(data_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            example = json.loads(line)

            # Check for messages field
            if "messages" not in example:
                print(f"❌ Line {line_num}: Missing 'messages' field. Found: {list(example.keys())}")
                return False

            # Check messages is a list
            if not isinstance(example["messages"], list):
                print(f"❌ Line {line_num}: 'messages' must be a list")
                return False

            # Check each message has role and content
            for msg in example["messages"]:
                if "role" not in msg or "content" not in msg:
                    print(f"❌ Line {line_num}: Message missing 'role' or 'content'")
                    return False

    print(f"✓ {data_file} has correct format")
    return True

if __name__ == "__main__":
    data_files = sys.argv[1:] if len(sys.argv) > 1 else ["data/splits/math_train.jsonl"]
    all_valid = all(verify_data_format(Path(f)) for f in data_files)
    sys.exit(0 if all_valid else 1)
```

### Unit Test: Verify SFTTrainer Configuration

**File:** `tests/test_loss_masking.py`

```python
import pytest
from unittest.mock import Mock, patch
from trl import SFTTrainer
from datasets import Dataset


def test_sft_trainer_configuration():
    """Test that SFTTrainer is correctly configured for SFT training."""
    # Create test dataset
    data = {
        "messages": [
            [
                {"role": "system", "content": "You are a tutor"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4"}
            ]
        ]
    }
    dataset = Dataset.from_dict(data)

    # Mock model and tokenizer
    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.config.vocab_size = 1000

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.apply_chat_template.return_value = "formatted text"

    # Create trainer
    trainer = SFTTrainer(
        model=mock_model,
        train_dataset=dataset,
        max_seq_length=128,
        dataset_text_field="messages",
        tokenizer=mock_tokenizer
    )

    # Verify trainer is configured correctly
    assert hasattr(trainer, 'train_dataset')
    assert trainer.train_dataset == dataset
    assert trainer.max_seq_length == 128


def test_sft_trainer_with_validation_dataset():
    """Test that SFTTrainer handles validation datasets correctly."""
    train_data = {
        "messages": [[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}
        ]]
    }
    val_data = {
        "messages": [[
            {"role": "user", "content": "Val Question"},
            {"role": "assistant", "content": "Val Answer"}
        ]]
    }

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.config.vocab_size = 1000

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1

    trainer = SFTTrainer(
        model=mock_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=128,
        dataset_text_field="messages",
        tokenizer=mock_tokenizer
    )

    assert trainer.eval_dataset == val_dataset
```

### Integration Test: End-to-End Training

**Command:**
```bash
# Run short training to verify everything works
python scripts/train_huggingface.py \
  --section math \
  --env production \
  --max-steps 10 \
  --output-dir test_run
```

**Verify:**
- ✅ Training completes without errors
- ✅ Loss decreases over steps
- ✅ Checkpoint is saved
- ✅ Model can generate responses

## Error Handling

### Potential Issues and Solutions

| Issue | Detection | Handling |
|-------|-----------|----------|
| `trl` not installed | ImportError on import | Clear error: "Install trl: pip install trl" |
| Invalid data format | JSON decode error | Show expected format with example |
| Missing "messages" field | KeyError in dataset | Specific error about required field |
| Empty dataset | Dataset length check | Error: "No training examples found" |
| Tokenizer lacks chat template | AttributeError | Use simple format fallback |

**Implementation:**
```python
# Check trl is installed
try:
    from trl import SFTTrainer
except ImportError as e:
    raise ImportError(
        "trl library is required for training. "
        "Install with: pip install trl"
    ) from e

# Verify tokenizer has chat template
if not hasattr(tokenizer, 'apply_chat_template'):
    raise ValueError(
        f"Tokenizer for {model_name} doesn't support chat templates. "
        f"SFTTrainer requires a tokenizer with chat_template attribute."
    )

# Validate dataset format
if "messages" not in dataset.column_names:
    raise ValueError(
        f"Dataset must have 'messages' field. "
        f"Found columns: {dataset.column_names}"
    )

# Verify data format before training
import json
with open(train_file) as f:
    first_line = f.readline()
    example = json.loads(first_line)

    if "messages" not in example:
        raise ValueError(
            f"Training data must have 'messages' field. "
            f"Found keys: {list(example.keys())}"
        )

    if not isinstance(example["messages"], list):
        raise ValueError("'messages' must be a list")
```

## Migration Steps

1. **Update dependencies** (5 min)
   ```bash
   # Note: trl>=0.7.0 should already be in requirements.txt
   pip install trl
   ```

2. **Verify chat template compatibility** (5 min)
   ```bash
   # Test that both models support required chat template
   python scripts/verify_chat_templates.py
   ```

3. **Verify data format** (5 min)
   ```bash
   python scripts/verify_training_data_format.py data/splits/math_train.jsonl
   python scripts/verify_training_data_format.py data/splits/reading_writing_train.jsonl
   ```

4. **Modify train_huggingface.py** (30 min)
   - Replace imports
   - Remove `create_dataset_from_jsonl()` function
   - Replace `Trainer` with `SFTTrainer`
   - Update dataset loading to use `load_dataset("json", ...)`
   - Add validation dataset support
   - Add chat template verification
   - Add data format validation

5. **Deprecate SFTDataset** (10 min)
   - Add deprecation warning
   - Update docstring
   - Keep code functional

6. **Update tests** (30 min)
   - Create `tests/test_loss_masking.py`
   - Mark `tests/test_training_dataset.py` as testing deprecated functionality
   - Add deprecation notice to existing tests

7. **Verify** (30 min)
   - Run unit tests
   - Run short training (10 steps)
   - Verify loss decreases
   - Test generation quality

**Total estimated time:** 2-3 hours

## Rollback Plan

If something goes wrong:

1. **Git revert** - Changes are isolated to minimal files
   ```bash
   git checkout HEAD -- scripts/train_huggingface.py src/training/dataset.py
   ```

2. **Fallback script** - `scripts/train_model.py` still exists (legacy, but functional)

3. **Data unchanged** - No modification to training data required

4. **Verification** - Before rollback, verify old script still works
   ```bash
   # Quick test with legacy script
   python scripts/train_model.py --section math --max-steps 5
   ```

## Success Criteria

✅ **Chat templates verified** - Both phi-4 and Qwen2.5 support required templates
✅ **Data format validated** - All training files have correct "messages" format
✅ **Unit tests pass** - SFTTrainer configuration tests pass
✅ **Training runs** - Integration test completes without errors
✅ **Loss decreases** - Training loss decreases over steps (not just on all tokens)
✅ **Validation works** - Validation dataset loads and evaluation runs
✅ **Model generates** - Can generate valid SAT questions
✅ **Zero data changes** - Existing JSONL files work without modification
✅ **Old tests marked** - Existing SFTDataset tests marked as deprecated

## References

- [TRL SFTTrainer Documentation](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Supervised Fine-Tuning Guide](https://huggingface.co/docs/trl/en/sft_trainer#training-with-the-sfttrainer)
- [CODE_ASSESSMENT.md](../CODE_ASSESSMENT.md) - Lines 49-79, 185-260 (loss masking issue)

---

**Next Steps:**
1. ✅ Design approved
2. ⏭️ Create implementation plan using `superpowers:writing-plans` skill
3. ⏭️ Implement changes
4. ⏭️ Run tests
5. ⏭️ Verify with training run
