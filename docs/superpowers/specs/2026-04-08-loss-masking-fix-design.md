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

# Load JSONL directly - no preprocessing needed
train_dataset = load_dataset(
    "json",
    data_files=str(train_file),
    split="train"
)

# SFTTrainer handles everything automatically
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
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

### Unit Test: Verify Loss Masking

**File:** `tests/test_loss_masking.py`

```python
import pytest
import torch
from transformers import AutoTokenizer
from trl import SFTTrainer
from datasets import Dataset

def test_loss_masking_only_computes_on_assistant_tokens():
    """Test that loss is only computed on assistant tokens."""
    # Create simple test dataset
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

    # Load small model for testing
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Train one step
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=128,
        dataset_text_field="messages"
    )

    # Get labels from first batch
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]

    # Verify non-assistant tokens are masked
    # (labels should be -100 for system/user tokens)
    assert (labels[0] == -100).any(), "System/user tokens should be masked"

    # Verify assistant tokens are not masked
    # (labels should have actual token IDs for assistant response)
    assert (labels[0] != -100).any(), "Assistant tokens should not be masked"
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
try:
    from trl import SFTTrainer
except ImportError as e:
    raise ImportError(
        "trl library is required for training. "
        "Install with: pip install trl"
    ) from e

# Validate dataset format
if "messages" not in dataset.column_names:
    raise ValueError(
        f"Dataset must have 'messages' field. "
        f"Found columns: {dataset.column_names}"
    )
```

## Migration Steps

1. **Update dependencies** (5 min)
   ```bash
   echo "trl>=0.7.0" >> requirements.txt
   pip install trl
   ```

2. **Modify train_huggingface.py** (30 min)
   - Replace imports
   - Remove `create_dataset_from_jsonl()` function
   - Replace `Trainer` with `SFTTrainer`
   - Update dataset loading to use `load_dataset("json", ...)`

3. **Deprecate SFTDataset** (10 min)
   - Add deprecation warning
   - Update docstring
   - Keep code functional

4. **Add tests** (30 min)
   - Create `tests/test_loss_masking.py`
   - Implement unit test
   - Verify with short training run

5. **Verify** (15 min)
   - Run integration test
   - Generate sample outputs
   - Confirm loss decreases

**Total estimated time:** 1.5-2 hours

## Rollback Plan

If something goes wrong:

1. **Git revert** - Changes are isolated to two files
   ```bash
   git checkout HEAD -- scripts/train_huggingface.py src/training/dataset.py
   ```

2. **Fallback script** - `scripts/train_model.py` still exists (legacy)

3. **Data unchanged** - No modification to training data required

## Success Criteria

✅ **Loss masking verified** - Unit test confirms loss only on assistant tokens
✅ **Training runs** - Integration test completes without errors
✅ **Loss decreases** - Training loss decreases over steps
✅ **Model generates** - Can generate valid SAT questions
✅ **Tests pass** - All unit and integration tests pass
✅ **Zero data changes** - Existing JSONL files work without modification

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
