# MLX Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a minimal viable MLX training loop for Apple Silicon to validate the fine-tuning infrastructure works end-to-end before running production training on vast.ai.

**Architecture:** Three-layer approach — utility converter (IIAS→chat), MLX-native dataset class, and trainer class that wraps mlx-lm model loading with a simple training loop. All files use MLX APIs (mx.array, mx.optim, mx.eval) instead of PyTorch.

**Tech Stack:** MLX 0.31+, mlx-lm 0.31+, Python 3.14+, loguru for logging

---

## File Structure

```
src/training/
├── converters.py         # NEW - convert_item_to_messages() utility
├── mlx_dataset.py        # NEW - MLXDataset class
├── mlx_trainer.py        # MODIFY - fill existing stub with MLXTrainer class
└── __init__.py           # MODIFY - export new classes

scripts/
├── train_local.py        # NEW - CLI script for local MLX training
└── create_mock_data.py   # EXISTS - created earlier for smoke testing

tests/
├── test_converters.py    # NEW - test IIAS→chat conversion
└── test_mlx_dataset.py   # NEW - test MLXDataset
```

---

## Task 1: Create converters.py with IIAS→chat conversion

**Files:**
- Create: `src/training/converters.py`
- Test: `tests/test_converters.py`

- [ ] **Step 1: Write failing test for converter**

```python
# tests/test_converters.py
import pytest
from src.training.converters import convert_item_to_messages

def test_convert_math_item():
    item = {
        "section": "math",
        "domain": "algebra.linear_equations_one_variable",
        "difficulty": "easy",
        "content_json": {
            "question": "Solve 2x + 5 = 15",
            "choices": [
                {"label": "A", "text": "x = 3"},
                {"label": "B", "text": "x = 4"},
                {"label": "C", "text": "x = 5"},
                {"label": "D", "text": "x = 6"}
            ],
            "correct_answer": "C",
            "correct_answer_text": "x = 5",
            "rationale": "Subtract 5, divide by 2",
            "math_format": "latex"
        }
    }

    messages = convert_item_to_messages(item)

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert "Math" in messages[1]["content"]
    assert "algebra.linear_equations_one_variable" in messages[1]["content"]
    assert "easy" in messages[1]["content"]

def test_convert_rw_item():
    item = {
        "section": "reading_writing",
        "domain": "standard_english_conventions.boundaries",
        "difficulty": "medium",
        "content_json": {
            "question": "Choose the correct punctuation",
            "choices": [
                {"label": "A", "text": "Period"},
                {"label": "B", "text": "Comma"},
                {"label": "C", "text": "Semicolon"},
                {"label": "D", "text": "Colon"}
            ],
            "correct_answer": "C",
            "correct_answer_text": "Semicolon",
            "rationale": "Separates independent clauses"
        }
    }

    messages = convert_item_to_messages(item)

    assert messages[0]["role"] == "system"
    assert "Reading & Writing" in messages[1]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_converters.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.training.converters'"

- [ ] **Step 3: Create converters.py with minimal implementation**

```python
# src/training/converters.py
import json
from typing import List, Dict, Any

def convert_item_to_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert IIAS schema item to chat messages format.

    Args:
        item: IIAS item with section, domain, difficulty, content_json

    Returns:
        List of message dicts with role and content
    """
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_converters.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/training/converters.py tests/test_converters.py
git commit -m "feat: add IIAS to chat message converter"
```

---

## Task 2: Create MLXDataset class

**Files:**
- Create: `src/training/mlx_dataset.py`
- Test: `tests/test_mlx_dataset.py`

- [ ] **Step 1: Write failing test for MLXDataset**

```python
# tests/test_mlx_dataset.py
import pytest
import json
import tempfile
from pathlib import Path
from src.training.mlx_dataset import MLXDataset

@pytest.fixture
def sample_data_file():
    """Create a temporary JSONL file with sample data."""
    items = [
        {
            "section": "math",
            "domain": "algebra.linear_equations_one_variable",
            "difficulty": "easy",
            "content_json": {
                "question": "Solve 2x + 5 = 15",
                "choices": [{"label": "A", "text": "x=3"}],
                "correct_answer": "A",
                "correct_answer_text": "x=3",
                "rationale": "Test"
            }
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
        return f.name

def test_dataset_load(sample_data_file):
    """Test that dataset loads items correctly."""
    # This will fail until we create MLXDataset
    # We'll need a mock tokenizer for this test
    pass  # Will implement in step 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mlx_dataset.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.training.mlx_dataset'"

- [ ] **Step 3: Create MLXDataset with minimal implementation**

```python
# src/training/mlx_dataset.py
import json
from typing import Tuple, Dict, Any
from pathlib import Path
from loguru import logger

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from src.training.converters import convert_item_to_messages


class MLXDataset:
    """
    Dataset class for MLX training.

    Loads IIAS items from JSONL and converts to chat format on-the-fly.
    Returns MLX arrays (mx.array) instead of PyTorch tensors.
    """

    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 2048):
        """
        Initialize MLXDataset.

        Args:
            data_path: Path to JSONL file with IIAS items
            tokenizer: MLX tokenizer with chat_template
            max_seq_length: Maximum sequence length for tokenization
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MLXDataset")

        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load items
        self.items = self._load_items()
        logger.info(f"Loaded {len(self.items)} items from {data_path}")

    def _load_items(self) -> list:
        """Load items from JSONL file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        items = []
        with open(self.data_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        return items

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """
        Get a single item as MLX arrays.

        Returns:
            Tuple of (input_ids, attention_mask) as mx.array
        """
        item = self.items[idx]

        # Convert to chat messages
        messages = convert_item_to_messages(item)

        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors="mx",
            truncation=True,
            max_length=self.max_seq_length
        )

        return tokens["input_ids"], tokens["attention_mask"]
```

- [ ] **Step 4: Update test to use mock tokenizer and verify it passes**

```python
# tests/test_mlx_dataset.py
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
from src.training.mlx_dataset import MLXDataset

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.apply_chat_template = Mock(return_value="System: You are expert.\n\nUser: Generate item.\n\nAssistant: {\"question\": \"Test\"}")
    tokenizer.__call__ = Mock(return_value={"input_ids": "mock_ids", "attention_mask": "mock_mask"})
    return tokenizer

@pytest.fixture
def sample_data_file():
    """Create a temporary JSONL file with sample data."""
    items = [
        {
            "section": "math",
            "domain": "algebra.linear_equations_one_variable",
            "difficulty": "easy",
            "content_json": {
                "question": "Solve 2x + 5 = 15",
                "choices": [{"label": "A", "text": "x=3"}],
                "correct_answer": "A",
                "correct_answer_text": "x=3",
                "rationale": "Test"
            }
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
        path = f.name
    yield path
    Path(path).unlink()

def test_dataset_length(sample_data_file, mock_tokenizer):
    """Test that dataset reports correct length."""
    dataset = MLXDataset(sample_data_file, mock_tokenizer, max_seq_length=512)
    assert len(dataset) == 1

def test_dataset_getitem(sample_data_file, mock_tokenizer):
    """Test that dataset returns tokenized items."""
    dataset = MLXDataset(sample_data_file, mock_tokenizer, max_seq_length=512)

    input_ids, attention_mask = dataset[0]

    assert input_ids == "mock_ids"
    assert attention_mask == "mock_mask"

    # Verify tokenizer was called correctly
    mock_tokenizer.apply_chat_template.assert_called_once()
    mock_tokenizer.__call__.assert_called_once()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_mlx_dataset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/training/mlx_dataset.py tests/test_mlx_dataset.py
git commit -m "feat: add MLXDataset class for MLX training"
```

---

## Task 3: Implement MLXTrainer class (fill existing stub)

**Files:**
- Modify: `src/training/mlx_trainer.py`
- Test: `tests/test_mlx_trainer.py` (integration-style test)

- [ ] **Step 1: Read existing stub to understand structure**

Run: `head -20 src/training/mlx_trainer.py`
Expected: File is empty or has minimal content

- [ ] **Step 2: Create integration test for trainer (no actual MLX, just structure)**

```python
# tests/test_mlx_trainer.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.training.mlx_trainer import MLXTrainer

def test_trainer_initialization():
    """Test that trainer can be initialized with mock components."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    # Create temporary data files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as train_f:
        train_f.write('{"section": "math", "domain": "test", "difficulty": "easy", "content_json": {"question": "Test", "choices": [], "correct_answer": "A", "correct_answer_text": "A", "rationale": "Test"}}\n')
        train_path = train_f.name

    try:
        # Mock MLX availability
        with patch('src.training.mlx_trainer.MLX_AVAILABLE', True):
            with patch('src.training.mlx_dataset.MLX_AVAILABLE', True):
                trainer = MLXTrainer(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    train_path=train_path,
                    val_path=None,
                    config=Mock(learning_rate=1e-4, max_seq_length=512)
                )

                assert trainer.model == mock_model
                assert trainer.tokenizer == mock_tokenizer
    finally:
        Path(train_path).unlink()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_mlx_trainer.py -v`
Expected: FAIL with actual error from MLXTrainer not existing or wrong structure

- [ ] **Step 4: Implement MLXTrainer class**

```python
# src/training/mlx_trainer.py
"""
MLX Trainer for local Apple Silicon training.

Minimal viable training loop using MLX APIs.
"""

from pathlib import Path
from loguru import logger

try:
    import mlx.core as mx
    from mlx.nn import Optimizer
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from src.training.mlx_dataset import MLXDataset


class MLXTrainer:
    """
    Minimal MLX training loop.

    Supports forward/backward, optimizer steps, validation, and checkpointing.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_path: str,
        val_path: str = None,
        config = None
    ):
        """
        Initialize MLXTrainer.

        Args:
            model: MLX model loaded via mlx_lm.load()
            tokenizer: MLX tokenizer
            train_path: Path to training JSONL
            val_path: Optional path to validation JSONL
            config: Config object with learning_rate, max_seq_length
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MLXTrainer")

        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Load datasets
        logger.info(f"Loading training data from {train_path}")
        self.train_dataset = MLXDataset(
            train_path,
            tokenizer,
            config.max_seq_length
        )

        self.val_dataset = None
        if val_path and Path(val_path).exists():
            logger.info(f"Loading validation data from {val_path}")
            self.val_dataset = MLXDataset(
                val_path,
                tokenizer,
                config.max_seq_length
            )

        # Setup optimizer
        self.optimizer = mx.optim.AdamW(learning_rate=config.learning_rate)

        logger.info(f"Training examples: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Validation examples: {len(self.val_dataset)}")

    def train(self, num_epochs: int, checkpoint_dir: str = None):
        """
        Run training loop.

        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        else:
            checkpoint_path = None

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training
            self.model.train()
            total_loss = 0

            for step, (input_ids, attention_mask) in enumerate(self.train_dataset):
                # Forward pass and loss computation
                loss = self._compute_loss(input_ids, attention_mask)

                # Backward pass
                loss.backward()

                # Optimizer step
                self.optimizer.step(self.model)

                # Update parameters (MLX-specific)
                mx.eval(self.model, self.optimizer)

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataset)
            logger.info(f"Average training loss: {avg_loss:.4f}")

            # Validation
            if self.val_dataset:
                val_loss = self._validate()
                logger.info(f"Validation loss: {val_loss:.4f}")

            # Save checkpoint
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, epoch, avg_loss)

        logger.info("Training complete!")

    def _compute_loss(self, input_ids, attention_mask):
        """
        Compute forward pass loss for causal language model.

        Note: This implementation assumes the MLX model follows the causal LM pattern
        where we compute cross-entropy loss between predictions and input_ids (shifted).
        The exact implementation depends on the model structure returned by mlx_lm.load().

        This is a basic implementation - you may need to adjust based on the actual
        model forward pass signature.
        """
        # For causal LM: predict next token, compute loss against input
        # The model should return a loss or we compute it manually
        try:
            # Try standard causal LM forward pass
            # MLX models typically expect: input_ids, attention_mask
            # And return either a loss directly or logits
            outputs = self.model(input_ids, attention_mask)

            # If model returns loss directly
            if isinstance(outputs, mx.array) and outputs.ndim == 0:
                return outputs

            # If model returns logits, compute cross-entropy loss
            # Shift logits and labels for next-token prediction
            logits = outputs
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            # Compute cross-entropy loss
            # This is a simplified version - MLX may have optimized loss functions
            from mlx.nn.losses import cross_entropy
            loss = cross_entropy(shift_logits, shift_labels)
            return loss.mean()

        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            logger.info(
                "The model structure from mlx_lm.load() may differ from expected. "
                "You may need to inspect model.forward() and adjust _compute_loss() accordingly."
            )
            raise

    def _validate(self):
        """Run validation loop."""
        self.model.eval()
        total_loss = 0

        for input_ids, attention_mask in self.val_dataset:
            loss = self._compute_loss(input_ids, attention_mask)
            total_loss += loss.item()

        return total_loss / len(self.val_dataset)

    def _save_checkpoint(self, checkpoint_path: Path, epoch: int, loss: float):
        """Save model checkpoint."""
        epoch_dir = checkpoint_path / f"epoch-{epoch + 1}"
        epoch_dir.mkdir(exist_ok=True)

        # Save using MLX's save_model
        try:
            from mlx_lm import save_model
            save_model(self.model, self.tokenizer, str(epoch_dir))
            logger.info(f"Saved checkpoint: {epoch_dir}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
```

- [ ] **Step 5: Run test to verify structure passes**

Run: `pytest tests/test_mlx_trainer.py::test_trainer_initialization -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/training/mlx_trainer.py tests/test_mlx_trainer.py
git commit -m "feat: implement MLXTrainer class"
```

---

## Task 4: Create train_local.py CLI script

**Files:**
- Create: `scripts/train_local.py`

- [ ] **Step 1: Create train_local.py script**

```python
#!/usr/bin/env python3
"""
Local MLX training script for Apple Silicon.

Usage:
    python scripts/train_local.py --section math --num_epochs 2
    python scripts/train_local.py --section reading_writing --batch_size 2
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

try:
    from mlx_lm import load
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logger.error("mlx-lm not available. Install with: pip install mlx-lm")

from src.config import load_config
from src.training.mlx_trainer import MLXTrainer


def main():
    parser = argparse.ArgumentParser(description="Train SAT models locally with MLX")
    parser.add_argument(
        "--section",
        required=True,
        choices=["math", "reading_writing", "rw", "readingwriting"],
        help="SAT section to train"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (default: from config)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: outputs/checkpoints/)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        help="MLX model ID (default: mlx-community/Qwen2.5-1.5B-Instruct-4bit)"
    )

    args = parser.parse_args()

    # Set environment
    os.environ["APP_ENV"] = "local"

    # Normalize section name
    section = args.section
    if section in ["rw", "readingwriting"]:
        section = "reading_writing"

    # Load config
    config = load_config()

    # Override model ID for local MLX training
    logger.info(f"Loading model: {args.model_id}")
    logger.info("Note: Using MLX proxy model for infrastructure validation")

    if not MLX_LM_AVAILABLE:
        logger.error("mlx-lm is required. Install with: pip install mlx-lm")
        sys.exit(1)

    # Load model and tokenizer
    try:
        model, tokenizer = load(args.model_id)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Check that the model ID is correct and available on HuggingFace")
        sys.exit(1)

    # Setup data paths
    data_dir = Path(config.paths.data_dir) / "splits"
    train_file = data_dir / f"{section}_train.jsonl"
    val_file = data_dir / f"{section}_val.jsonl"

    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.info("Run: python scripts/create_mock_data.py --section <section>")
        sys.exit(1)

    # Setup checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(config.paths.checkpoint_dir) / section / timestamp

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create config object for trainer
    class TrainerConfig:
        def __init__(self, config):
            self.learning_rate = config.training.learning_rate
            self.max_seq_length = (
                config.training.max_seq_length_rw
                if section == "reading_writing"
                else config.training.max_seq_length_math
            )

    trainer_config = TrainerConfig(config)

    # Create trainer
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_path=str(train_file),
        val_path=str(val_file) if val_file.exists() else None,
        config=trainer_config
    )

    # Train
    logger.info("=" * 50)
    logger.info(f"Starting training: {section}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info("=" * 50)

    try:
        trainer.train(num_epochs=args.num_epochs, checkpoint_dir=str(checkpoint_dir))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Training successful!")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make script executable**

Run: `chmod +x scripts/train_local.py`
Expected: No output (script is now executable)

- [ ] **Step 3: Verify script loads without errors (dry run)**

Run: `python scripts/train_local.py --help`
Expected: Help text displays with all arguments listed

- [ ] **Step 4: Commit**

```bash
git add scripts/train_local.py
git commit -m "feat: add local MLX training CLI script"
```

---

## Task 5: Update __init__.py to export new classes

**Files:**
- Modify: `src/training/__init__.py`

- [ ] **Step 1: Add exports for new classes**

```python
# src/training/__init__.py
"""
Training module for fine-tuning SAT item generation models.

Provides dataset classes and model loading utilities for both
PyTorch (production) and MLX (local) backends.
"""

# Existing PyTorch exports
from src.training.dataset import SFTDataset, create_dataloader
from src.training.models import (
    load_tokenizer,
    load_model,
    apply_lora,
    load_model_for_training,
    print_model_summary,
    save_model
)

# New MLX exports
from src.training.converters import convert_item_to_messages
from src.training.mlx_dataset import MLXDataset
from src.training.mlx_trainer import MLXTrainer

__all__ = [
    # PyTorch
    "SFTDataset",
    "create_dataloader",
    "load_tokenizer",
    "load_model",
    "apply_lora",
    "load_model_for_training",
    "print_model_summary",
    "save_model",
    # MLX
    "convert_item_to_messages",
    "MLXDataset",
    "MLXTrainer"
]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from src.training import MLXDataset, MLXTrainer, convert_item_to_messages; print('✓ Imports work')"`
Expected: ✓ Imports work

- [ ] **Step 3: Commit**

```bash
git add src/training/__init__.py
git commit -m "feat: export MLX classes from training module"
```

---

## Task 6: Integration smoke test

**Files:**
- No file creation/modification (execution only)

- [ ] **Step 1: Create mock data**

Run: `source .venv/bin/activate && python scripts/create_mock_data.py --section math --count 5`
Expected: ✓ Created mock dataset with train/val/test splits

- [ ] **Step 2: Run one epoch to validate pipeline**

Run: `source .venv/bin/activate && python scripts/train_local.py --section math --num_epochs 1`
Expected: Training runs, model loads, epoch completes, checkpoint saves

**Note:** The loss computation implementation may need adjustment based on the actual MLX model structure. If training fails at the loss step, inspect the model's forward pass signature and update `_compute_loss()` in `MLXTrainer` accordingly.

- [ ] **Step 3: Verify checkpoint was created**

Run: `ls -la outputs/checkpoints/math/`
Expected: Directory exists with epoch-1 checkpoint

- [ ] **Step 4: Run all tests to verify nothing broke**

Run: `pytest tests/ -v`
Expected: All tests pass

---

## Success Criteria

✅ All unit tests pass
✅ Mock data generates successfully
✅ MLX model loads without errors
✅ Dataset loads and tokenizes items
✅ Training loop completes at least one epoch
✅ Loss decreases or stays stable (not NaN/inf)
✅ Checkpoint files are created and non-zero
✅ CLI script accepts arguments and runs training end-to-end

---

## Next Steps After Implementation

1. **Test with real MLX model** — Run actual training to verify loss computation works
2. **Add LoRA support** (optional) — If full fine-tuning is too slow on 8GB
3. **Add progress monitoring** — tqdm or similar for longer runs
4. **Document in CLAUDE.md** — Update training section with local MLX instructions
