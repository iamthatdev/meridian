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
