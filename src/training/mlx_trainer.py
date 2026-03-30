"""
MLX Trainer for local Apple Silicon training.

Minimal viable training loop using MLX APIs.
"""

from pathlib import Path
from loguru import logger

try:
    import mlx.core as mx
    import mlx.optim as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    optim = None

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
        self.optimizer = optim.AdamW(learning_rate=config.learning_rate)

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
