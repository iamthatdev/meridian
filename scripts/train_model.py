#!/usr/bin/env python3
"""
Training script for fine-tuning SAT item generation models.

Usage:
    # Train Reading & Writing model
    python scripts/train_model.py --section reading_writing

    # Train Math model
    python scripts/train_model.py --section math

    # Resume from checkpoint
    python scripts/train_model.py --section reading_writing --checkpoint checkpoints/reading_writing/latest
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("PyTorch/transformers not available. Training will not work.")

from src.config import load_config
from src.training.dataset import SFTDataset, create_dataloader
from src.training.models import (
    load_model_for_training,
    save_model,
    load_checkpoint,
    print_model_summary
)


def train(
    section: str,
    config = None,
    checkpoint_dir: str = None,
    resume_from: str = None
):
    """
    Train a model on the specified section.

    Args:
        section: Model section ('reading_writing' or 'math')
        config: Config object
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Cannot train: PyTorch/transformers not available")
        sys.exit(1)

    if config is None:
        config = load_config()

    # Normalize section name
    if section in ["rw", "readingwriting"]:
        section = "reading_writing"

    logger.info("=" * 70)
    logger.info(f"Training {section} model")
    logger.info("=" * 70)

    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path(config.paths.checkpoint_dir) / section
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = checkpoint_dir / timestamp
    run_dir.mkdir(exist_ok=True)

    logger.info(f"Run directory: {run_dir}")

    def check_disk_space(required_mb: int = 500) -> bool:
        """Check if sufficient disk space is available."""
        try:
            stat = os.statvfs(run_dir)
            available_mb = stat.f_bavail * stat.f_frsize // (1024 * 1024)

            if available_mb < required_mb:
                logger.error(f"Insufficient disk space: {available_mb}MB available, {required_mb}MB required")
                raise RuntimeError(f"Insufficient disk space: {available_mb}MB available, {required_mb}MB required")

            return True
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            raise

    def verify_checkpoint(checkpoint_path: Path) -> bool:
        """Verify checkpoint was saved correctly."""
        try:
            # Check required files exist
            required_files = [
                checkpoint_path / "adapter_model.safetensors",
                checkpoint_path / "adapter_config.json",
                checkpoint_path / "tokenizer_config.json"
            ]

            for file in required_files:
                if not file.exists():
                    logger.error(f"Checkpoint file missing: {file}")
                    return False

            # Check model file size
            model_file = checkpoint_path / "adapter_model.safetensors"
            model_size_mb = model_file.stat().st_size / (1024 * 1024)

            if model_size_mb < 100:  # Less than 100MB is suspicious
                logger.error(f"Model file too small: {model_size_mb:.2f}MB")
                return False

            logger.info(f"✓ Checkpoint verified: {checkpoint_path.name} ({model_size_mb:.2f}MB)")
            return True

        except Exception as e:
            logger.error(f"Checkpoint verification failed: {e}")
            return False

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_for_training(
        section=section,
        config=config,
        use_4bit=True
    )

    if model is None or tokenizer is None:
        logger.error("Failed to load model or tokenizer")
        sys.exit(1)

    print_model_summary(model)

    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        training_state, model, tokenizer = load_checkpoint(
            model, tokenizer, resume_from
        )
        if training_state:
            start_epoch = training_state.get('epoch', 0)
            global_step = training_state.get('global_step', 0)
            best_val_loss = training_state.get('best_val_loss', float("inf"))
            logger.info(f"✓ Resumed from epoch {start_epoch}, step {global_step}")
        else:
            logger.warning("Failed to load training state, starting from scratch")

    # Load datasets
    logger.info("Loading datasets...")

    train_file = Path(config.paths.data_dir) / "splits" / f"{section.replace('_', '')}_train.jsonl"
    val_file = Path(config.paths.data_dir) / "splits" / f"{section.replace('_', '')}_val.jsonl"

    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.info("Expected file structure:")
        logger.info(f"  {train_file}")
        sys.exit(1)

    if not val_file.exists():
        logger.warning(f"Validation file not found: {val_file}")
        logger.info("Continuing without validation set...")
        val_file = None

    # Create datasets
    max_seq_length = config.training.max_seq_length_rw if section == "reading_writing" else config.training.max_seq_length_math

    train_dataset = SFTDataset(
        data_path=str(train_file),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        section=section
    )

    logger.info(f"Training examples: {len(train_dataset)}")

    if val_file:
        val_dataset = SFTDataset(
            data_path=str(val_file),
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            section=section
        )
        logger.info(f"Validation examples: {len(val_dataset)}")
    else:
        val_dataset = None

    # Create dataloaders
    train_dataloader = create_dataloader(
        data_path=str(train_file),
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_seq_length=max_seq_length,
        section=section,
        shuffle=True
    )

    if val_dataset:
        val_dataloader = create_dataloader(
            data_path=str(val_file),
            tokenizer=tokenizer,
            batch_size=config.training.batch_size,
            max_seq_length=max_seq_length,
            section=section,
            shuffle=False
        )
    else:
        val_dataloader = None

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=0.01
    )

    # Setup learning rate scheduler
    num_training_steps = len(train_dataloader) * config.training.num_epochs
    num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"Training steps per epoch: {len(train_dataloader)}")
    logger.info(f"Total epochs: {config.training.num_epochs}")
    logger.info(f"Total training steps: {num_training_steps}")

    # Load optimizer/scheduler state if resuming
    if resume_from:
        checkpoint_path = Path(resume_from)
        if (checkpoint_path / "optimizer.pt").exists():
            opt_state = torch.load(checkpoint_path / "optimizer.pt")
            optimizer.load_state_dict(opt_state)
            logger.info("✓ Optimizer state loaded")
        if (checkpoint_path / "scheduler.pt").exists():
            sched_state = torch.load(checkpoint_path / "scheduler.pt")
            scheduler.load_state_dict(sched_state)
            logger.info("✓ Scheduler state loaded")

    # Training loop
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    # Setup gradient accumulation
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    logger.info(f"Using gradient accumulation: {gradient_accumulation_steps} steps")

    for epoch in range(start_epoch, config.training.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")

        try:
            # Training
            model.train()
            total_train_loss = 0
            accumulated_loss = 0

            for step, batch in enumerate(train_dataloader):
                try:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(model.device)
                    attention_mask = batch["attention_mask"].to(model.device)
                    labels = batch["labels"].to(model.device)

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    # Scale loss for gradient accumulation
                    loss = outputs.loss / gradient_accumulation_steps

                    # Check for loss explosion (training divergence)
                    if not torch.isfinite(loss.item()):
                        logger.error(f"Loss exploded at epoch {epoch}, step {step}: {loss.item()}")
                        logger.error("This typically indicates training divergence. Stopping.")
                        raise RuntimeError("Training divergence - loss is NaN or Inf")

                    total_train_loss += loss.item() * gradient_accumulation_steps
                    accumulated_loss += loss.item() * gradient_accumulation_steps

                    # Backward pass (accumulate gradients)
                    try:
                        loss.backward()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.error(f"CUDA OOM at epoch {epoch}, step {step}")
                            raise
                        else:
                            raise

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA OOM at epoch {epoch}, step {step}: {e}")
                        logger.info("Attempting to save emergency checkpoint...")

                        # Try to save current state before dying
                        try:
                            checkpoint_path = run_dir / f"OOM_CHECKPOINT_epoch_{epoch}_step_{step}"
                            training_state = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "best_val_loss": best_val_loss,
                                "oom_error": str(e)
                            }
                            save_model(model, tokenizer, str(checkpoint_path), None, None, training_state)
                            logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
                        except Exception as save_error:
                            logger.error(f"Failed to save emergency checkpoint: {save_error}")

                        raise  # Re-raise to stop training
                    else:
                        logger.error(f"Runtime error at epoch {epoch}, step {step}: {e}")
                        raise

                # Only step optimizer after accumulating enough gradients
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log progress
                    if global_step % 10 == 0:
                        logger.info(
                            f"Epoch {epoch + 1} | Step {global_step} | "
                            f"Loss: {accumulated_loss / gradient_accumulation_steps:.4f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}"
                        )
                    accumulated_loss = 0

                    # Save checkpoint periodically
                    if global_step % 100 == 0:
                        try:
                            checkpoint_path = run_dir / f"checkpoint-step-{global_step}"
                            training_state = {
                                "epoch": epoch,
                                "global_step": global_step,
                                "best_val_loss": best_val_loss
                            }
                            check_disk_space(required_mb=500)  # Check before saving
                            save_model(model, tokenizer, str(checkpoint_path), optimizer, scheduler, training_state)
                            logger.info(f"Saved checkpoint: {checkpoint_path}")
                            verify_checkpoint(checkpoint_path)  # Verify save was successful
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint at step {global_step}: {e}")
                            # Don't raise - training can continue

            avg_train_loss = total_train_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            if val_dataloader:
                logger.info("Running validation...")
                model.eval()
                total_val_loss = 0

                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch["input_ids"].to(model.device)
                        attention_mask = batch["attention_mask"].to(model.device)
                        labels = batch["labels"].to(model.device)

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                        total_val_loss += outputs.loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                logger.info(f"Average validation loss: {avg_val_loss:.4f}")

                # Save best checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_checkpoint = run_dir / "best"
                    training_state = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val_loss": best_val_loss
                    }
                    check_disk_space(required_mb=500)
                    save_model(model, tokenizer, str(best_checkpoint), optimizer, scheduler, training_state)
                    logger.info(f"Saved best checkpoint (val_loss: {avg_val_loss:.4f})")
                    verify_checkpoint(best_checkpoint)

            # Save epoch checkpoint
            epoch_checkpoint = run_dir / f"epoch-{epoch + 1}"
            training_state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss
            }
            check_disk_space(required_mb=500)
            save_model(model, tokenizer, str(epoch_checkpoint), optimizer, scheduler, training_state)
            logger.info(f"Saved epoch checkpoint: {epoch_checkpoint}")
            verify_checkpoint(epoch_checkpoint)

        except Exception as e:
            logger.error(f"Exception in training loop at epoch {epoch}: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Try to save checkpoint before dying
            try:
                checkpoint_path = run_dir / f"ERROR_CHECKPOINT_epoch_{epoch}"
                training_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                save_model(model, tokenizer, str(checkpoint_path), None, None, training_state)
                logger.info(f"Error checkpoint saved: {checkpoint_path}")
            except Exception as save_error:
                logger.error(f"Failed to save error checkpoint: {save_error}")

            raise  # Re-raise to stop training

    # Save final checkpoint
    final_checkpoint = run_dir / "final"
    training_state = {
        "epoch": config.training.num_epochs,
        "global_step": global_step,
        "best_val_loss": best_val_loss
    }
    check_disk_space(required_mb=500)
    save_model(model, tokenizer, str(final_checkpoint), optimizer, scheduler, training_state)
    logger.info(f"Saved final checkpoint: {final_checkpoint}")
    verify_checkpoint(final_checkpoint)

    # Save training metadata
    metadata = {
        "section": section,
        "timestamp": timestamp,
        "epochs_completed": epoch + 1 if epoch >= start_epoch else config.training.num_epochs,
        "global_step": global_step,
        "final_train_loss": avg_train_loss,
        "best_val_loss": best_val_loss if val_dataloader else None,
        "resumed_from": resume_from,
        "config": {
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "num_epochs": config.training.num_epochs,
            "max_seq_length": max_seq_length,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.alpha
        }
    }

    import json
    with open(run_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train SAT item generation models"
    )
    parser.add_argument(
        "--section",
        required=True,
        choices=["reading_writing", "math", "rw", "readingwriting"],
        help="Model section to train"
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory to save checkpoints (default: checkpoints/<section>/)"
    )
    parser.add_argument(
        "--resume-from",
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--env",
        default=os.getenv("APP_ENV", "local"),
        help="Environment (local or production)"
    )

    args = parser.parse_args()

    # Set environment
    os.environ["APP_ENV"] = args.env

    # Normalize section name
    section = args.section
    if section in ["rw", "readingwriting"]:
        section = "reading_writing"

    # Train
    train(
        section=section,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
