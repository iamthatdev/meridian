#!/usr/bin/env python3
"""
HuggingFace-based training script for SAT item generation models.

Uses standard HuggingFace Transformers library with Trainer API.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


def setup_model_and_tokenizer(section: str, config: Dict, max_retries: int = 3):
    """Setup model, tokenizer, and LoRA configuration."""

    # Select model
    if section == "math":
        model_name = config.get("math_model_id", "microsoft/phi-4")
        max_length = config.get("max_seq_length_math", 2048)
    else:  # reading_writing
        model_name = config.get("rw_model_id", "Qwen/Qwen2.5-7B-Instruct")
        max_length = config.get("max_seq_length_rw", 4096)

    logger.info(f"Loading model: {model_name}")

    # Setup quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with retry logic
    model = None
    tokenizer = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}...")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=False,  # Allow download
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False,
            )

            logger.info("✅ Model and tokenizer loaded successfully")
            break

        except Exception as e:
            logger.error(f"❌ Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

                # Clean up potentially corrupted download
                logger.info("Cleaning up cache and retrying...")
                import shutil
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                if cache_dir.exists():
                    # Only remove the specific model cache
                    model_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
                    if model_cache.exists():
                        shutil.rmtree(model_cache)
                        logger.info(f"Removed corrupted cache: {model_cache}")
            else:
                logger.error("Failed to load model after all retries")
                logger.error("Suggestions:")
                logger.error("1. Check disk space: df -h")
                logger.error("2. Try downloading model first: python scripts/download_model.py --model phi-4")
                logger.error("3. Check network connection")
                raise

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA config
    lora_config = LoraConfig(
        r=config.get("lora_r", 32),
        lora_alpha=config.get("lora_alpha", 64),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info(f"Model loaded with max_length={max_length}")

    return model, tokenizer, max_length


def train(
    section: str,
    env: str = "production",
    checkpoint_dir: str = None,
    resume_from: str = None
):
    """Train a model using HuggingFace Trainer."""

    # Normalize section name
    if section in ["rw", "readingwriting"]:
        section = "reading_writing"

    logger.info("=" * 70)
    logger.info(f"Training {section} model")
    logger.info("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent.parent
    config_file = script_dir / "configs" / f"{env}.yaml"

    # Load config
    import yaml
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Extract relevant configs
    models_config = config_dict.get("models", {})
    lora_config = config_dict.get("lora", {})
    training_config = config_dict.get("training", {})
    paths_config = config_dict.get("paths", {})

    # Merge configs
    model_config = {
        "math_model_id": models_config.get("math_model_id", "microsoft/phi-4"),
        "rw_model_id": models_config.get("rw_model_id", "Qwen/Qwen2.5-7B-Instruct"),
        "max_seq_length_math": training_config.get("max_seq_length_math", 2048),
        "max_seq_length_rw": training_config.get("max_seq_length_rw", 4096),
        "lora_r": lora_config.get("r", 32),
        "lora_alpha": lora_config.get("alpha", 64),
        "lora_dropout": lora_config.get("dropout", 0.05),
    }

    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path(paths_config.get("checkpoint_dir", "checkpoints")) / section
    else:
        checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = checkpoint_dir / timestamp
    run_dir.mkdir(exist_ok=True)

    logger.info(f"Run directory: {run_dir}")

    # Setup logging
    log_file = Path(paths_config.get("log_dir", "logs")) / f"training_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="500 MB")
    logger.info(f"Logging to: {log_file}")

    # Setup model and tokenizer
    model, tokenizer, max_length = setup_model_and_tokenizer(section, model_config)

    # Verify tokenizer has chat template support
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError(
            f"Tokenizer for {model_config.get('rw_model_id' if section == 'reading_writing' else 'math_model_id')} "
            f"doesn't support chat templates. "
            f"SFTTrainer requires a tokenizer with chat_template attribute."
        )

    # Load datasets using load_dataset
    data_dir = Path(paths_config.get("data_dir", "data"))
    training_dir = Path(paths_config.get("training_dir", "data/training"))
    validated_dir = Path(paths_config.get("validated_dir", "data/validated"))

    # Find training file
    train_file = training_dir / f"{section}_train.jsonl"
    if not train_file.exists():
        train_file = data_dir / "splits" / f"{section}_train.jsonl"

    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        sys.exit(1)

    logger.info(f"Loading training data from: {train_file}")

    # Load training dataset
    train_dataset = load_dataset(
        "json",
        data_files=str(train_file),
        split="train"
    )
    logger.info(f"Loaded {len(train_dataset)} training examples")

    # Verify data format
    if "messages" not in train_dataset.column_names:
        raise ValueError(
            f"Training data must have 'messages' field. "
            f"Found columns: {train_dataset.column_names}"
        )

    # Find and load validation file
    val_file = validated_dir / f"{section}_val.jsonl"
    if not val_file.exists():
        val_file = data_dir / "splits" / f"{section}_val.jsonl"

    val_dataset = None
    if val_file.exists():
        logger.info(f"Loading validation data from: {val_file}")
        val_dataset = load_dataset(
            "json",
            data_files=str(val_file),
            split="train"
        )
        logger.info(f"Loaded {len(val_dataset)} validation examples")

    # Setup training arguments using SFTConfig
    sft_config = SFTConfig(
        output_dir=str(run_dir),
        learning_rate=training_config.get("learning_rate", 2e-5),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        per_device_eval_batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        num_train_epochs=training_config.get("num_epochs", 3),
        warmup_ratio=training_config.get("warmup_ratio", 0.05),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=500 if val_dataset else None,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_dir=str(run_dir / "logs"),
        report_to=["tensorboard"],
        save_strategy="steps",
        load_best_model_at_end=False,
        run_name=f"{section}_{timestamp}",
        dataset_text_field="messages",  # Use the 'messages' field from JSONL
        max_length=max_length,
    )

    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Resume from checkpoint if specified
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        train_result = trainer.train(resume_from_checkpoint=resume_from)
    else:
        logger.info("Starting training...")
        train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(str(run_dir / "final_model"))
    tokenizer.save_pretrained(str(run_dir / "final_model"))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("=" * 70)
    logger.info("Training completed!")
    logger.info(f"Model saved to: {run_dir / 'final_model'}")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train SAT item generation models using HuggingFace Trainer"
    )
    parser.add_argument(
        "--section",
        type=str,
        required=True,
        choices=["reading_writing", "math", "rw", "readingwriting"],
        help="Model section to train"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="production",
        choices=["local", "production"],
        help="Environment (local or production)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: checkpoints/<section>/)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    try:
        train(
            section=args.section,
            env=args.env,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
