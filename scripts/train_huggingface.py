#!/usr/bin/env python3
"""
HuggingFace-based training script for SAT item generation models.

Uses standard HuggingFace Transformers library with Trainer API.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)


def load_jsonl_data(data_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_messages_for_training(messages: List[Dict], tokenizer) -> str:
    """Format messages using the model's chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def create_dataset_from_jsonl(data_path: str, tokenizer, max_length: int) -> Dataset:
    """Create HuggingFace Dataset from JSONL file."""
    logger.info(f"Loading data from {data_path}")

    # Load JSONL data
    raw_data = load_jsonl_data(data_path)
    logger.info(f"Loaded {len(raw_data)} examples")

    # Format messages using chat template
    formatted_texts = []
    for item in raw_data:
        if "messages" in item:
            formatted = format_messages_for_training(item["messages"], tokenizer)
            formatted_texts.append(formatted)
        else:
            logger.warning(f"Skipping item without 'messages' field: {item}")

    # Tokenize
    logger.info("Tokenizing data...")
    tokenized = tokenizer(
        formatted_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )

    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].copy()  # For causal LM, labels = input_ids
    })

    logger.info(f"Created dataset with {len(dataset)} examples")
    return dataset


def setup_model_and_tokenizer(section: str, config: Dict):
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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

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

    # Load datasets
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

    # Find validation file
    val_file = validated_dir / f"{section}_val.jsonl"
    if not val_file.exists():
        val_file = data_dir / "splits" / f"{section}_val.jsonl"

    # Create datasets
    train_dataset = create_dataset_from_jsonl(str(train_file), tokenizer, max_length)

    val_dataset = None
    if val_file.exists():
        val_dataset = create_dataset_from_jsonl(str(val_file), tokenizer, max_length)
    else:
        logger.warning(f"Validation file not found: {val_file}")

    # Setup training arguments
    training_args = TrainingArguments(
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
        evaluation_strategy="steps" if val_dataset else "no",
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
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
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
