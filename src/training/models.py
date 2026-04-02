"""
Model loading utilities with LoRA/QLoRA support.

Provides functions to load base models and apply LoRA/QLoRA adapters
for efficient fine-tuning.
"""

from typing import Optional, Dict, Any
from loguru import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    logger.warning("transformers or peft not available. Model loading will not work.")

from src.config import load_config, Config


# Model-specific LoRA target modules
MODEL_LORA_TARGETS = {
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}


def load_tokenizer(model_id: str, config: Config = None) -> Optional[Any]:
    """
    Load tokenizer for the specified model.

    Args:
        model_id: HuggingFace model ID
        config: Optional Config object

    Returns:
        Tokenizer instance or None if transformers not available
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers library not available")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = "<|endoftext|>"

        logger.info(f"Loaded tokenizer for {model_id}")
        logger.info(f"  Vocab size: {len(tokenizer)}")
        logger.info(f"  Pad token: {tokenizer.pad_token}")
        logger.info(f"  EOS token: {tokenizer.eos_token}")

        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_id}: {e}")
        return None


def load_model(
    model_id: str,
    config: Config = None,
    use_4bit: bool = True,
    device_map: str = "auto"
) -> Optional[Any]:
    """
    Load base model with optional 4-bit quantization.

    Args:
        model_id: HuggingFace model ID
        config: Config object with LoRA settings
        use_4bit: Whether to use 4-bit quantization (QLoRA)
        device_map: Device mapping strategy

    Returns:
        Model instance or None if transformers not available
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers library not available")
        return None

    if config is None:
        config = load_config()

    try:
        # Configure quantization
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map
        }

        if use_4bit:
            logger.info("Using 4-bit quantization (QLoRA)")

            # Get compute dtype from config
            dtype_str = config.quantization.bnb_4bit_compute_dtype
            if dtype_str == "bfloat16":
                compute_dtype = torch.bfloat16
            elif dtype_str == "float16":
                compute_dtype = torch.float16
            else:
                logger.warning(f"Unknown dtype {dtype_str}, defaulting to bfloat16")
                compute_dtype = torch.bfloat16

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type
            )
            logger.info(f"  Compute dtype: {dtype_str}")
        else:
            logger.info("Using full precision (fp16)")
            model_kwargs["torch_dtype"] = torch.float16

        # Load model
        logger.info(f"Loading model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        )

        logger.info(f"Loaded model {model_id}")
        logger.info(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")

        # Prepare for k-bit training if using quantization
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")

        return model

    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        return None


def apply_lora(
    model,
    config: Config = None,
    target_modules: list = None
) -> Optional[Any]:
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base model instance
        config: Config object with LoRA settings
        target_modules: Optional list of target module names

    Returns:
        Model with LoRA adapters or None if peft not available
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("peft library not available")
        return None

    if config is None:
        config = load_config()

    try:
        # Determine target modules
        if target_modules is None:
            # Detect model architecture from model name or config
            model_name = getattr(model, 'name_or_path', None)

            # Try to detect from config
            if hasattr(model.config, 'model_type'):
                model_type = model.config.model_type
                target_modules = MODEL_LORA_TARGETS.get(
                    model_type,
                    MODEL_LORA_TARGETS["default"]
                )
            else:
                target_modules = MODEL_LORA_TARGETS["default"]

        logger.info(f"Applying LoRA with target modules: {target_modules}")

        # Create LoRA config
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = model.num_parameters()

        logger.info("LoRA applied successfully")
        logger.info(f"  Rank (r): {config.lora.r}")
        logger.info(f"  Alpha: {config.lora.alpha}")
        logger.info(f"  Dropout: {config.lora.dropout}")
        try:
            logger.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
            logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        except (TypeError, ValueError):
            logger.info(f"  Trainable parameters: {trainable_params}")
            logger.info(f"  Total parameters: {total_params}")

        return model

    except Exception as e:
        logger.error(f"Failed to apply LoRA: {e}")
        return None


def load_model_for_training(
    section: str,
    config: Config = None,
    use_4bit: bool = True
) -> Optional[tuple]:
    """
    Load model and tokenizer for training.

    This is a convenience function that loads both the model and tokenizer
    and applies LoRA adapters.

    Args:
        section: Model section ('reading_writing' or 'math')
        config: Config object
        use_4bit: Whether to use 4-bit quantization

    Returns:
        Tuple of (model, tokenizer) or (None, None) on error
    """
    if config is None:
        config = load_config()

    # Select model based on section
    if section == "reading_writing" or section == "rw":
        model_id = config.models.rw_model_id
    elif section == "math":
        model_id = config.models.math_model_id
    else:
        logger.error(f"Unknown section: {section}")
        return None, None

    logger.info(f"Loading {section} model: {model_id}")

    # Load tokenizer
    tokenizer = load_tokenizer(model_id, config)
    if tokenizer is None:
        return None, None

    # Load model
    model = load_model(model_id, config, use_4bit=use_4bit)
    if model is None:
        return None, None

    # Apply LoRA
    model = apply_lora(model, config)
    if model is None:
        return None, None

    # Enable gradient checkpointing if configured
    if config.quantization.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled (reduces memory usage)")

    logger.info(f"✅ Model and tokenizer loaded successfully for {section}")

    return model, tokenizer


def print_model_summary(model):
    """
    Print a summary of the model architecture.

    Args:
        model: Model instance
    """
    if not TRANSFORMERS_AVAILABLE:
        return

    logger.info("=" * 60)
    logger.info("Model Summary")
    logger.info("=" * 60)

    total_params = model.num_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    try:
        logger.info(f"Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        logger.info(f"Trainable %: {trainable_params / total_params * 100:.2f}%")
    except (TypeError, ValueError):
        # In tests, parameters might be MagicMock objects
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")

    if hasattr(model, "hf_device_map"):
        logger.info(f"Device map: {model.hf_device_map}")

    logger.info("=" * 60)


def save_model(model, tokenizer, output_dir: str, optimizer=None, scheduler=None, training_state: dict = None):
    """
    Save model, tokenizer, and training state.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        output_dir: Directory to save to
        optimizer: Optional optimizer state to save
        scheduler: Optional scheduler state to save
        training_state: Optional training state dict (epoch, global_step, etc.)
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers library not available")
        return

    from pathlib import Path
    import torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save LoRA adapters
        model.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")

        # Save tokenizer
        tokenizer.save_pretrained(output_path)
        logger.info(f"Tokenizer saved to {output_path}")

        # Save training state for resumption
        if training_state:
            state_file = output_path / "training_state.pt"
            torch.save(training_state, state_file)
            logger.info(f"Training state saved to {state_file}")

        # Save optimizer state
        if optimizer:
            opt_file = output_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), opt_file)
            logger.info(f"Optimizer state saved to {opt_file}")

        # Save scheduler state
        if scheduler:
            sched_file = output_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), sched_file)
            logger.info(f"Scheduler state saved to {sched_file}")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def load_checkpoint(model, tokenizer, checkpoint_dir: str, optimizer=None, scheduler=None):
    """
    Load model checkpoint and optionally optimizer/scheduler state.

    Args:
        model: Model instance to load weights into
        tokenizer: Tokenizer instance
        checkpoint_dir: Directory containing checkpoint
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Tuple of (training_state_dict, model, tokenizer)
        training_state_dict contains: epoch, global_step, best_val_loss, etc.
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers library not available")
        return None, model, tokenizer

    from pathlib import Path
    import torch
    from peft import PeftModel

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None, model, tokenizer

    try:
        # Load LoRA adapters
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        logger.info("✓ Checkpoint loaded successfully")

        # Load training state if available
        training_state = {}
        state_file = checkpoint_path / "training_state.pt"
        if state_file.exists():
            training_state = torch.load(state_file)
            logger.info(f"✓ Training state loaded: epoch {training_state.get('epoch', 0)}, step {training_state.get('global_step', 0)}")

        # Load optimizer state if available and optimizer provided
        if optimizer and (checkpoint_path / "optimizer.pt").exists():
            opt_state = torch.load(checkpoint_path / "optimizer.pt")
            optimizer.load_state_dict(opt_state)
            logger.info("✓ Optimizer state loaded")

        # Load scheduler state if available and scheduler provided
        if scheduler and (checkpoint_path / "scheduler.pt").exists():
            sched_state = torch.load(checkpoint_path / "scheduler.pt")
            scheduler.load_state_dict(sched_state)
            logger.info("✓ Scheduler state loaded")

        return training_state, model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None, model, tokenizer

