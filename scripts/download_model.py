#!/usr/bin/env python3
"""
Download a model before training to avoid download-time errors.

This script downloads the model and tokenizer to the HuggingFace cache
so that training can proceed without download issues.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    logger.error("Required packages not installed. Run: pip install torch transformers")
    sys.exit(1)


def download_model(model_name: str, output_dir: str = None):
    """Download model and tokenizer to cache or specified directory."""
    logger.info(f"Downloading model: {model_name}")

    try:
        # Download tokenizer first (smaller)
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(output_path)
            logger.info(f"Tokenizer saved to {output_path}")

        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Don't load to GPU yet
            trust_remote_code=True,
        )

        if output_dir:
            model.save_pretrained(output_path)
            logger.info(f"Model saved to {output_path}")
        else:
            logger.info("Model cached in HuggingFace hub cache")

        logger.info(f"✅ Download complete: {model_name}")
        logger.info(f"Model size: {model.num_parameters() / 1e9:.2f}B parameters")
        return 0

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Download model before training"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["phi-4", "qwen2.5-7b", "all"],
        default="all",
        help="Model to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to HF cache)"
    )

    args = parser.parse_args()

    models = {
        "phi-4": "microsoft/phi-4",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    }

    if args.model == "all":
        logger.info("Downloading all models...")
        results = []
        for name, model_id in models.items():
            result = download_model(model_id, args.output_dir)
            results.append((name, result))

        # Report summary
        logger.info("=" * 70)
        logger.info("Download Summary")
        logger.info("=" * 70)
        for name, result in results:
            status = "✅ SUCCESS" if result == 0 else "❌ FAILED"
            logger.info(f"{status}: {name}")

        return 0 if all(r == 0 for _, r in results) else 1
    else:
        return download_model(models[args.model], args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
