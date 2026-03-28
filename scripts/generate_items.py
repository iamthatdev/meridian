#!/usr/bin/env python3
"""
Generate SAT items using fine-tuned models.

Usage:
    # Generate single item
    python scripts/generate_items.py --section math --domain algebra.quadratic_equations --difficulty medium

    # Generate batch
    python scripts/generate_items.py --section reading_writing --batch --items-per-domain 2

    # Save to file
    python scripts/generate_items.py --section math --domain algebra.linear_equations --difficulty easy --output generated_items.jsonl
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

from src.config import load_config
from src.generation.generator import ItemGenerator
from src.auto_qa.pipeline import AutoQAPipeline


def generate_items(
    checkpoint: str,
    section: str,
    domains: list,
    difficulty: str,
    items_per_domain: int = 1,
    validate: bool = True,
    output: str = None,
    batch: bool = False
):
    """
    Generate SAT items.

    Args:
        checkpoint: Path to model checkpoint
        section: SAT section (reading_writing, math)
        domains: List of domains to generate items for
        difficulty: Difficulty tier
        items_per_domain: Number of items per domain
        validate: Whether to run Auto-QA validation
        output: Output file path
        batch: Generate batch across domains
    """
    # Initialize generator
    try:
        generator = ItemGenerator(checkpoint_path=checkpoint)
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        sys.exit(1)

    # Initialize Auto-QA pipeline if validation enabled
    pipeline = None
    if validate:
        pipeline = AutoQAPipeline()

    # Generate items
    if batch:
        # Batch generation across domains
        all_items = generator.generate_batch(
            section=section,
            domains=domains,
            difficulty=difficulty,
            items_per_domain=items_per_domain
        )
    else:
        # Single domain generation
        all_items = []
        for domain in domains:
            items = generator.generate(
                section=section,
                domain=domain,
                difficulty=difficulty,
                num_return_sequences=items_per_domain
            )
            all_items.extend(items)

    logger.info(f"Generated {len(all_items)} total items")

    # Validate items
    if validate and pipeline:
        logger.info("Running Auto-QA validation...")

        validated_items = []
        for item in all_items:
            result = pipeline.validate(item)

            if result.get("auto_qa_passed", False):
                validated_items.append(item)
            else:
                logger.warning(
                    f"Item {item['id']} failed validation: "
                    f"{result.get('qa_flags', [])}"
                )

        logger.info(f"Validation passed: {len(validated_items)}/{len(all_items)}")

        all_items = validated_items

    # Save to file
    if output and all_items:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".jsonl":
            with open(output_path, "w") as f:
                for item in all_items:
                    f.write(json.dumps(item) + "\n")
        elif output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(all_items, f, indent=2)
        else:
            logger.error(f"Unsupported output format: {output_path.suffix}")
            logger.info("Supported formats: .json, .jsonl")
            sys.exit(1)

        logger.info(f"Saved {len(all_items)} items to {output_path}")

    # Print summary
    logger.info("=" * 50)
    logger.info("Generation Summary")
    logger.info("=" * 50)
    logger.info(f"Total generated: {len(all_items)}")
    logger.info(f"Section: {section}")
    logger.info(f"Difficulty: {difficulty}")

    # Count by domain
    domain_counts = {}
    for item in all_items:
        domain = item.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    logger.info("Items by domain:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain}: {count}")

    logger.info("=" * 50)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SAT items using fine-tuned models"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--section",
        required=True,
        choices=["reading_writing", "math", "rw", "readingwriting"],
        help="SAT section"
    )
    parser.add_argument(
        "--domain",
        help="Single domain to generate items for (for single item generation)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="List of domains to generate items for"
    )
    parser.add_argument(
        "--difficulty",
        required=True,
        choices=["easy", "medium", "hard"],
        help="Difficulty tier"
    )
    parser.add_argument(
        "--items-per-domain",
        type=int,
        default=1,
        help="Number of items to generate per domain (default: 1)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch generation mode"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip Auto-QA validation"
    )
    parser.add_argument(
        "--output",
        help="Output file path (JSON or JSONL)"
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

    # Determine domains
    if args.domain:
        domains = [args.domain]
    elif args.domains:
        domains = args.domains
    else:
        logger.error("Must specify --domain or --domains")
        sys.exit(1)

    # Generate
    generate_items(
        checkpoint=args.checkpoint,
        section=section,
        domains=domains,
        difficulty=args.difficulty,
        items_per_domain=args.items_per_domain,
        validate=not args.no_validate,
        output=args.output,
        batch=args.batch
    )


if __name__ == "__main__":
    main()
