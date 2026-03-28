#!/usr/bin/env python3
"""
Load validated items from JSON/JSONL files into the Item Bank.

Usage:
    python scripts/load_items.py --input data/validated/math_items.jsonl --section math
"""

import argparse
import json
import sys
from pathlib import Path
from loguru import logger

from src.config import load_config
from src.item_bank.database import DatabaseManager
from src.item_bank.repositories.item_repository import ItemRepository
from src.auto_qa.pipeline import AutoQAPipeline


def load_items(input_path: str, section: str = None):
    """
    Load items from a JSON/JSONL file into the Item Bank.

    Args:
        input_path: Path to input file (JSON or JSONL)
        section: Optional section filter (only load items from this section)
    """
    # Initialize database
    config = load_config()
    db = DatabaseManager()

    try:
        db.initialize(config)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    # Initialize pipeline for validation
    pipeline = AutoQAPipeline()

    # Read input file
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Detect format and load items
    logger.info(f"Loading items from {input_path}")

    items = []
    if input_file.suffix == ".jsonl":
        with open(input_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    elif input_file.suffix == ".json":
        with open(input_file) as f:
            data = json.load(f)
            # Handle both list and single item
            if isinstance(data, list):
                items = data
            else:
                items = [data]
    else:
        logger.error(f"Unsupported file format: {input_file.suffix}")
        sys.exit(1)

    logger.info(f"Loaded {len(items)} items from file")

    # Filter by section if specified
    if section:
        items = [item for item in items if item.get("section") == section]
        logger.info(f"Filtered to {len(items)} items with section='{section}'")

    # Load items into database
    stats = {
        "total": len(items),
        "loaded": 0,
        "skipped": 0,
        "failed": 0
    }

    with db.get_connection() as conn:
        repo = ItemRepository()

        for idx, item in enumerate(items, 1):
            item_id = item.get("id", f"unknown-{idx}")

            try:
                # Validate with Auto-QA
                qa_result = pipeline.validate(item)

                if not qa_result.get("auto_qa_passed", False):
                    logger.warning(
                        f"Skipping item {item_id}: Auto-QA failed - "
                        f"flags: {qa_result.get('qa_flags', [])}"
                    )
                    stats["skipped"] += 1
                    continue

                # Check if item already exists
                existing = repo.get_by_id(item_id, conn)
                if existing:
                    logger.warning(f"Skipping item {item_id}: Already exists in database")
                    stats["skipped"] += 1
                    continue

                # Load item
                repo.create(item, qa_result, conn)
                stats["loaded"] += 1

                if idx % 100 == 0:
                    logger.info(f"Progress: {idx}/{len(items)} items processed")

            except Exception as e:
                logger.error(f"Failed to load item {item_id}: {e}")
                stats["failed"] += 1

    # Report statistics
    logger.info("=" * 50)
    logger.info("Loading complete!")
    logger.info(f"Total items:    {stats['total']}")
    logger.info(f"Loaded:         {stats['loaded']}")
    logger.info(f"Skipped:        {stats['skipped']}")
    logger.info(f"Failed:         {stats['failed']}")
    logger.info("=" * 50)

    db.close_all()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Load validated items from JSON/JSONL files into the Item Bank"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (JSON or JSONL)"
    )
    parser.add_argument(
        "--section",
        choices=["reading_writing", "math", "rw", "readingwriting"],
        help="Optional section filter (only load items from this section)"
    )

    args = parser.parse_args()

    # Normalize section name
    section = None
    if args.section:
        section = args.section
        if section in ["rw", "readingwriting"]:
            section = "reading_writing"

    load_items(args.input, section)


if __name__ == "__main__":
    main()
