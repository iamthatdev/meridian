#!/usr/bin/env python3
"""
Validate items using the Auto-QA pipeline.

Usage:
    # Validate items from file
    python scripts/validate_items.py --input data/generated/items.jsonl

    # Validate and save passed items
    python scripts/validate_items.py --input data/generated/items.jsonl --output data/validated/items.jsonl

    # Show detailed results
    python scripts/validate_items.py --input data/generated/items.jsonl --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

from src.auto_qa.pipeline import AutoQAPipeline


def validate_items(
    input_path: str,
    output: str = None,
    verbose: bool = False
) -> Tuple[Dict[str, int], List[Dict], List[Dict]]:
    """
    Validate items using Auto-QA pipeline.

    Args:
        input_path: Path to input file (JSON or JSONL)
        output: Optional path to save validated items
        verbose: Whether to show detailed results

    Returns:
        Tuple of (statistics, passed_items, failed_items)
    """
    # Initialize pipeline
    pipeline = AutoQAPipeline()

    # Load items
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

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
            if isinstance(data, list):
                items = data
            else:
                items = [data]
    else:
        logger.error(f"Unsupported file format: {input_file.suffix}")
        logger.info("Supported formats: .json, .jsonl")
        sys.exit(1)

    logger.info(f"Loaded {len(items)} items")

    # Validate items
    logger.info("Running Auto-QA validation...")

    passed_items = []
    failed_items = []

    for idx, item in enumerate(items, 1):
        item_id = item.get("id", f"item-{idx}")

        try:
            result = pipeline.validate(item)

            if verbose:
                logger.info(f"\nItem {idx}/{len(items)}: {item_id}")
                logger.info(f"  Schema valid: {result.get('schema_valid', False)}")
                logger.info(f"  Auto-QA passed: {result.get('auto_qa_passed', False)}")
                logger.info(f"  QA Score: {result.get('qa_score', 0.0)}")
                if result.get('qa_flags'):
                    logger.info(f"  Flags: {', '.join(result.get('qa_flags', []))}")

            if result.get("auto_qa_passed", False):
                passed_items.append(item)
                logger.debug(f"✓ {item_id} passed validation")
            else:
                failed_items.append({
                    "item": item,
                    "result": result
                })
                logger.warning(f"✗ {item_id} failed validation: {result.get('qa_flags', [])}")

        except Exception as e:
            logger.error(f"Error validating {item_id}: {e}")
            failed_items.append({
                "item": item,
                "result": {"error": str(e)}
            })

    # Calculate statistics
    total = len(items)
    passed = len(passed_items)
    failed = len(failed_items)

    stats = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total * 100) if total > 0 else 0.0
    }

    # Save passed items if output specified
    if output and passed_items:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".jsonl":
            with open(output_path, "w") as f:
                for item in passed_items:
                    f.write(json.dumps(item) + "\n")
        elif output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(passed_items, f, indent=2)
        else:
            logger.error(f"Unsupported output format: {output_path.suffix}")
            sys.exit(1)

        logger.info(f"Saved {passed} validated items to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Total items:    {stats['total']}")
    print(f"Passed:         {stats['passed']} ({stats['pass_rate']:.1f}%)")
    print(f"Failed:         {stats['failed']}")
    print("=" * 60)

    # Print failure details if any
    if failed_items and not verbose:
        logger.info("\nTop failure reasons:")
        flag_counts = {}
        for fail_info in failed_items:
            flags = fail_info.get("result", {}).get("qa_flags", [])
            for flag in flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        for flag, count in sorted(flag_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {flag}: {count}")

    return stats, passed_items, failed_items


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate items using the Auto-QA pipeline"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (JSON or JSONL)"
    )
    parser.add_argument(
        "--output",
        help="Path to save validated items (JSON or JSONL)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation results"
    )

    args = parser.parse_args()

    # Validate
    validate_items(
        input_path=args.input,
        output=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
