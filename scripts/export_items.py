#!/usr/bin/env python3
"""
Export items from the Item Bank to JSON/JSONL format.

Usage:
    python scripts/export_items.py --output data/exports/operational_items.jsonl --status operational
    python scripts/export_items.py --output data/exports/math_items.json --section math --status operational
"""

import argparse
import json
import sys
from pathlib import Path
from loguru import logger

from src.config import load_config
from src.item_bank.database import DatabaseManager
from src.item_bank.repositories.item_repository import ItemRepository


def export_items(
    output_path: str,
    section: str = None,
    domain: str = None,
    difficulty: str = None,
    status: str = None,
    include_metadata: bool = True
):
    """
    Export items from the Item Bank.

    Args:
        output_path: Path to output file (JSON or JSONL)
        section: Filter by section
        domain: Filter by domain
        difficulty: Filter by difficulty
        status: Filter by status
        include_metadata: Include auto-QA metadata in export
    """
    # Initialize database
    config = load_config()
    db = DatabaseManager()

    try:
        db.initialize(config)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    with db.get_connection() as conn:
        repo = ItemRepository()

        # Build filters
        filters = {}
        if section:
            filters["section"] = section
        if domain:
            filters["domain"] = domain
        if difficulty:
            filters["difficulty"] = difficulty
        if status:
            filters["status"] = status

        # No limit for exports - get all matching items
        # But we'll use a large limit for practical purposes
        filters["limit"] = 100000

        # Query items
        items = repo.query(conn, **filters)

        if not items:
            logger.warning("No items found matching the specified filters")
            db.close_all()
            return

        logger.info(f"Found {len(items)} items to export")

        # Prepare output data
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if include_metadata:
            # Include database fields (auto_qa_passed, qa_score, etc.)
            export_data = items
        else:
            # Export only the core item data (as generated)
            export_data = []
            for item in items:
                export_item = {
                    "id": item["id"],
                    "section": item["section"],
                    "domain": item["domain"],
                    "difficulty": item["difficulty"],
                    "content_json": item["content_json"],
                    "model_version": item.get("model_version", "unknown")
                }
                export_data.append(export_item)

        # Write output file
        if output_file.suffix == ".jsonl":
            with open(output_file, "w") as f:
                for item in export_data:
                    f.write(json.dumps(item) + "\n")
        elif output_file.suffix == ".json":
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            logger.error(f"Unsupported output format: {output_file.suffix}")
            logger.info("Supported formats: .json, .jsonl")
            sys.exit(1)

        logger.info(f"✅ Exported {len(export_data)} items to {output_path}")

        # Print summary statistics
        sections = {}
        difficulties = {}
        for item in export_data:
            sec = item.get("section", "unknown")
            diff = item.get("difficulty", "unknown")
            sections[sec] = sections.get(sec, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1

        logger.info("=" * 50)
        logger.info("Export Summary:")
        logger.info(f"  Total items: {len(export_data)}")
        logger.info("  By section:")
        for section, count in sorted(sections.items()):
            logger.info(f"    {section}: {count}")
        logger.info("  By difficulty:")
        for difficulty, count in sorted(difficulties.items()):
            logger.info(f"    {difficulty}: {count}")
        logger.info("=" * 50)

    db.close_all()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export items from the Item Bank to JSON/JSONL format"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output file (JSON or JSONL)"
    )
    parser.add_argument(
        "--section",
        choices=["reading_writing", "math", "rw", "readingwriting"],
        help="Filter by section"
    )
    parser.add_argument(
        "--domain",
        help="Filter by domain"
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty"
    )
    parser.add_argument(
        "--status",
        choices=["draft", "pretesting", "operational", "retired"],
        help="Filter by status"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude auto-QA metadata from export"
    )

    args = parser.parse_args()

    # Normalize section name
    section = None
    if args.section:
        section = args.section
        if section in ["rw", "readingwriting"]:
            section = "reading_writing"

    export_items(
        output_path=args.output,
        section=section,
        domain=args.domain,
        difficulty=args.difficulty,
        status=args.status,
        include_metadata=not args.no_metadata
    )


if __name__ == "__main__":
    main()
