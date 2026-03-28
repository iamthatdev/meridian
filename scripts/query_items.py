#!/usr/bin/env python3
"""
Query and display items from the Item Bank.

Usage:
    python scripts/query_items.py --section math --limit 10
    python scripts/query_items.py --status draft --section reading_writing
"""

import argparse
import json
import sys
from loguru import logger

from src.config import load_config
from src.item_bank.database import DatabaseManager
from src.item_bank.repositories.item_repository import ItemRepository


def query_items(
    section: str = None,
    domain: str = None,
    difficulty: str = None,
    status: str = None,
    limit: int = 50,
    offset: int = 0,
    output_format: str = "table"
):
    """
    Query items from the Item Bank.

    Args:
        section: Filter by section (reading_writing, math)
        domain: Filter by domain
        difficulty: Filter by difficulty (easy, medium, hard)
        status: Filter by status (draft, pretesting, operational, retired)
        limit: Maximum number of items to return
        offset: Number of items to skip
        output_format: Output format (table, json)
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

        filters["limit"] = limit

        # Query items
        items = repo.query(conn, **filters)

        # Apply offset
        if offset > 0:
            items = items[offset:]

        logger.info(f"Found {len(items)} items")

        if output_format == "json":
            # Output as JSON
            print(json.dumps(items, indent=2))
        else:
            # Output as table
            display_items_table(items)

    db.close_all()


def display_items_table(items: list):
    """Display items in a formatted table."""
    if not items:
        logger.info("No items found")
        return

    # Print header
    print("=" * 120)
    print(
        f"{'ID':<38} {'Section':<15} {'Domain':<30} {'Difficulty':<12} {'Status':<12} {'QA Passed':<10}"
    )
    print("=" * 120)

    # Print items
    for item in items:
        item_id = item["id"]
        section = item.get("section", "N/A")
        domain = item.get("domain", "N/A")
        difficulty = item.get("difficulty", "N/A")
        status = item.get("status", "N/A")
        qa_passed = "✓" if item.get("auto_qa_passed", False) else "✗"

        # Truncate domain if too long
        if len(domain) > 28:
            domain = domain[:25] + "..."

        print(
            f"{item_id:<38} {section:<15} {domain:<30} {difficulty:<12} {status:<12} {qa_passed:<10}"
        )

    print("=" * 120)
    print(f"Total: {len(items)} items")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query and display items from the Item Bank"
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
        "--limit",
        type=int,
        default=50,
        help="Maximum number of items to return (default: 50)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of items to skip (default: 0)"
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )

    args = parser.parse_args()

    # Normalize section name
    section = None
    if args.section:
        section = args.section
        if section in ["rw", "readingwriting"]:
            section = "reading_writing"

    query_items(
        section=section,
        domain=args.domain,
        difficulty=args.difficulty,
        status=args.status,
        limit=args.limit,
        offset=args.offset,
        output_format=args.format
    )


if __name__ == "__main__":
    main()
