#!/usr/bin/env python3
"""
Approve an item and move it to the next status in the lifecycle.

Usage:
    python scripts/approve_item.py --item-id 550e8400-e29b-41d4-a716-446655440001 --reviewer-id user-123
"""

import argparse
import sys
from loguru import logger

from src.config import load_config
from src.item_bank.database import DatabaseManager
from src.item_bank.repositories.item_repository import ItemRepository


# Valid status transitions
VALID_TRANSITIONS = {
    "draft": "pretesting",
    "pretesting": "operational",
}


def approve_item(item_id: str, reviewer_id: str):
    """
    Approve an item and move it to the next status.

    Args:
        item_id: UUID of the item to approve
        reviewer_id: UUID of the reviewer approving the item
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

        # Get current item
        item = repo.get_by_id(item_id, conn)
        if not item:
            logger.error(f"Item not found: {item_id}")
            sys.exit(1)

        current_status = item["status"]
        logger.info(f"Item {item_id} current status: {current_status}")

        # Check if item can be approved
        if current_status not in VALID_TRANSITIONS:
            logger.error(
                f"Item cannot be approved from status '{current_status}'. "
                f"Valid transitions: {list(VALID_TRANSITIONS.keys())}"
            )
            sys.exit(1)

        # Determine next status
        next_status = VALID_TRANSITIONS[current_status]
        logger.info(f"Moving item to status: {next_status}")

        # Update status
        repo.update_status(
            item_id,
            next_status,
            conn,
            reviewer_id=reviewer_id
        )

        logger.info(f"✅ Item {item_id} approved successfully!")
        logger.info(f"   Status: {current_status} → {next_status}")
        logger.info(f"   Reviewer: {reviewer_id}")

    db.close_all()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Approve an item and move it to the next status"
    )
    parser.add_argument(
        "--item-id",
        required=True,
        help="UUID of the item to approve"
    )
    parser.add_argument(
        "--reviewer-id",
        required=True,
        help="UUID of the reviewer approving the item"
    )

    args = parser.parse_args()

    approve_item(args.item_id, args.reviewer_id)


if __name__ == "__main__":
    main()
