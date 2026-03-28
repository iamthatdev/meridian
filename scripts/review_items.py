#!/usr/bin/env python3
"""
Review items (approve or reject with reasons).

Usage:
    python scripts/review_items.py --item-id 550e8400-e29b-41d4-a716-446655440001 --decision approve --reviewer-id user-123
    python scripts/review_items.py --item-id 550e8400-e29b-41d4-a716-446655440001 --decision reject --rejection-reasons INCORRECT_ANSWER --reviewer-id user-456
"""

import argparse
import sys
from loguru import logger

from src.config import load_config
from src.item_bank.database import DatabaseManager
from src.item_bank.repositories.item_repository import ItemRepository


# Valid rejection reasons
VALID_REJECTION_REASONS = [
    "INCORRECT_ANSWER",
    "POOR_DISTRACTOR_QUALITY",
    "AMBIGUOUS_QUESTION",
    "CONTENT_MISMATCH",
    "GRAMMAR_ERRORS",
    "RATIONALE_INSUFFICIENT",
    "PASSAGE_ISSUES",
    "OTHER"
]


def review_item(
    item_id: str,
    decision: str,
    reviewer_id: str,
    rejection_reasons: list = None,
    notes: str = None
):
    """
    Review an item (approve or reject).

    Args:
        item_id: UUID of the item to review
        decision: Review decision (approve or reject)
        reviewer_id: UUID of the reviewer
        rejection_reasons: List of rejection reasons (if rejecting)
        notes: Optional review notes
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
        logger.info(f"Reviewing item {item_id}")
        logger.info(f"Current status: {current_status}")
        logger.info(f"Decision: {decision}")

        if decision == "approve":
            # Approve: move to operational
            if current_status == "operational":
                logger.warning("Item is already operational")
                sys.exit(0)

            logger.info(f"Moving item to: operational")
            repo.update_status(
                item_id,
                "operational",
                conn,
                reviewer_id=reviewer_id
            )

            logger.info(f"✅ Item {item_id} approved successfully!")
            logger.info(f"   Status: {current_status} → operational")
            logger.info(f"   Reviewer: {reviewer_id}")

        elif decision == "reject":
            # Reject: move to retired
            if current_status == "retired":
                logger.warning("Item is already retired")
                sys.exit(0)

            # Validate rejection reasons
            if not rejection_reasons:
                logger.error("Rejection reasons are required when rejecting an item")
                logger.info(f"Valid reasons: {', '.join(VALID_REJECTION_REASONS)}")
                sys.exit(1)

            invalid_reasons = [r for r in rejection_reasons if r not in VALID_REJECTION_REASONS]
            if invalid_reasons:
                logger.error(f"Invalid rejection reasons: {', '.join(invalid_reasons)}")
                logger.info(f"Valid reasons: {', '.join(VALID_REJECTION_REASONS)}")
                sys.exit(1)

            logger.info(f"Rejection reasons: {', '.join(rejection_reasons)}")
            if notes:
                logger.info(f"Notes: {notes}")

            # Update status to retired
            repo.update_status(
                item_id,
                "retired",
                conn,
                reviewer_id=reviewer_id,
                rejection_reasons=rejection_reasons,
                notes=notes
            )

            logger.info(f"✅ Item {item_id} rejected successfully!")
            logger.info(f"   Status: {current_status} → retired")
            logger.info(f"   Reviewer: {reviewer_id}")
            logger.info(f"   Reasons: {', '.join(rejection_reasons)}")

        else:
            logger.error(f"Invalid decision: {decision}")
            logger.info("Valid decisions: approve, reject")
            sys.exit(1)

    db.close_all()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Review items (approve or reject with reasons)"
    )
    parser.add_argument(
        "--item-id",
        required=True,
        help="UUID of the item to review"
    )
    parser.add_argument(
        "--decision",
        required=True,
        choices=["approve", "reject"],
        help="Review decision"
    )
    parser.add_argument(
        "--reviewer-id",
        required=True,
        help="UUID of the reviewer"
    )
    parser.add_argument(
        "--rejection-reasons",
        nargs="+",
        choices=VALID_REJECTION_REASONS,
        help="Rejection reasons (required if decision=reject)"
    )
    parser.add_argument(
        "--notes",
        help="Optional review notes"
    )

    args = parser.parse_args()

    review_item(
        item_id=args.item_id,
        decision=args.decision,
        reviewer_id=args.reviewer_id,
        rejection_reasons=args.rejection_reasons,
        notes=args.notes
    )


if __name__ == "__main__":
    main()
