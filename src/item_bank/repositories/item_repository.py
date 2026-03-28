"""Item repository for CRUD operations on the Item Bank."""

import uuid
from datetime import datetime
from loguru import logger
from src.item_bank.database import db
from typing import Dict, Any, List, Optional


class ItemRepository:
    """Repository for item CRUD operations."""

    def create(self, item: Dict[str, Any], qa_result: Dict[str, Any], conn) -> str:
        """
        Create a new item in the Item Bank.

        Args:
            item: Item dictionary with section, domain, difficulty, content_json
            qa_result: Auto-QA validation result
            conn: Database connection (from context manager)

        Returns:
            The created item's UUID
        """
        item_id = item.get("id", str(uuid.uuid4()))

        cur = conn.cursor()

        # Build the INSERT query
        cur.execute("""
            INSERT INTO items (
                id, status, created_at, updated_at,
                section, domain, difficulty,
                irt_a, irt_b, irt_c, irt_source,
                content_json,
                auto_qa_passed, qa_score, qa_flags,
                model_version
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s,
                %s, %s, %s,
                %s
            )
        """, (
            item_id,
            'draft',
            datetime.utcnow(),
            datetime.utcnow(),
            item['section'],
            item['domain'],
            item['difficulty'],
            1.0,  # irt_a (discrimination)
            0.0,  # irt_b (difficulty)
            0.25,  # irt_c (guessing)
            'seeded',  # irt_source
            item['content_json'],
            qa_result.get('auto_qa_passed', False),
            qa_result.get('qa_score', 0.0),
            qa_result.get('qa_flags', []),
            item.get('model_version', 'unknown')
        ))

        logger.info(f"Item created: {item_id}")
        return item_id

    def get_by_id(self, item_id: str, conn) -> Optional[Dict[str, Any]]:
        """
        Get item by ID.

        Args:
            item_id: UUID of the item
            conn: Database connection

        Returns:
            Item dictionary or None if not found
        """
        cur = conn.cursor()
        cur.execute("SELECT * FROM items WHERE id = %s", (item_id,))
        row = cur.fetchone()

        if row is None:
            return None

        # Convert to dictionary (assuming column order from schema)
        return self._row_to_dict(cur, row)

    def query(self, conn, **filters) -> List[Dict[str, Any]]:
        """
        Query items with filters.

        Args:
            conn: Database connection
            **filters: Query filters (status, section, domain, difficulty, limit)

        Returns:
            List of item dictionaries
        """
        query = "SELECT * FROM items WHERE 1=1"
        params = []

        if "status" in filters:
            query += " AND status = %s"
            params.append(filters["status"])

        if "section" in filters:
            query += " AND section = %s"
            params.append(filters["section"])

        if "domain" in filters:
            query += " AND domain = %s"
            params.append(filters["domain"])

        if "difficulty" in filters:
            query += " AND difficulty = %s"
            params.append(filters["difficulty"])

        query += " ORDER BY created_at DESC"

        if "limit" in filters:
            query += " LIMIT %s"
            params.append(filters["limit"])

        cur = conn.cursor()
        cur.execute(query, params)

        results = []
        for row in cur.fetchall():
            results.append(self._row_to_dict(cur, row))

        return results

    def update_status(
        self,
        item_id: str,
        new_status: str,
        conn,
        reviewer_id: Optional[str] = None,
        rejection_reasons: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        """
        Update item status and log the change.

        Args:
            item_id: UUID of the item
            new_status: New status (draft/pretesting/operational/retired)
            conn: Database connection
            reviewer_id: UUID of the reviewer (if human action)
            rejection_reasons: List of rejection reasons (if rejecting)
            notes: Optional notes
        """
        cur = conn.cursor()

        # Get old status
        cur.execute("SELECT status FROM items WHERE id = %s", (item_id,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Item {item_id} not found")
        old_status = result[0]

        # Update status
        cur.execute("""
            UPDATE items
            SET status = %s, updated_at = %s
            WHERE id = %s
        """, (new_status, datetime.utcnow(), item_id))

        # Log status change in audit_log
        audit_log_id = str(uuid.uuid4())
        cur.execute("""
            INSERT INTO audit_log (log_id, item_id, timestamp, action_type, actor_type, actor_id, before_status, after_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            audit_log_id,
            item_id,
            datetime.utcnow(),
            f"status_change_to_{new_status}",
            'user' if reviewer_id else 'system',
            reviewer_id,
            old_status,
            new_status
        ))

        # If rejecting, create review record
        if new_status == 'retired' and rejection_reasons:
            review_record_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO review_records (record_id, item_id, reviewer_id, timestamp, decision, rejection_reasons, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                review_record_id,
                item_id,
                reviewer_id,
                datetime.utcnow(),
                'rejected',
                rejection_reasons,
                notes
            ))

        logger.info(f"Item {item_id}: {old_status} → {new_status}")

    def _row_to_dict(self, cur, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        # Get column names from cursor description
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
