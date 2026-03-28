"""Tests for Item repository CRUD operations.

These tests require PostgreSQL to be available. Tests are skipped if
the database is not accessible.
"""

import os
import pytest
from psycopg2 import OperationalError
from src.item_bank.repositories.item_repository import ItemRepository
from src.item_bank.database import DatabaseManager
from src.config import load_config


@pytest.fixture
def test_db():
    """Test database configuration."""
    config = load_config("local")
    return config


@pytest.fixture
def clean_db(test_db):
    """Ensure database manager is clean before and after tests."""
    # Clean up before test
    if DatabaseManager.is_initialized():
        DatabaseManager.close_all()

    yield test_db

    # Clean up after test
    if DatabaseManager.is_initialized():
        DatabaseManager.close_all()


@pytest.fixture
def check_db_available(clean_db):
    """Check if database is available. Skip test if not."""
    try:
        db = DatabaseManager()
        db.initialize(clean_db)
        # Try to get a connection to verify it works
        with db.get_connection() as conn:
            pass
        db.close_all()
    except OperationalError:
        pytest.skip("PostgreSQL not available")


def test_item_repository_create(clean_db, check_db_available):
    """Test creating an item in the database."""
    database = DatabaseManager()
    database.initialize(clean_db)

    try:
        with database.get_connection() as conn:
            repo = ItemRepository()

            # Create item
            item = {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "section": "math",
                "domain": "algebra.quadratic_equations",
                "difficulty": "medium",
                "content_json": {
                    "passage": None,
                    "question": "If x squared minus 5x plus 6 equals 0, what are the roots? " * 2,
                    "math_format": "latex",
                    "choices": [
                        {"label": "A", "text": "2 and 3"},
                        {"label": "B", "text": "-2 and -3"},
                        {"label": "C", "text": "2 and -3"},
                        {"label": "D", "text": "-2 and 3"}
                    ],
                    "correct_answer": "A",
                    "correct_answer_text": "2 and 3",
                    "rationale": "Factoring the quadratic equation gives us x minus 2 times x minus 3 equals 0. Setting each factor to zero gives x equals 2 or x equals 3. Therefore the roots are 2 and 3. This is a complete solution showing all steps clearly."
                }
            }
            qa_result = {"auto_qa_passed": True, "qa_score": 1.0, "qa_flags": []}

            item_id = repo.create(item, qa_result, conn)
            assert item_id == item["id"]

            # Verify item was created
            retrieved = repo.get_by_id(item_id, conn)
            assert retrieved is not None
            assert retrieved["section"] == "math"
            assert retrieved["status"] == "draft"

    finally:
        database.close_all()


def test_item_repository_query(clean_db, check_db_available):
    """Test querying items with filters."""
    database = DatabaseManager()
    database.initialize(clean_db)

    try:
        with database.get_connection() as conn:
            repo = ItemRepository()

            # Create multiple items
            for i in range(3):
                item = {
                    "id": f"550e8400-e29b-41d4-a716-446655440{i:03d}",
                    "section": "math" if i % 2 == 0 else "reading_writing",
                    "domain": "algebra.quadratic_equations" if i % 2 == 0 else "central_idea",
                    "difficulty": "medium",
                    "content_json": {
                        "passage": None,
                        "question": f"Test question {i} " * 10,
                        "math_format": "plain",
                        "choices": [
                            {"label": "A", "text": f"Option A{i}"},
                            {"label": "B", "text": f"Option B{i}"},
                            {"label": "C", "text": f"Option C{i}"},
                            {"label": "D", "text": f"Option D{i}"}
                        ],
                        "correct_answer": "A",
                        "correct_answer_text": f"Option A{i}",
                        "rationale": "This is a test rationale for question " + str(i) + ". " * 20
                    }
                }
                qa_result = {"auto_qa_passed": True, "qa_score": 1.0, "qa_flags": []}
                repo.create(item, qa_result, conn)

            # Query all items
            all_items = repo.query(conn, limit=10)
            assert len(all_items) == 3

            # Query by section
            math_items = repo.query(conn, section="math", limit=10)
            assert len(math_items) == 2  # items 0 and 2

            # Query by status (no items yet with non-draft status)
            draft_items = repo.query(conn, status="draft", limit=10)
            assert len(draft_items) == 3

    finally:
        database.close_all()


def test_item_repository_update_status(clean_db, check_db_available):
    """Test updating item status."""
    database = DatabaseManager()
    database.initialize(clean_db)

    try:
        with database.get_connection() as conn:
            repo = ItemRepository()

            # Create item
            item = {
                "id": "550e8400-e29b-41d4-a716-446655440099",
                "section": "math",
                "domain": "algebra",
                "difficulty": "medium",
                "content_json": {
                    "passage": None,
                    "question": "Test question for status update",
                    "math_format": "plain",
                    "choices": [
                        {"label": "A", "text": "A"},
                        {"label": "B", "text": "B"},
                        {"label": "C", "text": "C"},
                        {"label": "D", "text": "D"}
                    ],
                    "correct_answer": "A",
                    "correct_answer_text": "A",
                    "rationale": "This is a test rationale that is long enough to pass validation."
                }
            }
            qa_result = {"auto_qa_passed": True, "qa_score": 1.0, "qa_flags": []}
            item_id = repo.create(item, qa_result, conn)

            # Update status to pretesting
            repo.update_status(item_id, "pretesting", conn, reviewer_id="test-reviewer-123")

            # Verify status changed
            updated = repo.get_by_id(item_id, conn)
            assert updated["status"] == "pretesting"

            # Update to retired (rejection)
            repo.update_status(
                item_id,
                "retired",
                conn,
                reviewer_id="test-reviewer-456",
                rejection_reasons=["INCORRECT_ANSWER", "POOR_DISTRACTOR_QUALITY"],
                notes="The correct answer is actually B, not A"
            )

            # Verify retired status
            retired = repo.get_by_id(item_id, conn)
            assert retired["status"] == "retired"

    finally:
        database.close_all()


def test_item_repository_get_not_found(clean_db, check_db_available):
    """Test getting non-existent item returns None."""
    database = DatabaseManager()
    database.initialize(clean_db)

    try:
        with database.get_connection() as conn:
            repo = ItemRepository()
            result = repo.get_by_id("nonexistent-id", conn)
            assert result is None

    finally:
        database.close_all()
