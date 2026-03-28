"""Tests for CLI scripts."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLoadItems:
    """Tests for load_items.py script."""

    def test_load_items_valid_jsonl(self, tmp_path):
        """Test loading items from valid JSONL file."""
        from scripts.load_items import load_items

        # Create test input file
        input_file = tmp_path / "test_items.jsonl"
        test_items = [
            {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "section": "math",
                "domain": "algebra.quadratic_equations",
                "difficulty": "medium",
                "content_json": {
                    "passage": None,
                    "question": "If x squared minus 5x plus 6 equals 0, what are the roots? This is a valid question that is long enough to pass validation.",
                    "math_format": "latex",
                    "choices": [
                        {"label": "A", "text": "2 and 3"},
                        {"label": "B", "text": "-2 and -3"},
                        {"label": "C", "text": "2 and -3"},
                        {"label": "D", "text": "-2 and 3"}
                    ],
                    "correct_answer": "A",
                    "correct_answer_text": "2 and 3",
                    "rationale": "Factoring the quadratic equation gives us x minus 2 times x minus 3 equals 0. Setting each factor to zero gives x equals 2 or x equals 3. Therefore the roots are 2 and 3."
                }
            }
        ]

        with open(input_file, "w") as f:
            for item in test_items:
                f.write(json.dumps(item) + "\n")

        # Mock database and pipeline
        with patch('scripts.load_items.DatabaseManager') as mock_db_cls, \
             patch('scripts.load_items.AutoQAPipeline') as mock_pipeline_cls:

            # Setup mocks
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db

            mock_conn = MagicMock()
            mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            mock_pipeline = MagicMock()
            mock_pipeline.validate.return_value = {
                "auto_qa_passed": True,
                "qa_score": 1.0,
                "qa_flags": []
            }
            mock_pipeline_cls.return_value = mock_pipeline

            mock_repo = MagicMock()
            mock_repo.get_by_id.return_value = None  # Item doesn't exist
            mock_repo.create.return_value = test_items[0]["id"]

            with patch('scripts.load_items.ItemRepository', return_value=mock_repo):
                # Run load_items
                load_items(str(input_file), section="math")

                # Verify
                mock_pipeline.validate.assert_called_once()
                mock_repo.create.assert_called_once()

    def test_load_items_skip_invalid_qa(self, tmp_path):
        """Test that items failing Auto-QA are skipped."""
        from scripts.load_items import load_items

        # Create test input file
        input_file = tmp_path / "test_items.jsonl"
        test_item = {
            "id": "550e8400-e29b-41d4-a716-446655440001",
            "section": "math",
            "domain": "algebra.quadratic_equations",
            "difficulty": "medium",
            "content_json": {
                "passage": None,
                "question": "Short",  # Too short - will fail validation
                "math_format": "latex",
                "choices": [
                    {"label": "A", "text": "A"},
                    {"label": "B", "text": "B"},
                    {"label": "C", "text": "C"},
                    {"label": "D", "text": "D"}
                ],
                "correct_answer": "A",
                "correct_answer_text": "A",
                "rationale": "Too short" * 10  # Still might fail
            }
        }

        with open(input_file, "w") as f:
            f.write(json.dumps(test_item) + "\n")

        # Mock database and pipeline
        with patch('scripts.load_items.DatabaseManager') as mock_db_cls, \
             patch('scripts.load_items.AutoQAPipeline') as mock_pipeline_cls:

            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db

            mock_conn = MagicMock()
            mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            # Mock pipeline to fail validation
            mock_pipeline = MagicMock()
            mock_pipeline.validate.return_value = {
                "auto_qa_passed": False,
                "qa_score": 0.0,
                "qa_flags": ["Question too short"]
            }
            mock_pipeline_cls.return_value = mock_pipeline

            mock_repo = MagicMock()

            with patch('scripts.load_items.ItemRepository', return_value=mock_repo):
                # Run load_items
                load_items(str(input_file), section="math")

                # Verify item was not created (skipped due to failed QA)
                mock_repo.create.assert_not_called()


class TestQueryItems:
    """Tests for query_items.py script."""

    def test_query_items_table_format(self, capsys):
        """Test querying items with table output."""
        from scripts.query_items import query_items, display_items_table

        # Mock items
        test_items = [
            {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "section": "math",
                "domain": "algebra.quadratic_equations",
                "difficulty": "medium",
                "status": "draft",
                "auto_qa_passed": True
            }
        ]

        # Display items
        display_items_table(test_items)

        # Capture output
        captured = capsys.readouterr()
        assert "550e8400-e29b-41d4-a716-446655440001" in captured.out
        assert "math" in captured.out
        assert "draft" in captured.out

    def test_query_items_empty(self):
        """Test querying when no items found."""
        from scripts.query_items import display_items_table

        # Display empty items - should not raise exception
        display_items_table([])

        # If we get here without exception, test passes
        # The function handles empty lists correctly by logging and returning


class TestReviewItems:
    """Tests for review_items.py script."""

    def test_review_item_approve(self):
        """Test approving an item."""
        from scripts.review_items import review_item

        # Mock database
        with patch('scripts.review_items.DatabaseManager') as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db

            mock_conn = MagicMock()
            mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            mock_repo = MagicMock()
            mock_repo.get_by_id.return_value = {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "pretesting"
            }

            with patch('scripts.review_items.ItemRepository', return_value=mock_repo):
                # Run approve
                review_item(
                    item_id="550e8400-e29b-41d4-a716-446655440001",
                    decision="approve",
                    reviewer_id="user-123"
                )

                # Verify status update
                mock_repo.update_status.assert_called_once_with(
                    "550e8400-e29b-41d4-a716-446655440001",
                    "operational",
                    mock_conn,
                    reviewer_id="user-123"
                )

    def test_review_item_reject(self):
        """Test rejecting an item with reasons."""
        from scripts.review_items import review_item

        # Mock database
        with patch('scripts.review_items.DatabaseManager') as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db

            mock_conn = MagicMock()
            mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            mock_repo = MagicMock()
            mock_repo.get_by_id.return_value = {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "status": "pretesting"
            }

            with patch('scripts.review_items.ItemRepository', return_value=mock_repo):
                # Run reject
                review_item(
                    item_id="550e8400-e29b-41d4-a716-446655440001",
                    decision="reject",
                    reviewer_id="user-456",
                    rejection_reasons=["INCORRECT_ANSWER", "POOR_DISTRACTOR_QUALITY"],
                    notes="The correct answer is actually B"
                )

                # Verify status update with rejection reasons
                mock_repo.update_status.assert_called_once()
                call_args = mock_repo.update_status.call_args

                assert call_args[0][0] == "550e8400-e29b-41d4-a716-446655440001"
                assert call_args[0][1] == "retired"
                assert call_args[1]["rejection_reasons"] == ["INCORRECT_ANSWER", "POOR_DISTRACTOR_QUALITY"]
                assert call_args[1]["notes"] == "The correct answer is actually B"


class TestExportItems:
    """Tests for export_items.py script."""

    def test_export_items_jsonl(self, tmp_path):
        """Test exporting items to JSONL format."""
        from scripts.export_items import export_items

        output_file = tmp_path / "export.jsonl"

        # Mock database
        with patch('scripts.export_items.DatabaseManager') as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db

            mock_conn = MagicMock()
            mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            test_items = [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440001",
                    "section": "math",
                    "domain": "algebra.quadratic_equations",
                    "difficulty": "medium",
                    "status": "operational",
                    "content_json": {"question": "Test"},
                    "model_version": "test-v1",
                    "auto_qa_passed": True,
                    "qa_score": 1.0
                }
            ]

            mock_repo = MagicMock()
            mock_repo.query.return_value = test_items

            with patch('scripts.export_items.ItemRepository', return_value=mock_repo):
                # Run export
                export_items(
                    output_path=str(output_file),
                    status="operational",
                    include_metadata=True
                )

                # Verify file was created
                assert output_file.exists()

                # Verify content
                with open(output_file) as f:
                    lines = f.readlines()
                    assert len(lines) == 1
                    exported_item = json.loads(lines[0])
                    assert exported_item["id"] == "550e8400-e29b-41d4-a716-446655440001"
                    assert exported_item["auto_qa_passed"] == True

    def test_export_items_no_metadata(self, tmp_path):
        """Test exporting items without metadata."""
        from scripts.export_items import export_items

        output_file = tmp_path / "export.jsonl"

        # Mock database
        with patch('scripts.export_items.DatabaseManager') as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db

            mock_conn = MagicMock()
            mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=False)

            test_items = [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440001",
                    "section": "math",
                    "domain": "algebra.quadratic_equations",
                    "difficulty": "medium",
                    "status": "operational",
                    "content_json": {"question": "Test"},
                    "model_version": "test-v1",
                    "auto_qa_passed": True,
                    "qa_score": 1.0
                }
            ]

            mock_repo = MagicMock()
            mock_repo.query.return_value = test_items

            with patch('scripts.export_items.ItemRepository', return_value=mock_repo):
                # Run export without metadata
                export_items(
                    output_path=str(output_file),
                    status="operational",
                    include_metadata=False
                )

                # Verify file was created
                assert output_file.exists()

                # Verify content doesn't have metadata
                with open(output_file) as f:
                    lines = f.readlines()
                    exported_item = json.loads(lines[0])
                    assert "auto_qa_passed" not in exported_item
                    assert "qa_score" not in exported_item
                    assert exported_item["id"] == "550e8400-e29b-41d4-a716-446655440001"
