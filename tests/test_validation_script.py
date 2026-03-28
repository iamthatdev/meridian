"""Tests for validation script."""

import json
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestValidateItems:
    """Tests for validate_items function."""

    @patch('scripts.validate_items.AutoQAPipeline')
    def test_validate_all_passed(self, mock_pipeline_cls):
        """Test validation when all items pass."""
        from scripts.validate_items import validate_items

        # Mock pipeline to return all passing
        mock_pipeline = MagicMock()
        mock_pipeline.validate.return_value = {
            "auto_qa_passed": True,
            "qa_score": 1.0,
            "qa_flags": []
        }
        mock_pipeline_cls.return_value = mock_pipeline

        # Create test input file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            test_items = [
                {"id": "1", "section": "math", "domain": "test", "difficulty": "medium", "content_json": {}},
                {"id": "2", "section": "math", "domain": "test", "difficulty": "medium", "content_json": {}}
            ]

            with open(input_file, "w") as f:
                for item in test_items:
                    f.write(json.dumps(item) + "\n")

            # Validate
            stats, passed_items, failed_items = validate_items(
                input_path=str(input_file)
            )

            # Verify
            assert stats["total"] == 2
            assert stats["passed"] == 2
            assert stats["failed"] == 0
            assert len(passed_items) == 2
            assert len(failed_items) == 0

    @patch('scripts.validate_items.AutoQAPipeline')
    def test_validate_some_failed(self, mock_pipeline_cls):
        """Test validation when some items fail."""
        from scripts.validate_items import validate_items

        mock_pipeline = MagicMock()

        # Mock alternating pass/fail
        def mock_validate(item):
            item_id = item.get("id")
            if item_id == "1":
                return {"auto_qa_passed": True, "qa_score": 1.0, "qa_flags": []}
            else:
                return {"auto_qa_passed": False, "qa_score": 0.5, "qa_flags": ["QUESTION_TOO_SHORT"]}

        mock_pipeline.validate = mock_validate
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            test_items = [
                {"id": "1", "section": "math", "domain": "test", "difficulty": "medium", "content_json": {}},
                {"id": "2", "section": "math", "domain": "test", "difficulty": "medium", "content_json": {}}
            ]

            with open(input_file, "w") as f:
                for item in test_items:
                    f.write(json.dumps(item) + "\n")

            stats, passed_items, failed_items = validate_items(
                input_path=str(input_file)
            )

            assert stats["passed"] == 1
            assert stats["failed"] == 1
            assert len(passed_items) == 1
            assert len(failed_items) == 1

    @patch('scripts.validate_items.AutoQAPipeline')
    def test_validate_with_output(self, mock_pipeline_cls):
        """Test validation with output file."""
        from scripts.validate_items import validate_items

        mock_pipeline = MagicMock()
        mock_pipeline.validate.return_value = {
            "auto_qa_passed": True,
            "qa_score": 1.0,
            "qa_flags": []
        }
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            output_file = Path(tmpdir) / "output.jsonl"

            test_items = [
                {"id": "1", "section": "math", "domain": "test", "difficulty": "medium", "content_json": {}}
            ]

            with open(input_file, "w") as f:
                for item in test_items:
                    f.write(json.dumps(item) + "\n")

            stats, passed_items, failed_items = validate_items(
                input_path=str(input_file),
                output=str(output_file)
            )

            # Verify output file was created
            assert output_file.exists()

            with open(output_file) as f:
                content = f.read()
                assert len(content) > 0

    @patch('scripts.validate_items.AutoQAPipeline')
    def test_validate_json_format(self, mock_pipeline_cls):
        """Test validation with JSON input."""
        from scripts.validate_items import validate_items

        mock_pipeline = MagicMock()
        mock_pipeline.validate.return_value = {
            "auto_qa_passed": True,
            "qa_score": 1.0,
            "qa_flags": []
        }
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.json"
            test_items = [
                {"id": "1", "section": "math", "domain": "test", "difficulty": "medium", "content_json": {}}
            ]

            with open(input_file, "w") as f:
                json.dump(test_items, f)

            stats, passed_items, failed_items = validate_items(
                input_path=str(input_file)
            )

            assert stats["total"] == 1
            assert stats["passed"] == 1

    def test_validate_missing_file(self):
        """Test validation with missing input file."""
        from scripts.validate_items import validate_items

        try:
            validate_items(input_path="/nonexistent/file.jsonl")
            assert False, "Should have exited"
        except SystemExit:
            # Expected to exit when file not found
            pass

    @patch('scripts.validate_items.AutoQAPipeline')
    def test_validate_invalid_json(self, mock_pipeline_cls):
        """Test validation with invalid JSON lines."""
        from scripts.validate_items import validate_items

        mock_pipeline = MagicMock()
        mock_pipeline.validate.return_value = {
            "auto_qa_passed": True,
            "qa_score": 1.0,
            "qa_flags": []
        }
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"

            with open(input_file, "w") as f:
                f.write('{"id": "1", "section": "math"}\n')
                f.write('invalid json\n')
                f.write('{"id": "2", "section": "math"}\n')

            stats, passed_items, failed_items = validate_items(
                input_path=str(input_file)
            )

            # Should skip invalid line and process valid ones
            assert stats["total"] == 2
            assert stats["passed"] == 2


class TestValidateCLI:
    """Tests for validate_items.py CLI."""

    @patch('scripts.validate_items.validate_items')
    def test_cli_basic(self, mock_validate):
        """Test basic CLI usage."""
        from scripts.validate_items import main

        test_args = [
            "validate_items.py",
            "--input", "test_items.jsonl"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            mock_validate.assert_called_once()
            call_args = mock_validate.call_args
            assert call_args[1]["input_path"] == "test_items.jsonl"
            assert call_args[1]["output"] is None
            assert call_args[1]["verbose"] is False

    @patch('scripts.validate_items.validate_items')
    def test_cli_with_output(self, mock_validate):
        """Test CLI with output file."""
        from scripts.validate_items import main

        test_args = [
            "validate_items.py",
            "--input", "test_items.jsonl",
            "--output", "validated_items.jsonl"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            if mock_validate.called:
                call_args = mock_validate.call_args
                assert call_args[1]["output"] == "validated_items.jsonl"

    @patch('scripts.validate_items.validate_items')
    def test_cli_verbose(self, mock_validate):
        """Test CLI with verbose flag."""
        from scripts.validate_items import main

        test_args = [
            "validate_items.py",
            "--input", "test_items.jsonl",
            "--verbose"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            if mock_validate.called:
                call_args = mock_validate.call_args
                assert call_args[1]["verbose"] is True
