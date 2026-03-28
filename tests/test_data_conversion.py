"""Tests for ItemBank conversion script."""

import json
import pytest
import tempfile
import shutil
from pathlib import Path


def test_convert_itembank():
    """Test ItemBank conversion script."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create sample ItemBank data
        sample_data = [
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "section": "math",
                "domain": "algebra.quadratic_equations",
                "difficulty_tier": "medium",
                "content_json": {
                    "passage": None,
                    "question": "If x^2 - 5x + 6 = 0, what are the roots?",
                    "math_format": "plain",
                    "choices": [
                        {"label": "A", "text": "2 and 3"},
                        {"label": "B", "text": "-2 and -3"},
                        {"label": "C", "text": "2 and -3"},
                        {"label": "D", "text": "-2 and 3"}
                    ],
                    "correct_answer": "A",
                    "correct_answer_text": "2 and 3",
                    "rationale": "Test rationale that is long enough to pass validation",
                    "metadata": {"topic": "quadratic equations"}
                }
            },
            {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "section": "reading_writing",
                "domain": "central_idea",
                "difficulty_tier": "easy",
                "content_json": {
                    "passage": "This is a test passage about a topic.",
                    "question": "What is the main idea?",
                    "math_format": "plain",
                    "choices": [
                        {"label": "A", "text": "Idea A"},
                        {"label": "B", "text": "Idea B"},
                        {"label": "C", "text": "Idea C"},
                        {"label": "D", "text": "Idea D"}
                    ],
                    "correct_answer": "A",
                    "correct_answer_text": "Idea A",
                    "rationale": "This is the main idea of the passage provided above."
                }
            }
        ]

        source_file = tmpdir / "itembank.json"
        with open(source_file, "w") as f:
            json.dump(sample_data, f)

        # Run conversion
        from scripts.convert_itembank import convert_itembank
        output_dir = tmpdir / "output"
        convert_itembank(str(source_file), str(output_dir))

        # Verify output files exist
        assert (output_dir / "math_train.jsonl").exists()
        assert (output_dir / "math_val.jsonl").exists()
        assert (output_dir / "rw_train.jsonl").exists()
        assert (output_dir / "rw_val.jsonl").exists()

        # Verify JSONL format
        with open(output_dir / "math_train.jsonl") as f:
            for line in f:
                example = json.loads(line)
                assert "messages" in example
                assert example["section"] == "math"
                # Verify chat message structure
                assert example["messages"][0]["role"] == "system"
                assert example["messages"][1]["role"] == "user"
                assert example["messages"][2]["role"] == "assistant"
                break

        # Verify split (1 Math item = 0 train, 1 val due to 85% split)
        with open(output_dir / "math_train.jsonl") as f:
            math_train_count = sum(1 for _ in f)
        assert math_train_count == 0  # 1 item * 0.85 = 0.85 → 0

        with open(output_dir / "math_val.jsonl") as f:
            math_val_count = sum(1 for _ in f)
        assert math_val_count == 1  # 1 item * 0.15 = 0.15 → 1
