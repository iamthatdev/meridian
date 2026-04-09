"""
Tests for SFTDataset class.

.. deprecated::
    These tests cover the deprecated SFTDataset class.
    New code should use trl.SFTTrainer instead.
    See: tests/test_loss_masking.py for tests of the new approach.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


class TestSFTDataset:
    """Tests for SFTDataset class."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample training data file."""
        data_file = tmp_path / "train.jsonl"

        examples = [
            {
                "dataset_version": "math-sft-v1.0",
                "schema_version": "item-schema-v1",
                "section": "math",
                "domain": "algebra.quadratic_equations",
                "difficulty_tier": "medium",
                "messages": [
                    {"role": "system", "content": "You are an expert SAT item writer."},
                    {"role": "user", "content": "Generate a math item"},
                    {"role": "assistant", "content": '{"question": "What is 2+2?"}'}
                ]
            },
            {
                "dataset_version": "math-sft-v1.0",
                "schema_version": "item-schema-v1",
                "section": "math",
                "domain": "linear_equations",
                "difficulty_tier": "easy",
                "messages": [
                    {"role": "system", "content": "You are an expert SAT item writer."},
                    {"role": "user", "content": "Generate a math item"},
                    {"role": "assistant", "content": '{"question": "Solve x + 1 = 3"}'}
                ]
            },
            {
                "dataset_version": "rw-sft-v1.0",
                "schema_version": "item-schema-v1",
                "section": "reading_writing",
                "domain": "central_idea",
                "difficulty_tier": "medium",
                "messages": [
                    {"role": "system", "content": "You are an expert SAT item writer."},
                    {"role": "user", "content": "Generate an RW item"},
                    {"role": "assistant", "content": '{"question": "What is the main idea?"}'}
                ]
            }
        ]

        with open(data_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        return data_file

    def test_dataset_load_data(self, sample_data):
        """Test loading data from JSONL file."""
        from src.training.dataset import SFTDataset
        import torch

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = "formatted prompt"

        # Mock tokenize to return tensors
        mock_encodings = {
            "input_ids": torch.zeros((1, 100), dtype=torch.long),
            "attention_mask": torch.ones((1, 100), dtype=torch.long)
        }
        mock_tokenizer.return_value = mock_encodings

        try:
            dataset = SFTDataset(
                data_path=str(sample_data),
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                section=None
            )

            assert len(dataset) == 3

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_dataset_filter_by_section(self, sample_data):
        """Test filtering by section."""
        from src.training.dataset import SFTDataset
        from unittest.mock import MagicMock
        import torch

        mock_tokenizer = MagicMock()
        mock_encodings = {
            "input_ids": torch.zeros((1, 100), dtype=torch.long),
            "attention_mask": torch.ones((1, 100), dtype=torch.long)
        }
        mock_tokenizer.return_value = mock_encodings
        mock_tokenizer.apply_chat_template.return_value = "formatted"
        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = "formatted"

        try:
            dataset = SFTDataset(
                data_path=str(sample_data),
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                section="math"
            )

            # Should only load math items (2 out of 3)
            assert len(dataset) == 2

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_dataset_missing_file(self):
        """Test handling of missing file."""
        from src.training.dataset import SFTDataset

        mock_tokenizer = MagicMock()

        try:
            with pytest.raises(FileNotFoundError):
                SFTDataset(
                    data_path="/nonexistent/file.jsonl",
                    tokenizer=mock_tokenizer,
                    max_seq_length=512
                )

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_collate_fn(self):
        """Test collate function for batching."""
        from src.training.dataset import SFTDataset
        import torch

        try:
            # Create mock batch
            batch = [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                    "labels": torch.tensor([1, 2, 3])
                },
                {
                    "input_ids": torch.tensor([4, 5, 6]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                    "labels": torch.tensor([4, 5, 6])
                }
            ]

            result = SFTDataset.collate_fn(batch)

            assert result["input_ids"].shape == (2, 3)
            assert result["attention_mask"].shape == (2, 3)
            assert result["labels"].shape == (2, 3)

        except ImportError:
            pytest.skip("PyTorch not available")
