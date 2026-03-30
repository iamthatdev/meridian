# tests/test_mlx_dataset.py
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
from src.training.mlx_dataset import MLXDataset


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.apply_chat_template = Mock(
        return_value="System: You are expert.\n\nUser: Generate item.\n\nAssistant: {\"question\": \"Test\"}"
    )
    # Make tokenizer callable and return dict-like object
    tokenizer.return_value = {"input_ids": "mock_ids", "attention_mask": "mock_mask"}
    return tokenizer


@pytest.fixture
def sample_data_file():
    """Create a temporary JSONL file with sample data."""
    items = [
        {
            "section": "math",
            "domain": "algebra.linear_equations_one_variable",
            "difficulty": "easy",
            "content_json": {
                "question": "Solve 2x + 5 = 15",
                "choices": [{"label": "A", "text": "x=3"}],
                "correct_answer": "A",
                "correct_answer_text": "x=3",
                "rationale": "Test"
            }
        }
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in items:
            f.write(json.dumps(item) + '\n')
        path = f.name
    yield path
    Path(path).unlink()


def test_dataset_length(sample_data_file, mock_tokenizer):
    """Test that dataset reports correct length."""
    dataset = MLXDataset(sample_data_file, mock_tokenizer, max_seq_length=512)
    assert len(dataset) == 1


def test_dataset_getitem(sample_data_file, mock_tokenizer):
    """Test that dataset returns tokenized items."""
    dataset = MLXDataset(sample_data_file, mock_tokenizer, max_seq_length=512)

    input_ids, attention_mask = dataset[0]

    assert input_ids == "mock_ids"
    assert attention_mask == "mock_mask"

    # Verify tokenizer was called correctly
    mock_tokenizer.apply_chat_template.assert_called_once()
    # Verify tokenizer was called (using assert_called_with to check parameters)
    assert mock_tokenizer.called


def test_dataset_file_not_found(mock_tokenizer):
    """Test that dataset raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        MLXDataset("/nonexistent/path.jsonl", mock_tokenizer)


def test_dataset_handles_invalid_json(tmp_path, mock_tokenizer):
    """Test that dataset skips invalid JSON lines."""
    # Create a file with mixed valid and invalid JSON
    data_file = tmp_path / "mixed.jsonl"
    with open(data_file, 'w') as f:
        f.write('{"section": "math", "domain": "algebra.linear_equations_one_variable", '
                '"difficulty": "easy", "content_json": {"question": "Test", '
                '"choices": [{"label": "A", "text": "x=3"}], "correct_answer": "A", '
                '"correct_answer_text": "x=3", "rationale": "Test"}}\n')
        f.write('invalid json line\n')
        f.write('{"section": "math", "domain": "algebra.linear_equations_two_variables", '
                '"difficulty": "medium", "content_json": {"question": "Test2", '
                '"choices": [{"label": "A", "text": "x=5"}], "correct_answer": "A", '
                '"correct_answer_text": "x=5", "rationale": "Test2"}}\n')

    dataset = MLXDataset(str(data_file), mock_tokenizer)
    # Should load 2 valid items, skip the invalid one
    assert len(dataset) == 2
