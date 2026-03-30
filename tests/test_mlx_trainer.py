"""
Tests for MLXTrainer class.

Integration-style tests that verify trainer structure without requiring actual MLX.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.training.mlx_trainer import MLXTrainer


def test_trainer_initialization():
    """Test that trainer can be initialized with mock components."""
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_optimizer = Mock()

    # Create temporary data files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as train_f:
        train_f.write('{"section": "math", "domain": "test", "difficulty": "easy", "content_json": {"question": "Test", "choices": [], "correct_answer": "A", "correct_answer_text": "A", "rationale": "Test"}}\n')
        train_path = train_f.name

    try:
        # Mock MLX availability and optimizer
        with patch('src.training.mlx_trainer.MLX_AVAILABLE', True):
            with patch('src.training.mlx_dataset.MLX_AVAILABLE', True):
                with patch('src.training.mlx_trainer.optim') as mock_optim:
                    mock_optim.AdamW.return_value = mock_optimizer
                    trainer = MLXTrainer(
                        model=mock_model,
                        tokenizer=mock_tokenizer,
                        train_path=train_path,
                        val_path=None,
                        config=Mock(learning_rate=1e-4, max_seq_length=512)
                    )

                    assert trainer.model == mock_model
                    assert trainer.tokenizer == mock_tokenizer
                    assert trainer.config is not None
                    assert trainer.optimizer == mock_optimizer
    finally:
        Path(train_path).unlink()


def test_trainer_requires_mlx():
    """Test that trainer raises ImportError when MLX is not available."""
    mock_model = Mock()
    mock_tokenizer = Mock()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as train_f:
        train_f.write('{"section": "math", "domain": "test", "difficulty": "easy", "content_json": {"question": "Test", "choices": [], "correct_answer": "A", "correct_answer_text": "A", "rationale": "Test"}}\n')
        train_path = train_f.name

    try:
        with patch('src.training.mlx_trainer.MLX_AVAILABLE', False):
            with pytest.raises(ImportError, match="MLX is required"):
                MLXTrainer(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    train_path=train_path,
                    val_path=None,
                    config=Mock(learning_rate=1e-4, max_seq_length=512)
                )
    finally:
        Path(train_path).unlink()


def test_trainer_with_validation_data():
    """Test that trainer loads validation dataset when path is provided."""
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_optimizer = Mock()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as train_f:
        train_f.write('{"section": "math", "domain": "test", "difficulty": "easy", "content_json": {"question": "Test", "choices": [], "correct_answer": "A", "correct_answer_text": "A", "rationale": "Test"}}\n')
        train_path = train_f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as val_f:
        val_f.write('{"section": "math", "domain": "test", "difficulty": "easy", "content_json": {"question": "Test", "choices": [], "correct_answer": "A", "correct_answer_text": "A", "rationale": "Test"}}\n')
        val_path = val_f.name

    try:
        with patch('src.training.mlx_trainer.MLX_AVAILABLE', True):
            with patch('src.training.mlx_dataset.MLX_AVAILABLE', True):
                with patch('src.training.mlx_trainer.optim') as mock_optim:
                    mock_optim.AdamW.return_value = mock_optimizer
                    trainer = MLXTrainer(
                        model=mock_model,
                        tokenizer=mock_tokenizer,
                        train_path=train_path,
                        val_path=val_path,
                        config=Mock(learning_rate=1e-4, max_seq_length=512)
                    )

                    assert trainer.val_dataset is not None
    finally:
        Path(train_path).unlink()
        Path(val_path).unlink()
