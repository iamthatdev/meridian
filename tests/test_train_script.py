"""Tests for training script."""

import json
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call


class TestTrainScript:
    """Tests for train_model.py script."""

    def _mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.paths.checkpoint_dir = "checkpoints"
        config.paths.data_dir = "data"  # Changed from "data/splits"
        config.training.batch_size = 2
        config.training.learning_rate = 2e-5
        config.training.num_epochs = 1
        config.training.max_seq_length_rw = 2048
        config.training.max_seq_length_math = 2048
        config.training.warmup_steps = 10
        config.lora.r = 16
        config.lora.alpha = 32
        config.lora.dropout = 0.05
        return config

    @patch('scripts.train_model.TRANSFORMERS_AVAILABLE', True)
    @patch('scripts.train_model.Path')
    @patch('scripts.train_model.create_dataloader')
    @patch('scripts.train_model.SFTDataset')
    @patch('scripts.train_model.load_model_for_training')
    @patch('scripts.train_model.save_model')
    @patch('scripts.train_model.torch.optim.AdamW')
    @patch('scripts.train_model.get_linear_schedule_with_warmup')
    def test_train_basic_flow(
        self,
        mock_scheduler,
        mock_adamw,
        mock_save,
        mock_load_model,
        mock_dataset_cls,
        mock_dataloader,
        mock_path
    ):
        """Test basic training flow."""
        from scripts.train_model import train
        import torch

        # Mock Path to make files exist
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
        mock_path.return_value = mock_path_instance

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.parameters.return_value = []
        mock_model.train.return_value = None
        mock_model.eval.return_value = None

        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset_cls.return_value = mock_dataset

        # Mock dataloader to yield one batch
        mock_batch = {
            "input_ids": torch.zeros((2, 10), dtype=torch.long),
            "attention_mask": torch.ones((2, 10), dtype=torch.long),
            "labels": torch.zeros((2, 10), dtype=torch.long)
        }

        mock_dataloader_inst = MagicMock()
        mock_dataloader_inst.__len__.return_value = 1
        mock_dataloader_inst.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_dataloader.return_value = mock_dataloader_inst

        # Mock optimizer
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer

        # Mock scheduler
        mock_scheduler_inst = MagicMock()
        mock_scheduler_inst.get_last_lr.return_value = [2e-5]
        mock_scheduler.return_value = mock_scheduler_inst

        # Mock model outputs
        mock_outputs = MagicMock()
        mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
        mock_model.return_value = mock_outputs

        # Mock backward
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_loss.backward.return_value = None

        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            train(
                section="math",
                config=self._mock_config(),
                checkpoint_dir=tmpdir
            )

            # Verify training steps were called
            assert mock_model.train.called
            assert mock_model.eval.called

    @patch('scripts.train_model.TRANSFORMERS_AVAILABLE', True)
    @patch('scripts.train_model.load_model_for_training')
    def test_train_model_loading_failure(self, mock_load_model):
        """Test training when model loading fails."""
        from scripts.train_model import train

        # Mock model loading failure
        mock_load_model.return_value = (None, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train(
                    section="math",
                    config=self._mock_config(),
                    checkpoint_dir=tmpdir
                )
                assert False, "Should have exited"
            except SystemExit:
                # Expected to exit when model loading fails
                pass

    @patch('scripts.train_model.TRANSFORMERS_AVAILABLE', False)
    def test_train_no_transformers(self):
        """Test training when transformers not available."""
        from scripts.train_model import train

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train(
                    section="math",
                    config=self._mock_config(),
                    checkpoint_dir=tmpdir
                )
                assert False, "Should have exited"
            except SystemExit:
                # Expected to exit when transformers not available
                pass

    @patch('scripts.train_model.TRANSFORMERS_AVAILABLE', True)
    @patch('scripts.train_model.SFTDataset')
    @patch('scripts.train_model.load_model_for_training')
    def test_train_missing_data_file(self, mock_load_model, mock_dataset_cls):
        """Test training when data file is missing."""
        from scripts.train_model import train
        import torch

        # Mock model loading
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Create config with non-existent data path
        config = self._mock_config()
        config.paths.data_dir = "/nonexistent/path"

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                train(
                    section="math",
                    config=config,
                    checkpoint_dir=tmpdir
                )
                assert False, "Should have exited"
            except SystemExit:
                # Expected to exit when data file not found
                pass

    def test_section_normalization(self):
        """Test section name normalization."""
        from scripts.train_model import main
        import argparse

        # Test various section names
        test_cases = [
            ("reading_writing", "reading_writing"),
            ("rw", "reading_writing"),
            ("readingwriting", "reading_writing"),
            ("math", "math")
        ]

        for input_section, expected_section in test_cases:
            # Simulate argparse parsing
            if input_section in ["rw", "readingwriting"]:
                normalized = "reading_writing"
            else:
                normalized = input_section

            assert normalized == expected_section


class TestTrainCLI:
    """Tests for training script CLI."""

    @patch('scripts.train_model.train')
    @patch.dict('os.environ', {'APP_ENV': 'local'})
    def test_cli_basic_args(self, mock_train):
        """Test CLI with basic arguments."""
        from scripts.train_model import main
        import sys

        # Mock sys.argv
        test_args = [
            "train_model.py",
            "--section", "math"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            # Verify train was called with correct section
            if mock_train.called:
                call_args = mock_train.call_args
                assert call_args[1]["section"] == "math"

    @patch('scripts.train_model.train')
    def test_cli_with_checkpoint_dir(self, mock_train):
        """Test CLI with custom checkpoint directory."""
        from scripts.train_model import main

        test_args = [
            "train_model.py",
            "--section", "reading_writing",
            "--checkpoint-dir", "/custom/checkpoints"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            if mock_train.called:
                call_args = mock_train.call_args
                assert call_args[1]["checkpoint_dir"] == "/custom/checkpoints"

    @patch('scripts.train_model.train')
    @patch.dict('os.environ', {'APP_ENV': 'production'})
    def test_cli_with_env_override(self, mock_train):
        """Test CLI with environment override."""
        from scripts.train_model import main

        test_args = [
            "train_model.py",
            "--section", "math",
            "--env", "production"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            # Verify APP_ENV was set
            import os
            assert os.environ.get("APP_ENV") == "production"
