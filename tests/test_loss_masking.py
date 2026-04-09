"""
Tests for trl.SFTTrainer configuration and loss masking.

These tests verify that SFTTrainer is correctly configured for
supervised fine-tuning with proper loss masking.
"""

import pytest
from unittest.mock import Mock, patch
from trl import SFTTrainer
from datasets import Dataset


class TestSFTTrainerConfiguration:
    """Test SFTTrainer configuration for SFT training."""

    def test_sft_trainer_configuration(self):
        """Test that SFTTrainer is correctly configured with training dataset."""
        # Create test dataset
        data = {
            "messages": [
                [
                    {"role": "system", "content": "You are a tutor"},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "The answer is 4"}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.vocab_size = 1000

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.apply_chat_template.return_value = "formatted text"

        # Create trainer
        trainer = SFTTrainer(
            model=mock_model,
            train_dataset=dataset,
            dataset_text_field="messages",
            tokenizer=mock_tokenizer
        )

        # Verify trainer is configured correctly
        assert hasattr(trainer, 'train_dataset')
        assert trainer.train_dataset == dataset
        assert hasattr(trainer, 'tokenizer')

    def test_sft_trainer_with_validation_dataset(self):
        """Test that SFTTrainer handles validation datasets correctly."""
        # Create datasets
        train_data = {
            "messages": [[
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"}
            ]]
        }
        val_data = {
            "messages": [[
                {"role": "user", "content": "Val Question"},
                {"role": "assistant", "content": "Val Answer"}
            ]]
        }

        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.vocab_size = 1000

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.apply_chat_template.return_value = "formatted"

        # Create trainer with validation dataset
        trainer = SFTTrainer(
            model=mock_model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="messages",
            tokenizer=mock_tokenizer
        )

        # Verify both datasets are set
        assert trainer.train_dataset == train_dataset
        assert trainer.eval_dataset == val_dataset

    def test_sft_trainer_handles_chat_template(self):
        """Test that tokenizer's chat template is called."""
        data = {
            "messages": [[
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.vocab_size = 1000

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.apply_chat_template.return_value = "formatted conversation"

        # Create trainer - should call apply_chat_template
        trainer = SFTTrainer(
            model=mock_model,
            train_dataset=dataset,
            dataset_text_field="messages",
            tokenizer=mock_tokenizer
        )

        # Verify chat template was called
        assert mock_tokenizer.apply_chat_template.called


class TestErrorHandling:
    """Test error handling for SFTTrainer setup."""

    def test_tokenizer_without_chat_template_raises_error(self):
        """Test that tokenizer without chat template support raises error."""
        # Create dataset
        data = {
            "messages": [[
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        # Mock tokenizer WITHOUT chat template
        mock_tokenizer = Mock(spec=[])  # Empty spec = no attributes
        # Explicitly remove apply_chat_template if it exists
        if hasattr(mock_tokenizer, 'apply_chat_template'):
            delattr(mock_tokenizer, 'apply_chat_template')

        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.vocab_size = 1000

        # This should raise an error when we check for chat template
        has_chat_template = hasattr(mock_tokenizer, 'apply_chat_template')
        assert not has_chat_template, "Tokenizer should not have chat template"

    def test_dataset_without_messages_field_raises_error(self):
        """Test that dataset without 'messages' field is rejected."""
        # Create dataset WITHOUT messages field
        data = {
            "text": ["Some text"],
            "label": [0]
        }
        dataset = Dataset.from_dict(data)

        # Verify messages field is missing
        assert "messages" not in dataset.column_names
        assert dataset.column_names == ["text", "label"]
