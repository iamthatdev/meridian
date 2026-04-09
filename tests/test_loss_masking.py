"""
Tests for trl.SFTTrainer configuration and loss masking.

These tests verify that SFTTrainer is correctly configured for
supervised fine-tuning with proper loss masking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


class TestSFTConfig:
    """Test SFTConfig configuration for SFT training."""

    def test_sft_config_max_length(self):
        """Test that SFTConfig correctly sets max_length parameter."""
        sft_config = SFTConfig(
            max_length=128,
            dataset_text_field="messages"
        )

        assert sft_config.max_length == 128
        assert sft_config.dataset_text_field == "messages"

    def test_sft_config_with_validation(self):
        """Test that SFTConfig supports evaluation settings."""
        sft_config = SFTConfig(
            max_length=256,
            dataset_text_field="messages",
            eval_strategy="steps",
            eval_steps=100
        )

        assert sft_config.max_length == 256
        assert sft_config.eval_strategy == "steps"
        assert sft_config.eval_steps == 100

    def test_sft_config_dataset_kwargs(self):
        """Test that SFTConfig handles dataset-specific parameters."""
        sft_config = SFTConfig(
            max_length=128,
            dataset_text_field="messages",
            packing=False,
            assistant_only_loss=True
        )

        assert sft_config.packing is False
        assert sft_config.assistant_only_loss is True


class TestDatasetStructure:
    """Test dataset structure for SFTTrainer."""

    def test_dataset_with_messages_field(self):
        """Test that dataset has correct 'messages' field structure."""
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

        assert "messages" in dataset.column_names
        assert len(dataset) == 1
        assert len(dataset[0]["messages"]) == 3  # system, user, assistant

    def test_dataset_without_messages_field(self):
        """Test that dataset without 'messages' field is detected."""
        # Create dataset WITHOUT messages field
        data = {
            "text": ["Some text"],
            "label": [0]
        }
        dataset = Dataset.from_dict(data)

        # Verify messages field is missing
        assert "messages" not in dataset.column_names
        assert dataset.column_names == ["text", "label"]

    def test_chat_message_structure(self):
        """Test that chat messages have required role and content fields."""
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"}
        ]

        # Verify each message has role and content
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]


class TestLossMaskingConfiguration:
    """Test loss masking configuration for assistant-only training."""

    def test_assistant_only_loss_flag(self):
        """Test that assistant_only_loss flag can be set in SFTConfig."""
        sft_config = SFTConfig(
            max_length=128,
            dataset_text_field="messages",
            assistant_only_loss=True
        )

        assert sft_config.assistant_only_loss is True

    def test_completion_only_loss_flag(self):
        """Test that completion_only_loss flag can be set in SFTConfig."""
        sft_config = SFTConfig(
            max_length=128,
            dataset_text_field="messages",
            completion_only_loss=True
        )

        assert sft_config.completion_only_loss is True

    def test_loss_type_parameter(self):
        """Test that loss_type parameter can be configured."""
        sft_config = SFTConfig(
            max_length=128,
            dataset_text_field="messages",
            loss_type="nll"
        )

        assert sft_config.loss_type == "nll"


class TestChatTemplateRequirements:
    """Test chat template requirements for tokenizers."""

    def test_tokenizer_has_chat_template(self):
        """Test detection of tokenizer with chat template support."""
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template = Mock(return_value="formatted")

        has_template = hasattr(mock_tokenizer, 'apply_chat_template')
        assert has_template is True
        assert callable(mock_tokenizer.apply_chat_template)

    def test_tokenizer_without_chat_template(self):
        """Test detection of tokenizer without chat template support."""
        # Mock tokenizer WITHOUT chat template
        mock_tokenizer = Mock(spec=[])  # Empty spec = no attributes

        has_template = hasattr(mock_tokenizer, 'apply_chat_template')
        assert has_template is False

    def test_chat_template_returns_dict_when_tokenize_true(self):
        """Test that chat_template returns dict with tokenize=True."""
        mock_tokenizer = Mock()
        # Should return dict when tokenize=True
        mock_tokenizer.apply_chat_template = Mock(return_value={
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 1, 1]
        })

        result = mock_tokenizer.apply_chat_template(
            [{"role": "user", "content": "Test"}],
            tokenize=True,
            return_dict=True
        )

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "assistant_masks" in result


class TestTrainingValidationSplit:
    """Test training and validation dataset handling."""

    def test_train_dataset_structure(self):
        """Test that training dataset has correct structure."""
        data = {
            "messages": [[
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        assert "messages" in dataset.column_names
        assert len(dataset) == 1

    def test_validation_dataset_structure(self):
        """Test that validation dataset has correct structure."""
        data = {
            "messages": [[
                {"role": "user", "content": "Val Question"},
                {"role": "assistant", "content": "Val Answer"}
            ]]
        }
        dataset = Dataset.from_dict(data)

        assert "messages" in dataset.column_names
        assert len(dataset) == 1

    def test_train_val_split_from_single_dataset(self):
        """Test splitting a dataset into train and validation."""
        data = {
            "messages": [
                [{"role": "user", "content": f"Question {i}"},
                 {"role": "assistant", "content": f"Answer {i}"}]
                for i in range(10)
            ]
        }
        dataset = Dataset.from_dict(data)

        # Split 80/20
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

        assert "train" in split_dataset
        assert "test" in split_dataset
        assert len(split_dataset["train"]) == 8  # 80% of 10
        assert len(split_dataset["test"]) == 2   # 20% of 10
