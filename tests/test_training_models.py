"""Tests for training models module."""

import pytest
from unittest.mock import MagicMock, patch


class TestLoadTokenizer:
    """Tests for load_tokenizer function."""

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', True)
    @patch('src.training.models.AutoTokenizer')
    def test_load_tokenizer_success(self, mock_tokenizer_cls):
        """Test successful tokenizer loading."""
        from src.training.models import load_tokenizer

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        result = load_tokenizer("Qwen/Qwen2.5-7B-Instruct")

        assert result is not None
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        assert result.pad_token == "</s>"

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', False)
    def test_load_tokenizer_no_transformers(self):
        """Test tokenizer loading when transformers not available."""
        from src.training.models import load_tokenizer

        result = load_tokenizer("Qwen/Qwen2.5-7B-Instruct")

        assert result is None


class TestLoadModel:
    """Tests for load_model function."""

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', True)
    @patch('src.training.models.AutoModelForCausalLM')
    @patch('src.training.models.prepare_model_for_kbit_training')
    def test_load_model_with_4bit(self, mock_prepare, mock_model_cls):
        """Test loading model with 4-bit quantization."""
        from src.training.models import load_model
        from src.config import load_config

        config = load_config()

        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 7000000000
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_prepare.return_value = mock_model

        result = load_model(
            "Qwen/Qwen2.5-7B-Instruct",
            config=config,
            use_4bit=True
        )

        assert result is not None
        mock_model_cls.from_pretrained.assert_called_once()
        mock_prepare.assert_called_once_with(mock_model)

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', True)
    @patch('src.training.models.AutoModelForCausalLM')
    def test_load_model_fp16(self, mock_model_cls):
        """Test loading model with fp16."""
        from src.training.models import load_model
        from src.config import load_config

        config = load_config()

        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 7000000000
        mock_model_cls.from_pretrained.return_value = mock_model

        result = load_model(
            "Qwen/Qwen2.5-7B-Instruct",
            config=config,
            use_4bit=False
        )

        assert result is not None
        # Check that torch_dtype was set
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "torch_dtype" in call_kwargs

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', False)
    def test_load_model_no_transformers(self):
        """Test model loading when transformers not available."""
        from src.training.models import load_model

        result = load_model("Qwen/Qwen2.5-7B-Instruct")

        assert result is None


class TestApplyLora:
    """Tests for apply_lora function."""

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', True)
    @patch('src.training.models.get_peft_model')
    @patch('src.training.models.LoraConfig')
    def test_apply_lora_success(self, mock_lora_config_cls, mock_get_peft):
        """Test successful LoRA application."""
        from src.training.models import apply_lora
        from src.config import load_config

        config = load_config()

        mock_model = MagicMock()
        mock_model.config.model_type = "qwen"
        mock_model.num_parameters.return_value = 7000000000

        mock_peft_model = MagicMock()
        mock_peft_model.num_parameters.return_value = 7000000000
        mock_get_peft.return_value = mock_peft_model

        mock_lora_config = MagicMock()
        mock_lora_config_cls.return_value = mock_lora_config

        result = apply_lora(mock_model, config)

        assert result is not None
        mock_lora_config_cls.assert_called_once()
        mock_get_peft.assert_called_once_with(mock_model, mock_lora_config)

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', True)
    @patch('src.training.models.get_peft_model')
    @patch('src.training.models.LoraConfig')
    def test_apply_lora_custom_targets(self, mock_lora_config_cls, mock_get_peft):
        """Test LoRA with custom target modules."""
        from src.training.models import apply_lora
        from src.config import load_config

        config = load_config()

        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 7000000000
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model

        custom_targets = ["q_proj", "v_proj"]

        result = apply_lora(mock_model, config, target_modules=custom_targets)

        assert result is not None

        # Check that custom targets were used
        call_kwargs = mock_lora_config_cls.call_args[1]
        assert call_kwargs["target_modules"] == custom_targets

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', False)
    def test_apply_lora_no_transformers(self):
        """Test LoRA when transformers not available."""
        from src.training.models import apply_lora

        result = apply_lora(MagicMock())

        assert result is None


class TestLoadModelForTraining:
    """Tests for load_model_for_training function."""

    @patch('src.training.models.apply_lora')
    @patch('src.training.models.load_model')
    @patch('src.training.models.load_tokenizer')
    def test_load_rw_model(self, mock_load_tok, mock_load_model, mock_apply_lora):
        """Test loading RW model."""
        from src.training.models import load_model_for_training

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load_tok.return_value = mock_tokenizer
        mock_load_model.return_value = mock_model
        mock_apply_lora.return_value = mock_model

        model, tokenizer = load_model_for_training("reading_writing")

        assert model is not None
        assert tokenizer is not None
        mock_load_tok.assert_called_once()
        mock_load_model.assert_called_once()
        mock_apply_lora.assert_called_once()

    @patch('src.training.models.apply_lora')
    @patch('src.training.models.load_model')
    @patch('src.training.models.load_tokenizer')
    def test_load_math_model(self, mock_load_tok, mock_load_model, mock_apply_lora):
        """Test loading math model."""
        from src.training.models import load_model_for_training

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load_tok.return_value = mock_tokenizer
        mock_load_model.return_value = mock_model
        mock_apply_lora.return_value = mock_model

        model, tokenizer = load_model_for_training("math")

        assert model is not None
        assert tokenizer is not None

    @patch('src.training.models.apply_lora')
    @patch('src.training.models.load_model')
    @patch('src.training.models.load_tokenizer')
    def test_load_model_invalid_section(self, mock_load_tok, mock_load_model, mock_apply_lora):
        """Test loading model with invalid section."""
        from src.training.models import load_model_for_training

        model, tokenizer = load_model_for_training("invalid_section")

        assert model is None
        assert tokenizer is None
        mock_load_tok.assert_not_called()
        mock_load_model.assert_not_called()

    @patch('src.training.models.apply_lora')
    @patch('src.training.models.load_model')
    @patch('src.training.models.load_tokenizer')
    def test_load_model_tokenizer_error(self, mock_load_tok, mock_load_model, mock_apply_lora):
        """Test error handling when tokenizer fails to load."""
        from src.training.models import load_model_for_training

        mock_load_tok.return_value = None

        model, tokenizer = load_model_for_training("math")

        assert model is None
        assert tokenizer is None
        mock_load_model.assert_not_called()


class TestPrintModelSummary:
    """Tests for print_model_summary function."""

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', True)
    def test_print_model_summary(self, caplog):
        """Test printing model summary."""
        from src.training.models import print_model_summary

        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 7000000000

        # Capture log output
        import logging
        import sys

        print_model_summary(mock_model)

        # If we get here without exception, test passes


class TestSaveModel:
    """Tests for save_model function."""

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', True)
    def test_save_model_success(self, tmp_path):
        """Test saving model and tokenizer."""
        from src.training.models import save_model

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        output_dir = str(tmp_path / "test_output")

        save_model(mock_model, mock_tokenizer, output_dir)

        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    @patch('src.training.models.TRANSFORMERS_AVAILABLE', False)
    def test_save_model_no_transformers(self, tmp_path):
        """Test saving model when transformers not available."""
        from src.training.models import save_model

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        output_dir = str(tmp_path / "test_output")

        # Should not raise exception, just log error
        save_model(mock_model, mock_tokenizer, output_dir)

        mock_model.save_pretrained.assert_not_called()
        mock_tokenizer.save_pretrained.assert_not_called()
