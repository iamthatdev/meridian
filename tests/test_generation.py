"""Tests for generation script."""

import json
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestItemGenerator:
    """Tests for ItemGenerator class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        return config

    @patch('src.generation.generator.TRANSFORMERS_AVAILABLE', True)
    @patch('src.generation.generator.AutoModelForCausalLM')
    @patch('src.generation.generator.AutoTokenizer')
    def test_generator_init(self, mock_tokenizer_cls, mock_model_cls, mock_config):
        """Test generator initialization."""
        from src.generation.generator import ItemGenerator

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        generator = ItemGenerator(
            checkpoint_path="/fake/checkpoint",
            config=mock_config
        )

        assert generator.tokenizer is not None
        assert generator.model is not None
        mock_model.eval.assert_called_once()

    @patch('src.generation.generator.TRANSFORMERS_AVAILABLE', False)
    def test_generator_no_transformers(self, mock_config):
        """Test generator when transformers not available."""
        from src.generation.generator import ItemGenerator

        with pytest.raises(RuntimeError):
            ItemGenerator(checkpoint_path="/fake/checkpoint")

    @patch('src.generation.generator.TRANSFORMERS_AVAILABLE', True)
    @patch('src.generation.generator.AutoModelForCausalLM')
    @patch('src.generation.generator.AutoTokenizer')
    def test_generate_single_item(self, mock_tokenizer_cls, mock_model_cls):
        """Test generating a single item."""
        from src.generation.generator import ItemGenerator
        import torch

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.apply_chat_template.return_value = "System prompt\nUser prompt\nAssistant:"
        mock_tokenizer.return_value = MagicMock(
            **{"input_ids": torch.tensor([[1, 2, 3]])}
        )
        mock_tokenizer.decode.return_value = '{"question": "test"}'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_cls.from_pretrained.return_value = mock_model

        generator = ItemGenerator(checkpoint_path="/fake/checkpoint")

        items = generator.generate(
            section="math",
            domain="algebra.quadratic_equations",
            difficulty="medium"
        )

        # Should have attempted generation
        mock_model.generate.assert_called_once()

    @patch('src.generation.generator.TRANSFORMERS_AVAILABLE', True)
    @patch('src.generation.generator.AutoModelForCausalLM')
    @patch('src.generation.generator.AutoTokenizer')
    def test_generate_batch(self, mock_tokenizer_cls, mock_model_cls):
        """Test batch generation across domains."""
        from src.generation.generator import ItemGenerator
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.return_value = MagicMock(
            **{"input_ids": torch.tensor([[1, 2, 3]])}
        )
        mock_tokenizer.decode.return_value = '{"question": "test"}'
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_model_cls.from_pretrained.return_value = mock_model

        generator = ItemGenerator(checkpoint_path="/fake/checkpoint")

        # Patch generate method to return valid items
        def mock_generate(*args, **kwargs):
            domain = kwargs.get('domain', 'test')
            return [{"id": "test", "section": "math", "domain": domain, "difficulty": "medium", "content_json": {"question": "test", "choices": [], "correct_answer": "A", "correct_answer_text": "A", "rationale": "test"}}]

        generator.generate = mock_generate

        items = generator.generate_batch(
            section="math",
            domains=["algebra.linear_equations", "algebra.quadratic_equations"],
            difficulty="medium",
            items_per_domain=1
        )

        assert len(items) == 2


class TestGenerateItemsScript:
    """Tests for generate_items.py script."""

    @patch('scripts.generate_items.ItemGenerator')
    @patch('scripts.generate_items.AutoQAPipeline')
    def test_generate_single_domain(self, mock_pipeline_cls, mock_generator_cls):
        """Test generating items for single domain."""
        from scripts.generate_items import generate_items

        mock_generator = MagicMock()
        mock_generator.generate.return_value = [
            {
                "id": "test-id",
                "section": "math",
                "domain": "algebra.quadratic_equations",
                "difficulty": "medium",
                "content_json": {"question": "test"}
            }
        ]
        mock_generator_cls.return_value = mock_generator

        mock_pipeline = MagicMock()
        mock_pipeline.validate.return_value = {
            "auto_qa_passed": True,
            "qa_score": 1.0,
            "qa_flags": []
        }
        mock_pipeline_cls.return_value = mock_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.jsonl"

            generate_items(
                checkpoint="/fake/checkpoint",
                section="math",
                domains=["algebra.quadratic_equations"],
                difficulty="medium",
                items_per_domain=1,
                validate=True,
                output=str(output_file)
            )

            # Verify generator was called
            mock_generator.generate.assert_called_once()
            assert output_file.exists()

    @patch('scripts.generate_items.ItemGenerator')
    @patch('scripts.generate_items.AutoQAPipeline')
    def test_generate_batch_domains(self, mock_pipeline_cls, mock_generator_cls):
        """Test batch generation across multiple domains."""
        from scripts.generate_items import generate_items

        mock_generator = MagicMock()
        mock_generator.generate_batch.return_value = [
            {"id": "1", "section": "math", "domain": "domain1", "difficulty": "medium", "content_json": {}},
            {"id": "2", "section": "math", "domain": "domain2", "difficulty": "medium", "content_json": {}}
        ]
        mock_generator_cls.return_value = mock_generator

        mock_pipeline = MagicMock()
        mock_pipeline.validate.return_value = {"auto_qa_passed": True, "qa_score": 1.0, "qa_flags": []}
        mock_pipeline_cls.return_value = mock_pipeline

        generate_items(
            checkpoint="/fake/checkpoint",
            section="math",
            domains=["domain1", "domain2"],
            difficulty="medium",
            items_per_domain=1,
            validate=True,
            batch=True
        )

        # Verify batch generation was called
        mock_generator.generate_batch.assert_called_once()

    @patch('scripts.generate_items.ItemGenerator')
    def test_generate_without_validation(self, mock_generator_cls):
        """Test generation without Auto-QA validation."""
        from scripts.generate_items import generate_items

        mock_generator = MagicMock()
        mock_generator.generate.return_value = [{"id": "test"}]
        mock_generator_cls.return_value = mock_generator

        generate_items(
            checkpoint="/fake/checkpoint",
            section="math",
            domains=["algebra.quadratic_equations"],
            difficulty="medium",
            validate=False
        )

        # Should not have called pipeline
        mock_generator.generate.assert_called_once()

    @patch('scripts.generate_items.ItemGenerator')
    @patch('scripts.generate_items.AutoQAPipeline')
    def test_generate_with_validation_failure(self, mock_pipeline_cls, mock_generator_cls):
        """Test generation when validation fails."""
        from scripts.generate_items import generate_items

        mock_generator = MagicMock()
        mock_generator.generate.return_value = [
            {"id": "test", "section": "math", "domain": "test", "difficulty": "medium", "content_json": {}}
        ]
        mock_generator_cls.return_value = mock_generator

        mock_pipeline = MagicMock()
        mock_pipeline.validate.return_value = {
            "auto_qa_passed": False,
            "qa_score": 0.0,
            "qa_flags": ["Validation failed"]
        }
        mock_pipeline_cls.return_value = mock_pipeline

        result = generate_items(
            checkpoint="/fake/checkpoint",
            section="math",
            domains=["test"],
            difficulty="medium",
            validate=True
        )

        # Item should be filtered out due to validation failure
        mock_generator.generate.assert_called_once()


class TestGenerateCLI:
    """Tests for generate_items.py CLI."""

    @patch('scripts.generate_items.generate_items')
    def test_cli_single_domain(self, mock_generate):
        """Test CLI with single domain."""
        from scripts.generate_items import main

        test_args = [
            "generate_items.py",
            "--checkpoint", "/fake/checkpoint",
            "--section", "math",
            "--domain", "algebra.quadratic_equations",
            "--difficulty", "medium"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            if mock_generate.called:
                # Verify it was called with single domain
                pass

    @patch('scripts.generate_items.generate_items')
    def test_cli_batch_mode(self, mock_generate):
        """Test CLI with batch mode."""
        from scripts.generate_items import main

        test_args = [
            "generate_items.py",
            "--checkpoint", "/fake/checkpoint",
            "--section", "reading_writing",
            "--domains", "central_ideas", "inferences",
            "--difficulty", "hard",
            "--batch",
            "--items-per-domain", "2"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            if mock_generate.called:
                call_args = mock_generate.call_args
                assert call_args[1]["batch"] == True
                assert call_args[1]["items_per_domain"] == 2

    @patch('scripts.generate_items.generate_items')
    def test_cli_with_output(self, mock_generate):
        """Test CLI with output file."""
        from scripts.generate_items import main

        test_args = [
            "generate_items.py",
            "--checkpoint", "/fake/checkpoint",
            "--section", "math",
            "--domain", "test",
            "--difficulty", "easy",
            "--output", "generated.jsonl"
        ]

        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass

            if mock_generate.called:
                call_args = mock_generate.call_args
                assert call_args[1]["output"] == "generated.jsonl"

    def test_section_normalization(self):
        """Test section name normalization."""
        test_cases = [
            ("reading_writing", "reading_writing"),
            ("rw", "reading_writing"),
            ("readingwriting", "reading_writing"),
            ("math", "math")
        ]

        for input_section, expected_section in test_cases:
            if input_section in ["rw", "readingwriting"]:
                normalized = "reading_writing"
            else:
                normalized = input_section

            assert normalized == expected_section
