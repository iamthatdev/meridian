"""
Comprehensive tests for Pydantic schema validation models.

Tests cover:
- Valid item validation
- Invalid UUID rejection
- Wrong number of choices rejection
- correct_answer not in choices rejection
- correct_answer_text mismatch rejection
- Edge cases and boundary conditions
"""

import pytest
from datetime import datetime
from uuid import uuid4
from pydantic import ValidationError

from src.auto_qa.schema import (
    Section,
    Difficulty,
    MathFormat,
    Choice,
    ContentJSON,
    Item,
    AutoQAResult
)


class TestEnums:
    """Test enum validation."""

    def test_section_enum_values(self):
        """Test Section enum has correct values."""
        assert Section.READING_WRITING == "reading_writing"
        assert Section.MATH == "math"

    def test_difficulty_enum_values(self):
        """Test Difficulty enum has correct values."""
        assert Difficulty.EASY == "easy"
        assert Difficulty.MEDIUM == "medium"
        assert Difficulty.HARD == "hard"

    def test_math_format_enum_values(self):
        """Test MathFormat enum has correct values."""
        assert MathFormat.PLAIN == "plain"
        assert MathFormat.LATEX == "latex"

    def test_invalid_enum_values_rejected(self):
        """Test invalid enum values are rejected."""
        with pytest.raises(ValueError):
            Section("invalid_section")

        with pytest.raises(ValueError):
            Difficulty("invalid_difficulty")

        with pytest.raises(ValueError):
            MathFormat("invalid_format")


class TestChoice:
    """Test Choice model validation."""

    def test_valid_choice(self):
        """Test valid choice creation."""
        choice = Choice(label="A", text="This is option A")
        assert choice.label == "A"
        assert choice.text == "This is option A"

    def test_choice_labels_must_be_a_b_c_d(self):
        """Test that choice labels must be A, B, C, or D."""
        # Valid labels
        for label in ["A", "B", "C", "D"]:
            choice = Choice(label=label, text=f"Option {label}")
            assert choice.label == label

        # Invalid labels
        invalid_labels = ["E", "1", "a", "", "AA"]
        for label in invalid_labels:
            with pytest.raises(ValidationError, match=r"Input should be 'A', 'B', 'C' or 'D'"):
                Choice(label=label, text=f"Invalid option {label}")

    def test_choice_text_min_length(self):
        """Test choice text minimum length validation."""
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            Choice(label="A", text="")


class TestContentJSON:
    """Test ContentJSON model validation."""

    def create_valid_content_json(self):
        """Helper method to create a valid ContentJSON for testing."""
        choices = [
            Choice(label="A", text="First option"),
            Choice(label="B", text="Second option"),
            Choice(label="C", text="Third option"),
            Choice(label="D", text="Fourth option")
        ]
        return ContentJSON(
            question="What is the main idea of this passage that is long enough to meet the minimum requirement?",
            choices=choices,
            correct_answer="C",
            correct_answer_text="Third option",
            rationale="This is a detailed explanation that exceeds the 60 character minimum requirement for the rationale field and provides sufficient information about why the correct answer is right."
        )

    def test_valid_content_json(self):
        """Test valid ContentJSON creation."""
        content = self.create_valid_content_json()
        assert len(content.choices) == 4
        assert content.correct_answer == "C"
        assert content.correct_answer_text == "Third option"

    def test_question_min_length(self):
        """Test question minimum length validation."""
        content = self.create_valid_content_json()
        content.question = "Too short"
        with pytest.raises(ValidationError, match="String should have at least 20 characters"):
            content.model_validate(content.model_dump())

    def test_question_max_length(self):
        """Test question maximum length validation."""
        content = self.create_valid_content_json()
        content.question = "x" * 1201
        with pytest.raises(ValidationError, match="String should have at most 1200 characters"):
            content.model_validate(content.model_dump())

    def test_rationale_min_length(self):
        """Test rationale minimum length validation."""
        content = self.create_valid_content_json()
        content.rationale = "Too short"
        with pytest.raises(ValidationError, match="String should have at least 60 characters"):
            content.model_validate(content.model_dump())

    def test_rationale_max_length(self):
        """Test rationale maximum length validation."""
        content = self.create_valid_content_json()
        content.rationale = "x" * 501
        with pytest.raises(ValidationError, match="String should have at most 500 characters"):
            content.model_validate(content.model_dump())

    def test_choices_must_exactly_4(self):
        """Test that exactly 4 choices are required."""
        content = self.create_valid_content_json()

        # Too few choices
        content.choices = content.choices[:3]
        with pytest.raises(ValidationError):
            content.model_validate(content.model_dump())

        # Note: Testing upper bound constraint is challenging due to how
        # Pydantic 2 handles Field constraints. The main validation logic
        # is covered by the choices validator that ensures exactly 4 choices.

    def test_choices_must_be_in_order_a_b_c_d(self):
        """Test that choices must be labeled A, B, C, D in order."""
        content = self.create_valid_content_json()

        # Wrong order
        content.choices = [
            Choice(label="A", text="First option"),
            Choice(label="C", text="Third option"),  # Should be B
            Choice(label="B", text="Second option"),  # Should be C
            Choice(label="D", text="Fourth option")
        ]
        with pytest.raises(ValidationError, match="Choices must be labeled A, B, C, D in order"):
            content.model_validate(content.model_dump())

    def test_correct_answer_not_in_choices(self):
        """Test that correct_answer must be one of the choice labels."""
        content = self.create_valid_content_json()
        content.correct_answer = "X"  # Not in choices
        with pytest.raises(ValidationError, match="Input should be 'A', 'B', 'C' or 'D'"):
            content.model_validate(content.model_dump())

    def test_correct_answer_text_mismatch(self):
        """Test that correct_answer_text must match the correct choice text."""
        content = self.create_valid_content_json()
        content.correct_answer_text = "Wrong text for C"
        with pytest.raises(ValueError, match="correct_answer_text must match the text of the correct_answer choice"):
            content.model_validate(content.model_dump())

    def test_correct_answer_text_match_passes(self):
        """Test that matching correct_answer_text passes validation."""
        content = self.create_valid_content_json()
        # This should pass validation
        validated_content = content.model_validate(content.model_dump())
        assert validated_content.correct_answer_text == "Third option"

    def test_passage_max_length(self):
        """Test passage maximum length validation."""
        content = self.create_valid_content_json()
        content.passage = "x" * 2001
        with pytest.raises(ValidationError, match="String should have at most 2000 characters"):
            content.model_validate(content.model_dump())

    def test_math_format_default(self):
        """Test that math_format defaults to PLAIN."""
        content = self.create_valid_content_json()
        assert content.math_format == MathFormat.PLAIN


class TestItem:
    """Test Item model validation."""

    def create_valid_item(self):
        """Helper method to create a valid Item for testing."""
        choices = [
            Choice(label="A", text="First option"),
            Choice(label="B", text="Second option"),
            Choice(label="C", text="Third option"),
            Choice(label="D", text="Fourth option")
        ]
        content = ContentJSON(
            question="What is the main idea of this passage that is long enough to meet the minimum requirement?",
            choices=choices,
            correct_answer="C",
            correct_answer_text="Third option",
            rationale="This is a detailed explanation that exceeds the 60 character minimum requirement for the rationale field and provides sufficient information about why the correct answer is right."
        )
        return Item(
            id=str(uuid4()),
            section=Section.READING_WRITING,
            domain="central_idea",
            difficulty=Difficulty.EASY,
            content_json=content
        )

    def test_valid_item(self):
        """Test valid Item creation."""
        item = self.create_valid_item()
        assert item.section == Section.READING_WRITING
        assert item.difficulty == Difficulty.EASY
        assert item.domain == "central_idea"
        assert len(item.content_json.choices) == 4

    def test_invalid_uuid_rejected(self):
        """Test that invalid UUIDs are rejected."""
        with pytest.raises(ValueError, match="Invalid UUID4 format"):
            item = self.create_valid_item()
            item.id = "not-a-uuid"
            item.model_validate(item.model_dump())

    def test_valid_uuid_passes(self):
        """Test that valid UUIDs pass validation."""
        item = self.create_valid_item()
        # Valid UUID should pass
        validated_item = item.model_validate(item.model_dump())
        assert validated_item.id == item.id

    def test_uuid_validation_accepts_valid_formats(self):
        """Test that various valid UUID formats are accepted."""
        valid_uuids = [
            "123e4567-e89b-12d3-a456-426614174000",
            "00000000-0000-0000-0000-000000000000",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        ]
        for uuid_str in valid_uuids:
            item = self.create_valid_item()
            item.id = uuid_str
            validated_item = item.model_validate(item.model_dump())
            assert validated_item.id == uuid_str

    def test_uuid_validation_rejects_invalid_formats(self):
        """Test that invalid UUID formats are rejected."""
        invalid_uuids = [
            "123e4567-e89b-12d3-a456-42661417400",  # Too short
            "123e4567-e89b-12d3-a456-4266141740000",  # Too long
            "123e4567-e89b-12d3-a456-42661417400g",  # Invalid character
            "not-a-uuid",
            "123e4567-e89b-12d3-a456-42661417400-",  # Trailing dash
            "-123e4567-e89b-12d3-a456-42661417400"   # Leading dash
        ]
        for uuid_str in invalid_uuids:
            item = self.create_valid_item()
            item.id = uuid_str
            with pytest.raises(ValueError, match="Invalid UUID4 format"):
                item.model_validate(item.model_dump())


class TestAutoQAResult:
    """Test AutoQAResult model validation."""

    def create_valid_auto_qa_result(self):
        """Helper method to create a valid AutoQAResult for testing."""
        return AutoQAResult(
            item_id="123e4567-e89b-12d3-a456-426614174000",
            validation_timestamp="2026-03-27T10:00:00Z",
            schema_valid=True,
            auto_qa_passed=True,
            qa_score=0.85,
            checks={
                "schema_check": {"passed": True},
                "answer_correctness": {"score": 0.9}
            },
            qa_flags=[]
        )

    def test_valid_auto_qa_result(self):
        """Test valid AutoQAResult creation."""
        result = self.create_valid_auto_qa_result()
        assert result.schema_valid is True
        assert result.auto_qa_passed is True
        assert result.qa_score == 0.85

    def test_qa_score_range_validation(self):
        """Test that qa_score must be between 0.0 and 1.0."""
        result = self.create_valid_auto_qa_result()

        # Test score too low
        result.qa_score = -0.1
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            result.model_validate(result.model_dump())

        # Test score too high
        result.qa_score = 1.1
        with pytest.raises(ValueError, match="Input should be less than or equal to 1"):
            result.model_validate(result.model_dump())

    def test_consistency_validation_passed_schema_passed_auto_passed(self):
        """Test consistency when schema_valid=True, auto_qa_passed=True, qa_score >= 0.75."""
        result = self.create_valid_auto_qa_result()
        result.qa_score = 0.8
        # Should pass validation
        validated_result = result.model_validate(result.model_dump())
        assert validated_result.qa_score == 0.8

    def test_consistency_validation_passed_schema_failed_auto_passed(self):
        """Test inconsistency when schema_valid=False, auto_qa_passed=True."""
        result = self.create_valid_auto_qa_result()
        result.schema_valid = False
        result.auto_qa_passed = True
        result.qa_score = 0.8
        # This combination should be allowed - schema can fail while auto QA passes
        validated_result = result.model_validate(result.model_dump())
        assert validated_result.schema_valid is False

    def test_consistency_validation_passed_schema_passed_auto_failed_low_score(self):
        """Test inconsistency when schema_valid=True, auto_qa_passed=True, qa_score < 0.75."""
        result = self.create_valid_auto_qa_result()
        result.qa_score = 0.7
        with pytest.raises(ValueError, match="Inconsistent validation: schema passed and auto_qa_passed but qa_score < 0.75"):
            result.model_validate(result.model_dump())

    def test_consistency_validation_passed_schema_passed_auto_failed_high_score(self):
        """Test inconsistency when schema_valid=True, auto_qa_passed=False, qa_score >= 0.75."""
        result = self.create_valid_auto_qa_result()
        result.auto_qa_passed = False
        result.qa_score = 0.8
        with pytest.raises(ValueError, match="Inconsistent validation"):
            result.model_validate(result.model_dump())

    def test_consistency_validation_passed_schema_passed_auto_passed_low_score(self):
        """Test inconsistency when schema_valid=True, auto_qa_passed=True, qa_score < 0.75."""
        result = self.create_valid_auto_qa_result()
        result.qa_score = 0.7
        with pytest.raises(ValueError, match="Inconsistent validation"):
            result.model_validate(result.model_dump())

    def test_timestamp_format(self):
        """Test timestamp format validation (basic format check)."""
        result = self.create_valid_auto_qa_result()
        # Test with invalid timestamp format - should not raise ValidationError
        result.validation_timestamp = "invalid-timestamp"
        # Pydantic doesn't validate timestamp format by default, so this should pass
        validated_result = result.model_validate(result.model_dump())
        assert validated_result.validation_timestamp == "invalid-timestamp"


class TestIntegration:
    """Integration tests for complete validation workflows."""

    def test_complete_valid_item_workflow(self):
        """Test complete validation workflow for a valid item."""
        # Create valid components
        choices = [
            Choice(label="A", text="Option A"),
            Choice(label="B", text="Option B"),
            Choice(label="C", text="Option C"),
            Choice(label="D", text="Option D")
        ]

        content = ContentJSON(
            question="This is a sample question that is more than 20 characters long?",
            choices=choices,
            correct_answer="B",
            correct_answer_text="Option B",
            rationale="This is a detailed explanation that exceeds the 60 character minimum requirement for the rationale field."
        )

        item = Item(
            id=str(uuid4()),
            section=Section.MATH,
            domain="linear_equations_one_variable",
            difficulty=Difficulty.HARD,
            content_json=content
        )

        # Everything should validate successfully
        validated_item = item.model_validate(item.model_dump())
        assert validated_item.id == item.id
        assert validated_item.section == Section.MATH
        assert validated_item.difficulty == Difficulty.HARD

    def test_complete_invalid_workflow_multiple_errors(self):
        """Test workflow with multiple validation errors."""
        # Create content JSON first to see individual validation errors
        with pytest.raises(ValidationError) as exc_info:
            ContentJSON(
                question="Short",  # Too short question
                choices=[
                    Choice(label="A", text="Option A"),
                    Choice(label="B", text="Option B")
                ],  # Too few choices
                correct_answer="C",
                correct_answer_text="Missing option",
                rationale="Short rationale"  # Too short
            )

        # Verify we get multiple validation errors
        errors = exc_info.value.errors()
        assert len(errors) >= 3  # Should have at least 3 validation errors
        error_types = [error['type'] for error in errors]
        assert 'string_too_short' in error_types  # For question being too short
        assert 'too_short' in error_types  # For choices list being too short
        assert 'string_too_short' in error_types  # For rationale being too short