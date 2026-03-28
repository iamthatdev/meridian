"""
Pydantic v2 models for SAT item validation with strict schema enforcement.

This module provides models for validating SAT items with strict type checking
and business rule validation.
"""

import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class Section(str, Enum):
    """Enum for SAT test sections."""
    READING_WRITING = "reading_writing"
    MATH = "math"


class Difficulty(str, Enum):
    """Enum for SAT difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MathFormat(str, Enum):
    """Enum for math content formatting."""
    PLAIN = "plain"
    LATEX = "latex"


class Choice(BaseModel):
    """Represents a single choice in a multiple-choice question."""

    label: Literal["A", "B", "C", "D"] = Field(
        description="The choice label (must be A, B, C, or D)"
    )
    text: str = Field(
        min_length=1,
        description="The choice text content"
    )

    # The Literal type already handles validation for label field


class ContentJSON(BaseModel):
    """Represents the content structure of a SAT item."""

    passage: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional passage text for reading items"
    )
    question: str = Field(
        min_length=20,
        max_length=1200,
        description="The question stem text"
    )
    math_format: MathFormat = Field(
        default=MathFormat.PLAIN,
        description="Format for math content"
    )
    choices: List[Choice] = Field(
        min_length=4,
        max_length=4,
        description="Exactly 4 choices with labels A, B, C, D"
    )
    correct_answer: Literal["A", "B", "C", "D"] = Field(
        description="The correct choice label"
    )
    correct_answer_text: str = Field(
        min_length=1,
        description="The text of the correct answer"
    )
    rationale: str = Field(
        min_length=60,
        max_length=500,
        description="Explanation for the correct answer"
    )
    solution_steps: Optional[str] = Field(
        default=None,
        description="Optional step-by-step solution"
    )

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, choices: List[Choice]) -> List[Choice]:
        """Validate that there are exactly 4 choices with labels A, B, C, D in order."""
        if len(choices) != 4:
            raise ValueError(f"Exactly 4 choices required, got {len(choices)}")

        # Check that labels are A, B, C, D in order
        expected_labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices):
            if choice.label != expected_labels[i]:
                raise ValueError(
                    f"Choices must be labeled A, B, C, D in order. "
                    f"Found {choice.label} at position {i}, expected {expected_labels[i]}"
                )

        return choices

    @field_validator('correct_answer')
    @classmethod
    def validate_correct_answer(cls, v, info):
        """Validate that correct_answer is one of the choice labels."""
        choices = info.data.get('choices', [])
        if choices and v not in [choice.label for choice in choices]:
            raise ValueError(f"correct_answer {v} must be one of the choice labels")
        return v

    @model_validator(mode='after')
    def validate_correct_answer_text(self) -> 'ContentJSON':
        """Validate that correct_answer_text matches the text of correct_answer."""
        if self.choices:
            correct_choice = next(
                (choice for choice in self.choices if choice.label == self.correct_answer),
                None
            )
            if correct_choice and self.correct_answer_text != correct_choice.text:
                raise ValueError(
                    "correct_answer_text must match the text of the correct_answer choice"
                )
        return self


class Item(BaseModel):
    """Represents a complete SAT item for validation."""

    id: str = Field(
        description="Unique identifier for the item"
    )
    section: Section = Field(description="SAT test section")
    domain: str = Field(description="Domain within the section")
    difficulty: Difficulty = Field(description="Difficulty level")
    content_json: ContentJSON = Field(description="Item content structure")
    model_version: Optional[str] = Field(
        default=None,
        description="Optional model version identifier"
    )

    @field_validator('id')
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        """Validate that id is a valid UUID4 string."""
        try:
            uuid.UUID(v, version=4)
            return v
        except ValueError:
            raise ValueError(f"Invalid UUID4 format: {v}")


class AutoQAResult(BaseModel):
    """Represents the results of automated QA validation."""

    item_id: str = Field(description="ID of the item that was validated")
    validation_timestamp: str = Field(description="Timestamp of validation")
    schema_valid: bool = Field(description="Whether schema validation passed")
    auto_qa_passed: bool = Field(description="Whether auto QA checks passed")
    qa_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall QA score (0.0 to 1.0)"
    )
    checks: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed results of individual validation checks"
    )
    qa_flags: List[str] = Field(
        default_factory=list,
        description="List of validation flags or issues"
    )

    @model_validator(mode='after')
    def validate_consistency(self) -> 'AutoQAResult':
        """Ensure consistency between schema_valid, auto_qa_passed, and qa_score."""
        if self.schema_valid and self.auto_qa_passed and self.qa_score < 0.75:
            raise ValueError(
                "Inconsistent validation: schema passed and auto_qa_passed "
                "but qa_score < 0.75"
            )
        if self.schema_valid and not self.auto_qa_passed and self.qa_score >= 0.75:
            raise ValueError(
                "Inconsistent validation: schema passed but auto_qa failed with high score"
            )
        return self