"""Tests for Auto-QA validation pipeline."""

import pytest
from src.auto_qa.pipeline import AutoQAPipeline


@pytest.fixture
def sample_valid_item():
    """Sample valid item for testing."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "section": "math",
        "domain": "algebra.quadratic_equations",
        "difficulty": "medium",
        "content_json": {
            "passage": None,
            "question": "If x squared minus 5x plus 6 equals 0, what are the roots? " * 2,
            "math_format": "latex",
            "choices": [
                {"label": "A", "text": "2 and 3"},
                {"label": "B", "text": "-2 and -3"},
                {"label": "C", "text": "2 and -3"},
                {"label": "D", "text": "-2 and 3"}
            ],
            "correct_answer": "A",
            "correct_answer_text": "2 and 3",
            "rationale": "Factoring the quadratic equation gives us x minus 2 times x minus 3 equals 0. Setting each factor to zero gives x equals 2 or x equals 3. Therefore the roots are 2 and 3. This is a complete solution showing all steps clearly."
        }
    }


@pytest.fixture
def sample_invalid_schema_item():
    """Item with invalid schema (wrong UUID)."""
    return {
        "id": "not-a-uuid",
        "section": "math",
        "domain": "algebra",
        "difficulty": "medium",
        "content_json": {
            "passage": None,
            "question": "Too short",
            "math_format": "plain",
            "choices": [
                {"label": "A", "text": "A"},
                {"label": "B", "text": "B"}
            ],
            "correct_answer": "A",
            "correct_answer_text": "A",
            "rationale": "Test"
        }
    }


@pytest.fixture
def sample_quality_fail_item():
    """Item that fails quality rules (duplicate choices)."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440001",
        "section": "math",
        "domain": "algebra",
        "difficulty": "medium",
        "content_json": {
            "passage": None,
            "question": "What are the roots of the quadratic equation?",  # Passes schema length
            "math_format": "latex",
            "choices": [
                {"label": "A", "text": "2 and 3"},
                {"label": "B", "text": "2 and 3"},  # Duplicate - will fail quality check
                {"label": "C", "text": "2 and -3"},
                {"label": "D", "text": "-2 and 3"}
            ],
            "correct_answer": "A",
            "correct_answer_text": "2 and 3",
            "rationale": "This is the correct answer because factoring gives (x-2)(x-3)=0."
        }
    }


def test_auto_qa_pass_valid_item(sample_valid_item):
    """Test that valid item passes Auto-QA."""
    pipeline = AutoQAPipeline()
    result = pipeline.validate(sample_valid_item)

    assert result["auto_qa_passed"] is True
    assert result["schema_valid"] is True
    assert result["qa_score"] == 1.0
    assert len(result["qa_flags"]) == 0
    assert result["checks"]["schema_validation"]["passed"] is True
    assert result["checks"]["quality_rules"]["passed"] is True


def test_auto_qa_fail_invalid_schema(sample_invalid_schema_item):
    """Test that invalid schema fails immediately."""
    pipeline = AutoQAPipeline()
    result = pipeline.validate(sample_invalid_schema_item)

    assert result["auto_qa_passed"] is False
    assert result["schema_valid"] is False
    assert len(result["qa_flags"]) > 0
    # Should not run quality checks if schema fails
    assert "quality_rules" not in result["checks"]


def test_auto_qa_fail_quality_rules(sample_quality_fail_item):
    """Test that quality rule failures block validation."""
    pipeline = AutoQAPipeline()
    result = pipeline.validate(sample_quality_fail_item)

    assert result["auto_qa_passed"] is False
    assert result["schema_valid"] is True  # Schema passes
    assert result["checks"]["quality_rules"]["passed"] is False
    assert "Duplicate choice texts found" in result["qa_flags"]


def test_auto_qa_readability_warning():
    """Test that readability issues are warnings, not hard gates."""
    pipeline = AutoQAPipeline()

    item = {
        "id": "550e8400-e29b-41d4-a716-446655440002",
        "section": "reading_writing",
        "domain": "central_idea",
        "difficulty": "medium",
        "content_json": {
            "passage": "This is a very simple passage with short words.",  # Low grade level
            "question": "What is the main idea of this passage?" * 2,
            "math_format": "plain",
            "choices": [
                {"label": "A", "text": "Option A"},
                {"label": "B", "text": "Option B"},
                {"label": "C", "text": "Option C"},
                {"label": "D", "text": "Option D"}
            ],
            "correct_answer": "A",
            "correct_answer_text": "Option A",
            "rationale": "Option A is correct because it captures the main idea. Option B focuses on details. Option C is unrelated. Option D is too broad."
        }
    }

    result = pipeline.validate(item)

    # Readability issues add flags but don't block
    if not result["checks"]["readability"].get("passed", True):
        assert result["schema_valid"] is True
        # Check that qa_flags include readability issues
        assert any("grade" in flag.lower() for flag in result["qa_flags"])


def test_auto_qa_no_passage():
    """Test that items without passages pass (Math items)."""
    pipeline = AutoQAPipeline()

    item = {
        "id": "550e8400-e29b-41d4-a716-446655440003",
        "section": "math",
        "domain": "algebra",
        "difficulty": "easy",
        "content_json": {
            "passage": None,  # No passage is OK for Math
            "question": "Solve for x: 2x + 4 = 10",
            "math_format": "plain",
            "choices": [
                {"label": "A", "text": "x = 2"},
                {"label": "B", "text": "x = 3"},
                {"label": "C", "text": "x = 4"},
                {"label": "D", "text": "x = 5"}
            ],
            "correct_answer": "B",
            "correct_answer_text": "x = 3",
            "rationale": "To solve 2x + 4 = 10, first subtract 4 from both sides to get 2x = 6. Then divide both sides by 2 to get x = 3. This is the correct solution."
        }
    }

    result = pipeline.validate(item)

    assert result["auto_qa_passed"] is True
    assert result["checks"]["readability"].get("note") == "No passage to check"
