import os
import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def temp_db(monkeypatch):
    """Temporary database URL for testing."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://test_user:test_pass@localhost:5432/test_meridian")

@pytest.fixture
def sample_item():
    """Sample valid item for testing."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "section": "math",
        "domain": "algebra.quadratic_equations",
        "difficulty": "medium",
        "content_json": {
            "passage": None,
            "question": "If $x^2 - 5x + 6 = 0$, what are the roots?",
            "math_format": "latex",
            "choices": [
                {"label": "A", "text": "$x = 2$ and $x = 3$"},
                {"label": "B", "text": "$x = -2$ and $x = -3$"},
                {"label": "C", "text": "$x = 2$ and $x = -3$"},
                {"label": "D", "text": "$x = -2$ and $x = 3$"}
            ],
            "correct_answer": "A",
            "correct_answer_text": "$x = 2$ and $x = 3$",
            "rationale": "Factoring gives $(x-2)(x-3)=0$, so $x=2$ or $x=3$.",
            "solution_steps": None
        }
    }