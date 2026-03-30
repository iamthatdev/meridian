# tests/test_converters.py
import pytest
from src.training.converters import convert_item_to_messages

def test_convert_math_item():
    item = {
        "section": "math",
        "domain": "algebra.linear_equations_one_variable",
        "difficulty": "easy",
        "content_json": {
            "question": "Solve 2x + 5 = 15",
            "choices": [
                {"label": "A", "text": "x = 3"},
                {"label": "B", "text": "x = 4"},
                {"label": "C", "text": "x = 5"},
                {"label": "D", "text": "x = 6"}
            ],
            "correct_answer": "C",
            "correct_answer_text": "x = 5",
            "rationale": "Subtract 5, divide by 2",
            "math_format": "latex"
        }
    }

    messages = convert_item_to_messages(item)

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert "Math" in messages[1]["content"]
    assert "algebra.linear_equations_one_variable" in messages[1]["content"]
    assert "easy" in messages[1]["content"]

def test_convert_rw_item():
    item = {
        "section": "reading_writing",
        "domain": "standard_english_conventions.boundaries",
        "difficulty": "medium",
        "content_json": {
            "question": "Choose the correct punctuation",
            "choices": [
                {"label": "A", "text": "Period"},
                {"label": "B", "text": "Comma"},
                {"label": "C", "text": "Semicolon"},
                {"label": "D", "text": "Colon"}
            ],
            "correct_answer": "C",
            "correct_answer_text": "Semicolon",
            "rationale": "Separates independent clauses"
        }
    }

    messages = convert_item_to_messages(item)

    assert messages[0]["role"] == "system"
    assert "Reading & Writing" in messages[1]["content"]
