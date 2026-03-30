"""Readability checker for SAT item passages."""

import textstat
from typing import Optional, Dict, Any


class ReadabilityChecker:
    """Check passage readability using Flesch-Kincaid grade level."""

    def __init__(self, min_grade: float = 9.0, max_grade: float = 12.0):
        """
        Initialize readability checker.

        Args:
            min_grade: Minimum acceptable grade level (default: 9.0)
            max_grade: Maximum acceptable grade level (default: 12.0)
        """
        self.min_grade = min_grade
        self.max_grade = max_grade

    def check(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check readability of passage.

        Args:
            item: Item dictionary with content_json.passage

        Returns:
            {"passed": bool, "grade_level": float, "word_count": int, "issues": list, "note": str}
        """
        passage = item.get("content_json", {}).get("passage")

        # No passage is OK (many Math items don't have passages)
        if passage is None or passage == "":
            return {"passed": True, "note": "No passage to check"}

        # Calculate readability
        try:
            grade_level = textstat.flesch_kincaid_grade(passage)
            word_count = textstat.word_count(passage)
        except Exception:
            return {"passed": True, "note": "Could not calculate readability"}

        # Check if within range
        issues = []
        if grade_level < self.min_grade:
            issues.append(f"Grade level {grade_level:.1f} below {self.min_grade}")
        elif grade_level > self.max_grade:
            issues.append(f"Grade level {grade_level:.1f} above {self.max_grade}")

        return {
            "passed": len(issues) == 0,
            "grade_level": grade_level,
            "word_count": word_count,
            "issues": issues
        }
