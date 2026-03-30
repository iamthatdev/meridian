"""Quality rules checker for SAT items."""

import re
from typing import Dict, Any, List


class QualityRulesChecker:
    """Check basic quality rules for SAT items."""

    def check(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check basic quality rules.

        Args:
            item: Item dictionary with content_json

        Returns:
            {"passed": bool, "issues": list[str]}
        """
        issues = []
        content = item.get("content_json", {})

        # Check duplicate choices
        if not self._check_duplicate_choices(content):
            issues.append("Duplicate choice texts found")

        # Check question length
        if not self._check_question_length(content):
            issues.append("Question too short (< 20 characters)")

        # Check rationale length
        if not self._check_rationale_length(content):
            issues.append("Rationale too short (< 60 characters)")

        # Check LaTeX for Math
        latex_check = self._check_math_latex(item, content)
        if not latex_check.get("passed", True):
            issues.append(latex_check.get("error", "LaTeX check failed"))

        return {"passed": len(issues) == 0, "issues": issues}

    def _check_duplicate_choices(self, content: Dict[str, Any]) -> bool:
        """Check for duplicate choice texts."""
        choices = content.get("choices", [])
        if not choices:
            return False
        texts = [choice.get("text", "") for choice in choices]
        return len(texts) == len(set(texts))

    def _check_question_length(self, content: Dict[str, Any]) -> bool:
        """Check if question meets minimum length."""
        question = content.get("question", "")
        return len(question.strip()) >= 20

    def _check_rationale_length(self, content: Dict[str, Any]) -> bool:
        """Check if rationale meets minimum length."""
        rationale = content.get("rationale", "")
        return len(rationale.strip()) >= 60

    def _check_math_latex(self, item: Dict[str, Any], content: Dict[str, Any]) -> Dict[str, Any]:
        """Check if Math item uses LaTeX appropriately."""
        if item.get("section") != "math":
            return {"passed": True}  # Not applicable for non-Math

        question = content.get("question", "")
        math_format = content.get("math_format", "plain")

        # Detect math expressions (variables with exponents or equals)
        has_expressions = bool(re.search(r"[a-z]\s*[\^=]", question))

        if has_expressions and math_format != "latex":
            return {
                "passed": False,
                "error": "Math item has expressions but math_format is not 'latex'"
            }

        return {"passed": True}
