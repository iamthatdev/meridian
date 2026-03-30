"""Schema validator for SAT items using Pydantic models."""

from pydantic import ValidationError
from src.auto_qa.schema import Item


class SchemaValidator:
    """Validate items against Pydantic schema."""

    def validate(self, item: dict) -> dict:
        """
        Validate item against schema.

        Args:
            item: Dictionary representing a SAT item

        Returns:
            {"passed": bool, "issues": list[str]}
        """
        try:
            Item(**item)
            return {"passed": True, "issues": []}
        except ValidationError as e:
            issues = [err["msg"] for err in e.errors()]
            return {"passed": False, "issues": issues}
