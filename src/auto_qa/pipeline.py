"""Main Auto-QA validation pipeline."""

import uuid
from datetime import datetime
from loguru import logger
from src.auto_qa.validators.schema_validator import SchemaValidator
from src.auto_qa.validators.readability_checker import ReadabilityChecker
from src.auto_qa.validators.quality_rules import QualityRulesChecker


class AutoQAPipeline:
    """Main Auto-QA validation pipeline for SAT items."""

    def __init__(self):
        """Initialize pipeline with all three validators."""
        self.schema_validator = SchemaValidator()
        self.readability_checker = ReadabilityChecker()
        self.quality_rules_checker = QualityRulesChecker()

    def validate(self, item: dict) -> dict:
        """
        Run full Auto-QA pipeline on item.

        Args:
            item: Item dictionary to validate

        Returns:
            {
                "item_id": str,
                "validation_timestamp": str,
                "schema_valid": bool,
                "auto_qa_passed": bool,
                "qa_score": float,
                "checks": dict,
                "qa_flags": list[str]
            }
        """
        result = {
            "item_id": item.get("id"),
            "validation_timestamp": datetime.utcnow().isoformat(),
            "schema_valid": None,
            "auto_qa_passed": None,
            "qa_score": 0.0,
            "checks": {},
            "qa_flags": []
        }

        # Stage 1: Schema validation (hard gate)
        schema_result = self.schema_validator.validate(item)
        result["checks"]["schema_validation"] = schema_result
        result["schema_valid"] = schema_result["passed"]

        if not schema_result["passed"]:
            result["auto_qa_passed"] = False
            result["qa_flags"].extend(schema_result["issues"])
            logger.warning(f"Schema validation failed: {schema_result['issues']}")
            return result

        # Stage 2: Readability check (warning gate)
        readability_result = self.readability_checker.check(item)
        result["checks"]["readability"] = readability_result

        # Readability issues are warnings, not hard gates
        if not readability_result.get("passed", True):
            result["qa_flags"].extend(readability_result.get("issues", []))

        # Stage 3: Quality rules (hard gate)
        quality_result = self.quality_rules_checker.check(item)
        result["checks"]["quality_rules"] = quality_result

        if not quality_result["passed"]:
            result["auto_qa_passed"] = False
            result["qa_flags"].extend(quality_result["issues"])
            logger.warning(f"Quality rules failed: {quality_result['issues']}")
            return result

        # All checks passed
        result["auto_qa_passed"] = True
        result["qa_score"] = 1.0

        logger.info(f"Item {item.get('id')} passed Auto-QA")
        return result
