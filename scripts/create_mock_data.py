#!/usr/bin/env python3
"""
Create mock SAT training data for smoke testing the training pipeline.

This generates synthetic SAT items in the correct schema format without
calling an LLM. Used for validating the training infrastructure works.
"""

import json
import uuid
from pathlib import Path

# Domain lists from CLAUDE.md
MATH_DOMAINS = [
    "algebra.linear_equations_one_variable",
    "algebra.linear_equations_two_variables",
    "algebra.linear_functions",
    "algebra.systems_of_linear_equations",
    "advanced_math.nonlinear_functions",
    "advanced_math.nonlinear_equations",
    "advanced_math.equivalent_expressions",
    "problem_solving_and_data_analysis.ratios_rates_proportions",
    "problem_solving_and_data_analysis.percentages",
    "problem_solving_and_data_analysis.one_variable_data",
    "problem_solving_and_data_analysis.two_variable_data",
    "problem_solving_and_data_analysis.probability",
    "problem_solving_and_data_analysis.inference_from_samples",
    "geometry_and_trigonometry.area_volume",
    "geometry_and_trigonometry.lines_angles_triangles",
    "geometry_and_trigonometry.right_triangles_trigonometry",
    "geometry_and_trigonometry.circles",
]

RW_DOMAINS = [
    "information_and_ideas.central_ideas_and_details",
    "information_and_ideas.command_of_evidence_textual",
    "information_and_ideas.inferences",
    "information_and_ideas.words_in_context",
    "craft_and_structure.text_structure_and_purpose",
    "craft_and_structure.cross_text_connections",
    "expression_of_ideas.rhetorical_synthesis",
    "expression_of_ideas.transitions",
    "standard_english_conventions.boundaries",
    "standard_english_conventions.form_structure_sense",
    "standard_english_conventions.standard_english",
]


def create_mock_math_item(difficulty: str, domain: str) -> dict:
    """Create a mock math item."""
    return {
        "id": str(uuid.uuid4()),
        "section": "math",
        "domain": domain,
        "difficulty": difficulty,
        "content_json": {
            "passage": None,
            "question": f"Solve for x in the equation 2x + 5 = 15. What is the value of x? This is a {difficulty} {domain} problem.",
            "math_format": "latex",
            "choices": [
                {"label": "A", "text": "x = 3"},
                {"label": "B", "text": "x = 4"},
                {"label": "C", "text": "x = 5"},
                {"label": "D", "text": "x = 6"},
            ],
            "correct_answer": "C",
            "correct_answer_text": "x = 5",
            "rationale": "Subtract 5 from both sides: 2x = 10. Then divide by 2: x = 5. This is the correct solution.",
            "solution_steps": "2x + 5 = 15 → 2x = 10 → x = 5",
        },
        "model_version": "mock-v1.0",
    }


def create_mock_rw_item(difficulty: str, domain: str) -> dict:
    """Create a mock reading/writing item."""
    return {
        "id": str(uuid.uuid4()),
        "section": "reading_writing",
        "domain": domain,
        "difficulty": difficulty,
        "content_json": {
            "passage": "The scientific method is a systematic approach to research. Scientists observe phenomena, form hypotheses, conduct experiments, and analyze results to draw conclusions.",
            "question": f"Which choice best states the main idea of the passage? This is a {difficulty} {domain} question.",
            "math_format": None,
            "choices": [
                {"label": "A", "text": "Science is only about experiments."},
                {"label": "B", "text": "The scientific method involves observation and analysis."},
                {"label": "C", "text": "Hypotheses are never proven correct."},
                {"label": "D", "text": "Research cannot be systematic."},
            ],
            "correct_answer": "B",
            "correct_answer_text": "The scientific method involves observation and analysis.",
            "rationale": "The passage describes the scientific method as a systematic process involving observation, hypotheses, experiments, and analysis. Choice B captures this comprehensive approach.",
            "solution_steps": None,
        },
        "model_version": "mock-v1.0",
    }


def create_mock_dataset(
    section: str, count_per_difficulty: int = 5, output_dir: str = "data/splits"
):
    """
    Create mock training dataset.

    Args:
        section: 'math' or 'reading_writing'
        count_per_difficulty: Number of items to generate per difficulty level
        output_dir: Directory to write splits
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select domains
    domains = MATH_DOMAINS if section == "math" else RW_DOMAINS

    # Generate items
    all_items = []

    for difficulty in ["easy", "medium", "hard"]:
        for i, domain in enumerate(domains[: count_per_difficulty]):
            if section == "math":
                item = create_mock_math_item(difficulty, domain)
            else:
                item = create_mock_rw_item(difficulty, domain)

            all_items.append(item)

    # Create splits (85% train, 10% val, 5% test)
    total = len(all_items)
    train_size = int(total * 0.85)
    val_size = int(total * 0.10)

    train_items = all_items[:train_size]
    val_items = all_items[train_size : train_size + val_size]
    test_items = all_items[train_size + val_size :]

    # Write splits
    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    print(f"✓ Created mock dataset for {section}")
    print(f"  Total items: {total}")
    print(f"  Train: {len(train_items)}")
    print(f"  Val: {len(val_items)}")
    print(f"  Test: {len(test_items)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create mock SAT training data")
    parser.add_argument(
        "--section", choices=["math", "reading_writing", "both"], default="both"
    )
    parser.add_argument("--count", type=int, default=5, help="Items per difficulty")
    parser.add_argument("--output", default="data/splits", help="Output directory")

    args = parser.parse_args()

    if args.section in ["math", "both"]:
        create_mock_dataset("math", args.count, args.output)

    if args.section in ["reading_writing", "both"]:
        create_mock_dataset("reading_writing", args.count, args.output)
