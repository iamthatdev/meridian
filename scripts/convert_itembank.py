"""Convert ItemBank data to training format for fine-tuning."""

import json
import uuid
from pathlib import Path
from loguru import logger


def convert_itembank(source_path: str, output_dir: str):
    """
    Convert ItemBank format to Meridian training format.

    Args:
        source_path: Path to ItemBank JSON file
        output_dir: Directory to write training JSONL files
    """
    source_file = Path(source_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {source_file}...")

    # Load ItemBank data
    with open(source_file) as f:
        items = json.load(f)

    logger.info(f"Loaded {len(items)} items from ItemBank")

    # Separate by section
    rw_items = []
    math_items = []

    for item in items:
        # Extract fields
        section = item.get("section", "")
        domain = item.get("domain", "")
        difficulty = item.get("difficulty_tier", "medium")
        content = item.get("content_json", {})
        topic = content.get("metadata", {}).get("topic", "")

        # Build training example with chat messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert SAT item writer. Generate items in strict JSON matching the provided schema. Output JSON only, with no preamble, explanation, or markdown formatting."
            },
            {
                "role": "user",
                "content": f"Generate a single SAT {'Reading & Writing' if section == 'reading_writing' else 'Math'} item.\n\nConstraints:\n- section: {section}\n- domain: {domain}\n- difficulty_tier: {difficulty}" + (f"\n- topic: {topic}" if topic else "")
            },
            {
                "role": "assistant",
                "content": json.dumps(content, indent=2)
            }
        ]

        training_example = {
            "dataset_version": f"{section}-sft-v1.0",
            "schema_version": "item-schema-v1",
            "section": section,
            "domain": domain,
            "difficulty_tier": difficulty,
            "messages": messages
        }

        # Separate by section
        if section == "reading_writing":
            rw_items.append(training_example)
        elif section == "math":
            math_items.append(training_example)

    logger.info(f"Separated {len(rw_items)} RW items, {len(math_items)} Math items")

    # Split into train/val (85/15)
    def split(items):
        split_idx = int(len(items) * 0.85)
        return items[:split_idx], items[split_idx:]

    rw_train, rw_val = split(rw_items)
    math_train, math_val = split(math_items)

    logger.info(f"Split: RW train={len(rw_train)}, val={len(rw_val)}")
    logger.info(f"Split: Math train={len(math_train)}, val={len(math_val)}")

    # Write JSONL files
    def write_jsonl(items, path):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    write_jsonl(rw_train, output_path / "rw_train.jsonl")
    write_jsonl(rw_val, output_path / "rw_val.jsonl")
    write_jsonl(math_train, output_path / "math_train.jsonl")
    write_jsonl(math_val, output_path / "math_val.jsonl")

    logger.info(f"✅ Conversion complete:")
    logger.info(f"  RW: {len(rw_train)} train, {len(rw_val)} val")
    logger.info(f"  Math: {len(math_train)} train, {len(math_val)} val")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python convert_itembank.py <source_path> <output_dir>")
        print("Example: python scripts/convert_itembank.py \\")
        print("  /Users/pradeep/projects/sat_synthetic_generator/data/itembank_questions_complete.json \\")
        print("  data/training/")
        sys.exit(1)

    convert_itembank(sys.argv[1], sys.argv[2])
