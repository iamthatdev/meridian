#!/usr/bin/env python3
"""
Verify training data files have correct format for trl.SFTTrainer.

Expected format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


def validate_example(example: dict, line_num: int) -> List[Tuple[int, str]]:
    """Validate a single example has correct format.

    Returns list of (line_num, error_message) tuples.
    """
    errors = []

    # Check for messages field
    if "messages" not in example:
        errors.append((
            line_num,
            f"Missing 'messages' field. Found keys: {list(example.keys())}"
        ))
        return errors

    messages = example["messages"]

    # Check messages is a list
    if not isinstance(messages, list):
        errors.append((line_num, "'messages' must be a list"))
        return errors

    # Check messages is not empty
    if len(messages) == 0:
        errors.append((line_num, "'messages' list is empty"))
        return errors

    # Check each message has required fields
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append((
                line_num,
                f"Message {msg_idx} is not a dict"
            ))
            continue

        if "role" not in msg:
            errors.append((
                line_num,
                f"Message {msg_idx} missing 'role' field"
            ))

        if "content" not in msg:
            errors.append((
                line_num,
                f"Message {msg_idx} missing 'content' field"
            ))

        # Validate role is one of the expected values
        if "role" in msg:
            valid_roles = {"system", "user", "assistant"}
            if msg["role"] not in valid_roles:
                errors.append((
                    line_num,
                    f"Message {msg_idx} has invalid role: {msg['role']}. "
                    f"Must be one of: {valid_roles}"
                ))

    return errors


def validate_data_file(data_file: Path, max_errors: int = 10) -> bool:
    """Validate a single data file.

    Returns True if valid, False otherwise.
    """
    print(f"\nValidating: {data_file}")

    if not data_file.exists():
        print(f"✗ File not found: {data_file}")
        return False

    errors = []
    line_count = 0

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                line_count += 1

                try:
                    example = json.loads(line)
                    example_errors = validate_example(example, line_num)
                    errors.extend(example_errors)

                    # Stop if we've found too many errors
                    if len(errors) >= max_errors:
                        break

                except json.JSONDecodeError as e:
                    errors.append((
                        line_num,
                        f"Invalid JSON: {e}"
                    ))
                    if len(errors) >= max_errors:
                        break

    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

    # Report results
    if errors:
        print(f"✗ Found {len(errors)} error(s) in {line_count} lines:")
        for line_num, error_msg in errors[:max_errors]:
            print(f"  Line {line_num}: {error_msg}")
        return False
    else:
        print(f"✓ All {line_count} examples have valid format")
        return True


def main():
    """Validate training data files."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify training data format for trl.SFTTrainer"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Training data files to validate"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Training Data Format Validation")
    print("=" * 70)

    results = []
    for data_file in args.files:
        valid = validate_data_file(data_file)
        results.append((data_file, valid))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_valid = all(valid for _, valid in results)

    for data_file, valid in results:
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"{status}: {data_file}")

    if all_valid:
        print("\n✓ All files have correct format for trl.SFTTrainer")
        return 0
    else:
        print("\n✗ Some files have format errors")
        print("  Please fix errors before proceeding with training")
        return 1


if __name__ == "__main__":
    sys.exit(main())