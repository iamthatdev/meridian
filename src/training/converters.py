# src/training/converters.py
import json
from typing import List, Dict, Any

def convert_item_to_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert IIAS schema item to chat messages format.

    Args:
        item: IIAS item with section, domain, difficulty, content_json

    Returns:
        List of message dicts with role and content
    """
    section = "Reading & Writing" if item["section"] == "reading_writing" else "Math"

    user_msg = f"""Generate a SAT {section} item with these constraints:
- Domain: {item["domain"]}
- Difficulty: {item["difficulty"]}

Output JSON only."""

    assistant_msg = json.dumps(item["content_json"])

    return [
        {"role": "system", "content": "You are an expert SAT item writer."},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ]
