#!/usr/bin/env python3
"""
Verify that model tokenizers support required chat template format.

This script tests that phi-4 and Qwen2.5 tokenizers have the
apply_chat_template() method required by trl.SFTTrainer.
"""

import sys
from transformers import AutoTokenizer

def verify_tokenizer_chat_template(model_id: str, model_name: str) -> bool:
    """Verify tokenizer supports chat templates."""
    print(f"\nTesting {model_name} tokenizer...")
    print(f"Model ID: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Check for chat template support
        if not hasattr(tokenizer, 'apply_chat_template'):
            print(f"❌ {model_name} does NOT support chat templates")
            return False

        # Test actual formatting
        test_messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]

        formatted = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        print(f"✓ {model_name} supports chat templates")
        print(f"  Sample formatted output:\n  {formatted[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return False

def main():
    """Verify all production models support chat templates."""
    print("=" * 70)
    print("Chat Template Compatibility Verification")
    print("=" * 70)

    models_to_test = [
        ("microsoft/phi-4", "phi-4 (Math)"),
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B (Reading/Writing)"),
    ]

    results = []
    for model_id, model_name in models_to_test:
        success = verify_tokenizer_chat_template(model_id, model_name)
        results.append((model_name, success))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = all(success for _, success in results)

    for model_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {model_name}")

    if all_passed:
        print("\n✓ All models support required chat templates")
        print("  Safe to proceed with trl.SFTTrainer migration")
        return 0
    else:
        print("\n✗ Some models lack chat template support")
        print("  Cannot proceed with migration")
        return 1

if __name__ == "__main__":
    sys.exit(main())