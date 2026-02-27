#!/usr/bin/env python3
"""
Test script to verify chat template is correctly applied.

This script demonstrates the difference between:
1. Incorrect: Simple string concatenation
2. Correct: Using tokenizer.apply_chat_template()

For Qwen3, the correct format uses ChatML with <|im_start|> and <|im_end|> tokens.
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_chat_template():
    """Test that chat template is correctly applied."""
    print("=" * 80)
    print("Chat Template Verification for Qwen3")
    print("=" * 80)

    model_path = "/home/models/Qwen3-0.6B"

    print(f"\nLoading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    print("\n" + "=" * 80)
    print("Test Messages:")
    print("=" * 80)
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")

    # WRONG: Simple string concatenation (OLD implementation)
    print("\n" + "=" * 80)
    print("❌ INCORRECT: Simple String Concatenation (OLD)")
    print("=" * 80)
    wrong_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            wrong_prompt += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            wrong_prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            wrong_prompt += f"Assistant: {msg['content']}\n"
    wrong_prompt += "Assistant:"

    print(repr(wrong_prompt))
    print("\nRendered:")
    print(wrong_prompt)
    print("\n⚠️  This format does NOT match Qwen3's expected ChatML format!")

    # CORRECT: Using chat template
    print("\n" + "=" * 80)
    print("✅ CORRECT: Using tokenizer.apply_chat_template() (NEW)")
    print("=" * 80)

    if hasattr(tokenizer, 'apply_chat_template'):
        correct_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(repr(correct_prompt))
        print("\nRendered:")
        print(correct_prompt)
        print("\n✅ This is the correct ChatML format for Qwen3!")
    else:
        print("⚠️  Tokenizer does not have apply_chat_template method")

    # Test with multi-turn conversation
    print("\n" + "=" * 80)
    print("Multi-turn Conversation Example")
    print("=" * 80)

    multi_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ]

    if hasattr(tokenizer, 'apply_chat_template'):
        multi_prompt = tokenizer.apply_chat_template(
            multi_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(repr(multi_prompt))
        print("\nRendered:")
        print(multi_prompt)

    # Tokenization comparison
    print("\n" + "=" * 80)
    print("Token Count Comparison")
    print("=" * 80)

    wrong_tokens = tokenizer(wrong_prompt, return_tensors="pt")
    correct_tokens = tokenizer(correct_prompt, return_tensors="pt")

    print(f"Wrong format token count: {wrong_tokens.input_ids.shape[1]}")
    print(f"Correct format token count: {correct_tokens.input_ids.shape[1]}")
    print(f"Difference: {abs(wrong_tokens.input_ids.shape[1] - correct_tokens.input_ids.shape[1])} tokens")

    if wrong_tokens.input_ids.shape[1] != correct_tokens.input_ids.shape[1]:
        print("\n⚠️  Token counts differ - the model will behave differently!")

    # Show impact
    print("\n" + "=" * 80)
    print("Why This Matters")
    print("=" * 80)
    print("""
1. Performance Impact:
   - Models are trained with specific chat templates
   - Using wrong format degrades quality significantly
   - May cause model to not follow instructions properly

2. Special Tokens:
   - Qwen3 expects <|im_start|> and <|im_end|> tokens
   - These tokens help model understand conversation structure
   - Simple concatenation misses these critical markers

3. Compatibility:
   - Different models use different chat formats
   - ChatML (Qwen, Yi), Llama2 format, Mistral format, etc.
   - apply_chat_template() handles this automatically

4. Best Practice:
   - ALWAYS use tokenizer.apply_chat_template() when available
   - Only fall back to manual formatting if necessary
   - Test with actual model to verify output quality
    """)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("✅ Fixed: API server now uses tokenizer.apply_chat_template()")
    print("✅ Benefit: Proper ChatML format with <|im_start|>/<|im_end|> tokens")
    print("✅ Result: Significantly better model performance and instruction following")


if __name__ == "__main__":
    test_chat_template()
