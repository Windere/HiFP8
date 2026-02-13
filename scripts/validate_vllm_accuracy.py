#!/usr/bin/env python3
"""
Validate HiFP8 vLLM Server Accuracy

Compares outputs between baseline vLLM server and HiFP8 server to verify
that fake quantization produces expected results.

Usage:
    # Terminal 1: Start baseline server
    vllm serve /home/models/Qwen3-0.6B --port 8000 --reasoning-parser qwen3

    # Terminal 2: Start HiFP8 server
    python scripts/start_vllm_hifp8_server_v2.py \\
        --model /home/data/quantized_qwen3_0.6b \\
        --port 8001 \\
        --reasoning-parser qwen3

    # Terminal 3: Run validation
    python scripts/validate_vllm_accuracy.py \\
        --baseline-url http://localhost:8000 \\
        --hifp8-url http://localhost:8001 \\
        --num-samples 50

Expected: >95% similarity between outputs (minor differences due to quantization)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time

try:
    import openai
except ImportError:
    print("Error: openai package required")
    print("Install with: pip install openai")
    sys.exit(1)


# Test prompts covering different scenarios
DEFAULT_TEST_PROMPTS = [
    # Simple factual
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is 2+2?",

    # Reasoning
    "If John is taller than Mary, and Mary is taller than Sue, who is the shortest?",
    "A train leaves at 3pm and arrives at 5pm. How long is the journey?",

    # Math
    "Calculate: 15 * 7",
    "What is 100 divided by 4?",

    # Multi-step
    "List three primary colors.",
    "Name two planets in our solar system.",

    # Longer context
    "Explain in one sentence what machine learning is.",
]


def compute_token_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two texts at token level.

    Uses simple word-based similarity for now.
    Could be enhanced with edit distance or token-level comparison.
    """
    if text1 == text2:
        return 1.0

    tokens1 = text1.split()
    tokens2 = text2.split()

    if not tokens1 or not tokens2:
        return 0.0

    # Simple token overlap
    set1 = set(tokens1)
    set2 = set(tokens2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


async def test_chat_completion(
    client: openai.AsyncOpenAI,
    prompt: str,
    temperature: float = 0.0
) -> str:
    """Test basic chat completion."""
    try:
        response = await client.chat.completions.create(
            model="test",  # Model name doesn't matter for vLLM
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Chat completion failed: {e}")


async def test_streaming(
    client: openai.AsyncOpenAI,
    prompt: str,
    temperature: float = 0.0
) -> str:
    """Test streaming responses."""
    try:
        chunks = []
        stream = await client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=temperature,
            max_tokens=100,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        return "".join(chunks)
    except Exception as e:
        raise RuntimeError(f"Streaming test failed: {e}")


async def test_multi_turn(
    client: openai.AsyncOpenAI,
    temperature: float = 0.0
) -> str:
    """Test multi-turn conversation."""
    try:
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"},
        ]

        response = await client.chat.completions.create(
            model="test",
            messages=messages,
            temperature=temperature,
            max_tokens=100,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Multi-turn test failed: {e}")


async def validate_single_prompt(
    baseline_client: openai.AsyncOpenAI,
    hifp8_client: openai.AsyncOpenAI,
    prompt: str,
    test_name: str = "chat",
    verbose: bool = True
) -> Dict:
    """Validate a single prompt across both servers."""
    if verbose:
        print(f"\n[Test] {test_name}: {prompt[:60]}...")

    try:
        # Get both outputs
        if test_name == "streaming":
            baseline_output = await test_streaming(baseline_client, prompt)
            hifp8_output = await test_streaming(hifp8_client, prompt)
        elif test_name == "multi_turn":
            baseline_output = await test_multi_turn(baseline_client)
            hifp8_output = await test_multi_turn(hifp8_client)
        else:
            baseline_output = await test_chat_completion(baseline_client, prompt)
            hifp8_output = await test_chat_completion(hifp8_client, prompt)

        # Compare
        exact_match = baseline_output == hifp8_output
        similarity = compute_token_similarity(baseline_output, hifp8_output)

        if exact_match:
            status = "✓ Exact match"
        elif similarity > 0.95:
            status = f"✓ Very similar ({similarity:.1%})"
        elif similarity > 0.80:
            status = f"⚠ Similar ({similarity:.1%})"
        else:
            status = f"✗ Different ({similarity:.1%})"

        if verbose:
            print(f"  {status}")
            if not exact_match and similarity < 0.95:
                print(f"  Baseline: {baseline_output[:100]}")
                print(f"  HiFP8:    {hifp8_output[:100]}")

        return {
            "prompt": prompt,
            "test_name": test_name,
            "exact_match": exact_match,
            "similarity": similarity,
            "baseline_output": baseline_output,
            "hifp8_output": hifp8_output,
            "passed": similarity > 0.80,  # 80% threshold
        }

    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
        return {
            "prompt": prompt,
            "test_name": test_name,
            "error": str(e),
            "passed": False,
        }


async def validate_accuracy(
    baseline_url: str,
    hifp8_url: str,
    test_prompts: List[str],
    num_samples: int = None,
    verbose: bool = True
) -> Dict:
    """Run full validation suite."""
    print("=" * 80)
    print("HiFP8 vLLM Server Accuracy Validation")
    print("=" * 80)
    print(f"Baseline URL: {baseline_url}")
    print(f"HiFP8 URL: {hifp8_url}")
    print("=" * 80)

    # Create clients
    baseline_client = openai.AsyncOpenAI(
        base_url=f"{baseline_url}/v1",
        api_key="dummy"  # vLLM doesn't require real API key
    )
    hifp8_client = openai.AsyncOpenAI(
        base_url=f"{hifp8_url}/v1",
        api_key="dummy"
    )

    # Limit samples if requested
    if num_samples:
        test_prompts = test_prompts[:num_samples]

    results = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "exact_matches": 0,
        "similarities": [],
        "details": []
    }

    # Test 1: Basic chat completions
    print("\n[Phase 1] Testing basic chat completions...")
    for prompt in test_prompts:
        result = await validate_single_prompt(
            baseline_client, hifp8_client, prompt, "chat", verbose
        )
        results["details"].append(result)
        results["total"] += 1

        if result.get("passed", False):
            results["passed"] += 1
            if result.get("exact_match", False):
                results["exact_matches"] += 1
        else:
            results["failed"] += 1

        if "similarity" in result:
            results["similarities"].append(result["similarity"])

    # Test 2: Streaming (use first 3 prompts)
    print("\n[Phase 2] Testing streaming...")
    for prompt in test_prompts[:3]:
        result = await validate_single_prompt(
            baseline_client, hifp8_client, prompt, "streaming", verbose
        )
        results["details"].append(result)
        results["total"] += 1

        if result.get("passed", False):
            results["passed"] += 1
        else:
            results["failed"] += 1

        if "similarity" in result:
            results["similarities"].append(result["similarity"])

    # Test 3: Multi-turn (one test)
    print("\n[Phase 3] Testing multi-turn conversation...")
    result = await validate_single_prompt(
        baseline_client, hifp8_client, "", "multi_turn", verbose
    )
    results["details"].append(result)
    results["total"] += 1

    if result.get("passed", False):
        results["passed"] += 1
    else:
        results["failed"] += 1

    if "similarity" in result:
        results["similarities"].append(result["similarity"])

    return results


def print_summary(results: Dict):
    """Print validation summary."""
    print("\n" + "=" * 80)
    print("Validation Results")
    print("=" * 80)

    total = results["total"]
    passed = results["passed"]
    failed = results["failed"]
    exact = results["exact_matches"]

    pass_rate = (passed / total * 100) if total > 0 else 0
    exact_rate = (exact / total * 100) if total > 0 else 0

    print(f"Total tests: {total}")
    print(f"Passed: {passed} ({pass_rate:.1f}%)")
    print(f"Failed: {failed} ({(100-pass_rate):.1f}%)")
    print(f"Exact matches: {exact} ({exact_rate:.1f}%)")

    if results["similarities"]:
        avg_similarity = sum(results["similarities"]) / len(results["similarities"])
        print(f"Average similarity: {avg_similarity:.1%}")

    print("\n" + "=" * 80)

    if pass_rate >= 95:
        print("✅ PASS: Accuracy validation successful (≥95% pass rate)")
        return 0
    elif pass_rate >= 80:
        print("⚠️  WARNING: Moderate accuracy (80-95% pass rate)")
        print("   Some differences detected, but within acceptable range")
        return 0
    else:
        print("❌ FAIL: Accuracy validation failed (<80% pass rate)")
        print("   Significant differences detected between baseline and HiFP8")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate HiFP8 vLLM server accuracy"
    )
    parser.add_argument(
        "--baseline-url",
        type=str,
        required=True,
        help="Baseline vLLM server URL (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--hifp8-url",
        type=str,
        required=True,
        help="HiFP8 vLLM server URL (e.g., http://localhost:8001)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of test prompts to use (default: all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (less verbose output)"
    )

    args = parser.parse_args()

    # Run validation
    results = asyncio.run(
        validate_accuracy(
            args.baseline_url,
            args.hifp8_url,
            DEFAULT_TEST_PROMPTS,
            args.num_samples,
            verbose=not args.quiet
        )
    )

    # Print summary and exit
    exit_code = print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
