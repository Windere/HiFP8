#!/usr/bin/env python3
"""
Compare BF16 baseline vs HiF8 exported GPT-OSS 20B outputs.

Uses vLLM offline inference (LLM class) for memory-efficient loading.
Loads each model sequentially, runs the same prompts, compares text.
"""

import argparse
import gc
import os
import sys
import time

os.environ["HF_HOME"] = "/home/data/.cache/huggingface"
os.environ["MODELSCOPE_CACHE"] = "/home/data/.cache/modelscope"
os.environ["TRANSFORMERS_CACHE"] = "/home/data/.cache/huggingface"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ao"))

BF16_MODEL = "/home/models/gpt-oss-20b-BF16"
HIF8_EXPORT = "/home/data/hifp8_eval/gpt_oss_20b_hif8"

TEST_PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is",
    "The Pythagorean theorem states that",
    "Water boils at a temperature of",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-model", default=BF16_MODEL)
    parser.add_argument("--hif8-model", default=HIF8_EXPORT)
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def vllm_generate(model_path, prompts, label, args):
    """Load model with vLLM offline LLM and generate text."""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"[{label}] Loading: {model_path}")
    print(f"{'='*60}")

    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"[{label}] Loaded in {time.time()-t0:.1f}s")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0,  # greedy
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        full_text = prompt + generated
        results.append(full_text)
        print(f"  [{label}] {prompt!r} -> {generated[:80]}...")

    # Free GPU
    del llm
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # vLLM may hold GPU memory; give it time to release
    time.sleep(5)

    return results


def compare_results(bf16_results, hif8_results, prompts):
    """Compare BF16 vs HiF8 generated text."""
    print(f"\n{'='*60}")
    print("OUTPUT COMPARISON: BF16 vs HiF8")
    print(f"{'='*60}")

    match_count = 0
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt: {prompt!r}")
        bf16_gen = bf16_results[i][len(prompt):]
        hif8_gen = hif8_results[i][len(prompt):]
        print(f"  BF16: {bf16_gen[:120]}")
        print(f"  HiF8: {hif8_gen[:120]}")
        match = bf16_gen.strip() == hif8_gen.strip()
        if match:
            match_count += 1
        print(f"  Match: {'YES' if match else 'NO'}")

    print(f"\n  Text match: {match_count}/{len(prompts)}")
    return match_count


def main():
    args = parse_args()

    # Step 1: BF16 baseline
    bf16_results = vllm_generate(args.bf16_model, TEST_PROMPTS, "BF16", args)

    # Step 2: HiF8 export (uses colleague's vLLM-HiF8 fork which is installed)
    hif8_results = vllm_generate(args.hif8_model, TEST_PROMPTS, "HiF8", args)

    # Step 3: Compare
    match_count = compare_results(bf16_results, hif8_results, TEST_PROMPTS)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Text match: {match_count}/{len(TEST_PROMPTS)}")
    if match_count == len(TEST_PROMPTS):
        print("  PASS: All outputs match exactly")
    elif match_count >= len(TEST_PROMPTS) * 0.5:
        print("  OK: Most outputs match (minor quant differences expected)")
    else:
        print("  WARN: Significant divergence in generated text")


if __name__ == "__main__":
    main()
