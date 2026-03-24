#!/usr/bin/env python3
"""
Sweep SmoothQuant alpha for Qwen3-0.6B.

Loads model once, then for each alpha:
  1. Deep-copy model
  2. Apply SmoothQuant with that alpha
  3. Evaluate perplexity on held-out wikitext validation set
  4. Run quick few-shot ARC-like accuracy check

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/sweep_alpha.py
"""

import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import (
    HiFP8FakeQuantizeConfig,
    prepare_hifp8_fake_quant,
    calibrate_and_smooth,
)
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear


def create_calibration_dataloader(tokenizer, max_length=512, num_samples=64):
    """Create calibration dataloader from wikitext train split."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    data = []
    for sample in dataset:
        text = sample["text"].strip()
        if len(text) < 20:
            continue
        tokens = tokenizer(
            text, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        data.append({
            "input_ids": tokens["input_ids"].cuda(),
            "attention_mask": tokens["attention_mask"].cuda(),
        })
        if len(data) >= num_samples:
            break
    return data


def compute_perplexity(model, tokenizer, max_samples=50, max_length=512):
    """Compute perplexity on wikitext-2 validation set."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    total_loss = 0.0
    total_tokens = 0
    count = 0

    model.eval()
    with torch.no_grad():
        for sample in dataset:
            text = sample["text"].strip()
            if len(text) < 50:
                continue

            tokens = tokenizer(
                text, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(model.device)

            input_ids = tokens["input_ids"]
            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :]  # [B, seq-1, vocab]
            targets = input_ids[:, 1:]           # [B, seq-1]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
            count += 1

            if count >= max_samples:
                break

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def quick_qa_accuracy(model, tokenizer, num_questions=30):
    """Quick multiple-choice accuracy test using simple factual questions."""
    questions = [
        ("The capital of France is", ["Paris", "London", "Berlin", "Madrid"], 0),
        ("Water freezes at", ["100 degrees", "0 degrees", "50 degrees", "-10 degrees"], 1),
        ("The largest planet in the solar system is", ["Earth", "Mars", "Jupiter", "Saturn"], 2),
        ("The chemical symbol for gold is", ["Ag", "Fe", "Au", "Cu"], 2),
        ("The speed of light is approximately", ["300 km/s", "300,000 km/s", "3,000 km/s", "30,000 km/s"], 1),
        ("DNA stands for", ["Deoxyribonucleic acid", "Dynamic nuclear acid", "Digital network array", "Dense nucleic acid"], 0),
        ("The largest ocean on Earth is the", ["Atlantic", "Indian", "Arctic", "Pacific"], 3),
        ("Photosynthesis converts sunlight into", ["heat", "chemical energy", "electricity", "kinetic energy"], 1),
        ("The boiling point of water at sea level is", ["90C", "100C", "110C", "80C"], 1),
        ("Newton's first law is also called the law of", ["gravity", "inertia", "motion", "force"], 1),
        ("The smallest unit of matter is an", ["molecule", "cell", "atom", "electron"], 2),
        ("The human body has how many chromosomes", ["23 pairs", "22 pairs", "24 pairs", "46 pairs"], 0),
        ("Sound travels fastest through", ["air", "water", "steel", "vacuum"], 2),
        ("The closest star to Earth is", ["Sirius", "Alpha Centauri", "The Sun", "Polaris"], 2),
        ("CO2 is the chemical formula for", ["carbon monoxide", "carbon dioxide", "calcium oxide", "copper oxide"], 1),
        ("The Great Wall of China was primarily built to", ["trade", "defend against invasions", "mark territory", "irrigation"], 1),
        ("Mitochondria are known as the", ["brain of the cell", "powerhouse of the cell", "wall of the cell", "nucleus of the cell"], 1),
        ("The pH of pure water is", ["5", "7", "9", "14"], 1),
        ("Light years measure", ["time", "distance", "speed", "brightness"], 1),
        ("The hardest natural substance is", ["iron", "quartz", "diamond", "granite"], 2),
        ("Gravity on the Moon is about", ["same as Earth", "1/6 of Earth", "1/2 of Earth", "1/3 of Earth"], 1),
        ("The main gas in Earth's atmosphere is", ["oxygen", "carbon dioxide", "nitrogen", "hydrogen"], 2),
        ("Absolute zero is approximately", ["-273C", "-100C", "0C", "-200C"], 0),
        ("Electrons have a", ["positive charge", "negative charge", "no charge", "variable charge"], 1),
        ("The human heart has", ["2 chambers", "3 chambers", "4 chambers", "5 chambers"], 2),
        ("The Pythagorean theorem relates to", ["circles", "right triangles", "squares", "cubes"], 1),
        ("Insulin is produced by the", ["liver", "kidneys", "pancreas", "stomach"], 2),
        ("The speed of sound in air is about", ["340 m/s", "1000 m/s", "170 m/s", "3000 m/s"], 0),
        ("Mammals are", ["cold-blooded", "warm-blooded", "neither", "both"], 1),
        ("The periodic table is organized by", ["weight", "atomic number", "size", "color"], 1),
    ]

    correct = 0
    total = min(num_questions, len(questions))
    model.eval()

    for i in range(total):
        question, choices, answer_idx = questions[i]
        # Format as MCQ
        letters = ["A", "B", "C", "D"]
        prompt = f"Answer with just the letter. {question}\n"
        for j, choice in enumerate(choices):
            prompt += f"{letters[j]}. {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"])

        # Get logits for last token, check which letter has highest prob
        last_logits = outputs.logits[0, -1, :]
        letter_ids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in letters]
        letter_logits = [last_logits[tid].item() for tid in letter_ids]
        predicted = letter_logits.index(max(letter_logits))

        if predicted == answer_idx:
            correct += 1

    return correct / total


def smooth_scale_stats(model):
    """Get statistics of smooth_scale values across all layers."""
    all_vals = []
    for name, module in model.named_modules():
        if isinstance(module, HiFP8FakeQuantizedLinear) and module.smooth_scale is not None:
            all_vals.append(module.smooth_scale.detach().cpu().float())
    if not all_vals:
        return {}
    cat = torch.cat(all_vals)
    return {
        "mean": cat.mean().item(),
        "std": cat.std().item(),
        "min": cat.min().item(),
        "max": cat.max().item(),
        "median": cat.median().item(),
        "pct_gt_2": (cat > 2.0).float().mean().item() * 100,
        "pct_gt_5": (cat > 5.0).float().mean().item() * 100,
        "pct_gt_10": (cat > 10.0).float().mean().item() * 100,
    }


def main():
    print("=" * 70)
    print("SmoothQuant Alpha Sweep - Qwen3-0.6B")
    print("=" * 70)

    model_name = "Qwen/Qwen3-0.6B"

    # Load model and tokenizer
    print(f"\n[1] Loading model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Apply HiFP8 fake quantization
    print("[2] Applying HiFP8 fake quantization...")
    weight_config = HiFP8FakeQuantizeConfig()
    activation_config = HiFP8FakeQuantizeConfig()
    base_model = prepare_hifp8_fake_quant(
        base_model, weight_config=weight_config, activation_config=activation_config,
    )

    # Prepare calibration data (shared across all alphas)
    print("[3] Preparing calibration data...")
    dataloader = create_calibration_dataloader(tokenizer, max_length=512, num_samples=32)

    # Baseline: no SmoothQuant
    print("\n[4] Evaluating baseline (no SmoothQuant)...")
    base_ppl = compute_perplexity(base_model, tokenizer)
    base_acc = quick_qa_accuracy(base_model, tokenizer)
    print(f"  Baseline PPL: {base_ppl:.2f}, QA Acc: {base_acc:.2%}")

    # Alpha sweep
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    results = []

    for alpha in alphas:
        print(f"\n{'='*50}")
        print(f"[Alpha = {alpha}]")
        print(f"{'='*50}")

        # Deep copy model to avoid contamination between alphas
        model = copy.deepcopy(base_model)

        # Calibrate and smooth
        smooth_scales = calibrate_and_smooth(
            model, dataloader, alpha=alpha, num_batches=len(dataloader),
        )

        # Scale stats
        stats = smooth_scale_stats(model)
        print(f"  Scale stats: mean={stats['mean']:.3f}, max={stats['max']:.1f}, "
              f"median={stats['median']:.3f}, >2: {stats['pct_gt_2']:.1f}%, "
              f">5: {stats['pct_gt_5']:.1f}%, >10: {stats['pct_gt_10']:.1f}%")

        # Perplexity
        ppl = compute_perplexity(model, tokenizer)
        print(f"  PPL: {ppl:.2f} (baseline: {base_ppl:.2f}, delta: {ppl - base_ppl:+.2f})")

        # Quick QA
        acc = quick_qa_accuracy(model, tokenizer)
        print(f"  QA Acc: {acc:.2%} (baseline: {base_acc:.2%}, delta: {acc - base_acc:+.2%})")

        results.append({
            "alpha": alpha,
            "ppl": ppl,
            "acc": acc,
            "scale_mean": stats["mean"],
            "scale_max": stats["max"],
            "scale_median": stats["median"],
            "pct_gt_2": stats["pct_gt_2"],
            "pct_gt_10": stats["pct_gt_10"],
        })

        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Alpha':>6} | {'PPL':>8} | {'dPPL':>8} | {'QA Acc':>8} | {'dAcc':>8} | "
          f"{'s_mean':>7} | {'s_max':>7} | {'s_med':>7} | {'>2%':>5} | {'>10%':>5}")
    print("-" * 100)
    print(f"{'base':>6} | {base_ppl:>8.2f} | {'':>8} | {base_acc:>8.2%} | {'':>8} | "
          f"{'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>5} | {'N/A':>5}")
    for r in results:
        print(f"{r['alpha']:>6.2f} | {r['ppl']:>8.2f} | {r['ppl'] - base_ppl:>+8.2f} | "
              f"{r['acc']:>8.2%} | {r['acc'] - base_acc:>+8.2%} | "
              f"{r['scale_mean']:>7.3f} | {r['scale_max']:>7.1f} | "
              f"{r['scale_median']:>7.3f} | {r['pct_gt_2']:>5.1f} | {r['pct_gt_10']:>5.1f}")

    # Find best alpha by minimum PPL
    best_ppl = min(results, key=lambda x: x["ppl"])
    best_acc = max(results, key=lambda x: x["acc"])
    print(f"\nBest by PPL:    alpha={best_ppl['alpha']} (PPL={best_ppl['ppl']:.2f})")
    print(f"Best by QA Acc: alpha={best_acc['alpha']} (Acc={best_acc['acc']:.2%})")


if __name__ == "__main__":
    main()
