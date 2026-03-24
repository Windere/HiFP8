#!/usr/bin/env python3
"""
Compare smooth-absorbed model vs original with HiFloat8 fake quantization.

This simulates the vLLM plugin path:
  1. Load model (standard HuggingFace)
  2. Apply HiFP8 fake quant to all Linear layers (weight + activation)
  3. Measure accuracy

For the smooth-absorbed model, smooth scales are already baked into
weights + LayerNorm, so no separate smooth_scale handling is needed.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_smooth_absorbed.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant


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
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
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
    return torch.exp(torch.tensor(avg_loss)).item()


def quick_qa_accuracy(model, tokenizer, num_questions=30):
    """Quick multiple-choice accuracy test."""
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
        letters = ["A", "B", "C", "D"]
        prompt = f"Answer with just the letter. {question}\n"
        for j, choice in enumerate(choices):
            prompt += f"{letters[j]}. {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(inputs["input_ids"])

        last_logits = outputs.logits[0, -1, :]
        letter_ids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in letters]
        letter_logits = [last_logits[tid].item() for tid in letter_ids]
        predicted = letter_logits.index(max(letter_logits))

        if predicted == answer_idx:
            correct += 1

    return correct / total


def run_model(model_path, label, tokenizer):
    """Load model, apply HiFP8 fake quant, measure accuracy."""
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    print(f"[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()

    print(f"[2] Applying HiFP8 fake quantization (weight + activation)...")
    weight_config = HiFP8FakeQuantizeConfig()
    activation_config = HiFP8FakeQuantizeConfig()
    model = prepare_hifp8_fake_quant(
        model, weight_config=weight_config, activation_config=activation_config,
    )

    print(f"[3] Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer)
    print(f"  PPL: {ppl:.2f}")

    print(f"[4] Running QA accuracy test (30 questions)...")
    acc = quick_qa_accuracy(model, tokenizer)
    print(f"  QA Accuracy: {acc:.2%} ({int(acc*30)}/30)")

    # Quick generation check
    print(f"[5] Sanity check generation...")
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"  Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    del model
    torch.cuda.empty_cache()

    return {"ppl": ppl, "acc": acc}


def main():
    print("=" * 60)
    print("Smooth-Absorbed vs Baseline Comparison")
    print("Both models through HiFP8 fake quant (vLLM plugin path)")
    print("=" * 60)

    original_model = "Qwen/Qwen3-0.6B"
    smooth_model = "/tmp/qwen3-0.6b-smooth-absorbed"

    tokenizer = AutoTokenizer.from_pretrained(original_model, trust_remote_code=True)

    baseline = run_model(original_model, "Baseline (no smooth)", tokenizer)
    smooth = run_model(smooth_model, "Smooth-Absorbed (alpha=0.3, clip=5)", tokenizer)

    print(f"\n{'='*60}")
    print("COMPARISON (HiFP8 fake quant applied to both)")
    print(f"{'='*60}")
    print(f"{'Config':<40} | {'PPL':>8} | {'QA Acc':>8}")
    print(f"{'-'*40}-+-{'-'*8}-+-{'-'*8}")
    print(f"{'Baseline (no smooth)':<40} | {baseline['ppl']:>8.2f} | {baseline['acc']:>8.2%}")
    print(f"{'Smooth-Absorbed (a=0.3, clip=5)':<40} | {smooth['ppl']:>8.2f} | {smooth['acc']:>8.2%}")
    print(f"{'Delta':<40} | {smooth['ppl']-baseline['ppl']:>+8.2f} | {smooth['acc']-baseline['acc']:>+8.2%}")
    print()


if __name__ == "__main__":
    main()
