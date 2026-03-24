#!/usr/bin/env python3
"""
Compare alpha-based SmoothQuant vs HiFloat8-aware scale search.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/test_smooth_search.py
"""
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear


def create_calibration_dataloader(tokenizer, max_length=512, num_samples=32):
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


def quick_qa_accuracy(model, tokenizer, num_questions=30):
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


def sanity_check(model, tokenizer):
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def scale_stats(model):
    all_vals = []
    for name, module in model.named_modules():
        if isinstance(module, HiFP8FakeQuantizedLinear) and module.smooth_scale is not None:
            all_vals.append(module.smooth_scale.detach().cpu().float())
    if not all_vals:
        return "no scales"
    cat = torch.cat(all_vals)
    return (f"mean={cat.mean():.3f}, median={cat.median():.3f}, "
            f"min={cat.min():.3f}, max={cat.max():.1f}")


def main():
    print("=" * 70)
    print("Alpha-based SmoothQuant vs HiFloat8-aware Scale Search")
    print("=" * 70)

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\n[1] Loading model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("[2] Applying HiFP8 fake quantization...")
    weight_config = HiFP8FakeQuantizeConfig()
    activation_config = HiFP8FakeQuantizeConfig()
    base_model = prepare_hifp8_fake_quant(
        base_model, weight_config=weight_config, activation_config=activation_config,
    )

    print("[3] Preparing calibration data...")
    dataloader = create_calibration_dataloader(tokenizer)

    # Baseline
    print("\n[4] Baseline (no SmoothQuant)...")
    base_acc = quick_qa_accuracy(base_model, tokenizer)
    print(f"  QA Acc: {base_acc:.2%}")
    print(f"  Sanity: {sanity_check(base_model, tokenizer)}")

    # Method 1: Alpha-based SmoothQuant (best from sweep)
    print("\n" + "=" * 70)
    print("[5] Alpha-based SmoothQuant (alpha=0.3, clip=5.0)")
    print("=" * 70)
    model_alpha = copy.deepcopy(base_model)
    from quantization.smooth import calibrate_and_smooth
    calibrate_and_smooth(model_alpha, dataloader, alpha=0.3,
                         num_batches=len(dataloader), max_scale=5.0)
    alpha_acc = quick_qa_accuracy(model_alpha, tokenizer)
    print(f"  Scale stats: {scale_stats(model_alpha)}")
    print(f"  QA Acc: {alpha_acc:.2%} (delta={alpha_acc-base_acc:+.2%})")
    print(f"  Sanity: {sanity_check(model_alpha, tokenizer)}")
    del model_alpha
    torch.cuda.empty_cache()

    # Method 2: HiFloat8-aware per-layer (alpha, clip) search
    print("\n" + "=" * 70)
    print("[6] HiFloat8-aware Per-Layer (alpha, clip) Search")
    print("=" * 70)
    model_search = copy.deepcopy(base_model)
    from quantization.smooth_search import calibrate_and_search_smooth
    calibrate_and_search_smooth(model_search, dataloader,
                                num_batches=len(dataloader))
    search_acc = quick_qa_accuracy(model_search, tokenizer)
    print(f"  Scale stats: {scale_stats(model_search)}")
    print(f"  QA Acc: {search_acc:.2%} (delta={search_acc-base_acc:+.2%})")
    print(f"  Sanity: {sanity_check(model_search, tokenizer)}")

    # Also test with wider alpha range
    print("\n" + "=" * 70)
    print("[7] HiFloat8-aware Search (wider alpha + finer clip)")
    print("=" * 70)
    model_search2 = copy.deepcopy(base_model)
    calibrate_and_search_smooth(model_search2, dataloader,
                                num_batches=len(dataloader),
                                alphas=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                                        0.35, 0.4, 0.45, 0.5, 0.6, 0.7),
                                clips=(1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, None))
    search2_acc = quick_qa_accuracy(model_search2, tokenizer)
    print(f"  Scale stats: {scale_stats(model_search2)}")
    print(f"  QA Acc: {search2_acc:.2%} (delta={search2_acc-base_acc:+.2%})")
    print(f"  Sanity: {sanity_check(model_search2, tokenizer)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline (no smooth):              {base_acc:.2%}")
    print(f"  Alpha-based (a=0.3, clip=5):       {alpha_acc:.2%} ({alpha_acc-base_acc:+.2%})")
    print(f"  HiF8-aware per-layer search:       {search_acc:.2%} ({search_acc-base_acc:+.2%})")
    print(f"  HiF8-aware wider search:           {search2_acc:.2%} ({search2_acc-base_acc:+.2%})")


if __name__ == "__main__":
    main()
