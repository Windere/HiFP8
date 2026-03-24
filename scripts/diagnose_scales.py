#!/usr/bin/env python3
"""
Diagnose why SmoothQuant scales are large for Qwen3-0.6B.
Analyzes per-channel weight and activation distributions.
Also tests scale clipping strategies.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/diagnose_scales.py
"""
import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import (
    HiFP8FakeQuantizeConfig,
    prepare_hifp8_fake_quant,
)
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from quantization.smooth import compute_smooth_scale


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


def collect_activation_stats(model, dataloader, num_batches=32):
    """Collect per-channel activation abs max without applying smoothing."""
    activation_stats = {}
    hooks = []

    def make_hook(name):
        def hook(module, input_tuple):
            x = input_tuple[0]
            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])
            cur_abs_max = torch.amax(torch.abs(x), dim=0)
            if name not in activation_stats:
                activation_stats[name] = cur_abs_max
            else:
                activation_stats[name] = torch.maximum(activation_stats[name], cur_abs_max)
        return hook

    skip_names = ("embed_tokens", "lm_head")
    for name, module in model.named_modules():
        if isinstance(module, HiFP8FakeQuantizedLinear):
            if any(s in name for s in skip_names):
                continue
            hooks.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            model(**batch)

    for h in hooks:
        h.remove()

    return activation_stats


def quick_qa_accuracy(model, tokenizer, num_questions=30):
    """Quick MCQ accuracy test."""
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


def apply_smooth_with_clipping(model, activation_stats, alpha, max_scale):
    """Apply SmoothQuant with scale clipping."""
    from quantization.smooth import apply_smooth_scale

    smooth_scales = {}
    skip_names = ("embed_tokens", "lm_head")
    for name, module in model.named_modules():
        if not isinstance(module, HiFP8FakeQuantizedLinear):
            continue
        if any(s in name for s in skip_names):
            continue
        if name not in activation_stats:
            continue

        x_abs_max = activation_stats[name]
        scale = compute_smooth_scale(x_abs_max, module.weight.data, alpha)

        # Clip scale to [1/max_scale, max_scale]
        scale = torch.clamp(scale, min=1.0 / max_scale, max=max_scale)
        smooth_scales[name] = scale

    apply_smooth_scale(model, smooth_scales)
    return smooth_scales


def main():
    print("=" * 70)
    print("SmoothQuant Scale Diagnosis - Qwen3-0.6B")
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

    # ---- Part 1: Diagnose weight and activation distributions ----
    print("\n[4] Collecting activation stats...")
    activation_stats = collect_activation_stats(base_model, dataloader)

    print("\n[5] Analyzing weight and activation distributions...")
    sample_layers = []
    for name, module in base_model.named_modules():
        if isinstance(module, HiFP8FakeQuantizedLinear) and name in activation_stats:
            sample_layers.append((name, module))

    all_w_max = []
    all_x_max = []
    for name, module in sample_layers:
        w = module.weight.data.float()
        w_max = torch.amax(torch.abs(w), dim=0)
        x_max = activation_stats[name].float()
        all_w_max.append(w_max.cpu())
        all_x_max.append(x_max.cpu())

    w_cat = torch.cat(all_w_max)
    x_cat = torch.cat(all_x_max)

    print(f"\n  Weight per-channel |w|_max distribution:")
    print(f"    mean={w_cat.mean():.4f}, median={w_cat.median():.4f}, "
          f"min={w_cat.min():.4f}, max={w_cat.max():.4f}")
    print(f"    <0.1: {(w_cat < 0.1).float().mean()*100:.1f}%, "
          f"<0.5: {(w_cat < 0.5).float().mean()*100:.1f}%, "
          f"<1.0: {(w_cat < 1.0).float().mean()*100:.1f}%")

    print(f"\n  Activation per-channel |x|_max distribution:")
    print(f"    mean={x_cat.mean():.4f}, median={x_cat.median():.4f}, "
          f"min={x_cat.min():.4f}, max={x_cat.max():.4f}")
    print(f"    >10: {(x_cat > 10).float().mean()*100:.1f}%, "
          f">100: {(x_cat > 100).float().mean()*100:.1f}%, "
          f">1000: {(x_cat > 1000).float().mean()*100:.1f}%")

    alpha = 0.1
    s_raw = (x_cat + 1e-7).pow(alpha) / (w_cat + 1e-7).pow(1 - alpha)
    print(f"\n  Raw scale (alpha={alpha}):")
    print(f"    mean={s_raw.mean():.3f}, median={s_raw.median():.3f}, max={s_raw.max():.1f}")
    print(f"    x^alpha  mean={(x_cat+1e-7).pow(alpha).mean():.3f}")
    print(f"    w^(1-a)  mean={(w_cat+1e-7).pow(1-alpha).mean():.3f}")
    print(f"    => ratio ~ {(x_cat+1e-7).pow(alpha).mean() / (w_cat+1e-7).pow(1-alpha).mean():.3f}")

    # ---- Part 2: Test scale clipping strategies ----
    print("\n" + "=" * 70)
    print("Scale Clipping Experiment (alpha=0.1)")
    print("=" * 70)

    base_acc = quick_qa_accuracy(base_model, tokenizer)
    print(f"  Baseline QA Acc: {base_acc:.2%}")

    best_alpha = 0.1
    clip_values = [1.5, 2.0, 3.0, 5.0, 10.0, 50.0, None]

    for clip_max in clip_values:
        model = copy.deepcopy(base_model)
        label = f"clip={clip_max}" if clip_max else "no-clip"

        if clip_max:
            scales = apply_smooth_with_clipping(
                model, activation_stats, alpha=best_alpha, max_scale=clip_max,
            )
        else:
            from quantization.smooth import calibrate_and_smooth
            scales = calibrate_and_smooth(
                model, dataloader, alpha=best_alpha, num_batches=len(dataloader),
            )

        all_s = []
        for name, module in model.named_modules():
            if isinstance(module, HiFP8FakeQuantizedLinear) and module.smooth_scale is not None:
                all_s.append(module.smooth_scale.detach().cpu().float())
        if all_s:
            s = torch.cat(all_s)
            s_info = f"mean={s.mean():.2f}, max={s.max():.1f}, med={s.median():.2f}"
        else:
            s_info = "N/A"

        acc = quick_qa_accuracy(model, tokenizer)
        print(f"  alpha={best_alpha}, {label:>12}: Acc={acc:.2%} (d={acc-base_acc:+.2%})  {s_info}")

        del model
        torch.cuda.empty_cache()

    # ---- Part 3: Grid search alpha x clip ----
    print("\n" + "=" * 70)
    print("Grid Search: alpha x clip_max")
    print("=" * 70)

    best_combos = []
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        for clip_max in [1.5, 2.0, 3.0, 5.0, 10.0]:
            model = copy.deepcopy(base_model)
            scales = apply_smooth_with_clipping(
                model, activation_stats, alpha=alpha, max_scale=clip_max,
            )
            acc = quick_qa_accuracy(model, tokenizer)
            best_combos.append((alpha, clip_max, acc))
            del model
            torch.cuda.empty_cache()

    best_combos.sort(key=lambda x: -x[2])
    print(f"  {'alpha':>6} | {'clip':>5} | {'QA Acc':>8} | {'dAcc':>8}")
    print("  " + "-" * 40)
    for alpha, clip, acc in best_combos[:15]:
        print(f"  {alpha:>6.2f} | {clip:>5.1f} | {acc:>8.2%} | {acc-base_acc:>+8.2%}")

    print(f"\n  Best: alpha={best_combos[0][0]}, clip={best_combos[0][1]}, "
          f"Acc={best_combos[0][2]:.2%}")


if __name__ == "__main__":
    main()
