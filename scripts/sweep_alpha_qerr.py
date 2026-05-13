#!/usr/bin/env python
"""
Sweep SmoothQuant alpha (and optionally scale_factor) and measure direct
quantization error vs the FP BF16 reference.

Why not ARC accuracy: dataset metrics on small calibration sets risk
overfitting and have high variance at limit=100. Quantization error (KL
divergence on logits, cosine similarity / relative RMSE on the last hidden
state) is a direct mathematical signal of fidelity to the FP forward pass.

Runs all configs in-process (no vLLM, no server). Held-out prompts are
disjoint from calibration prompts.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
_OUTPUT_ROOT = Path.home() / "outputs" / "HiFP8"
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization.hifp8_linear import prepare_hifp8_fake_quant
from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.smooth import calibrate_and_smooth
from quantization.smooth_fuse import fuse_smooth_into_norms, rollback_unfoldable_smooths


CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, artificial intelligence made significant advances in reasoning.",
    "Quantization reduces model size by representing weights with fewer bits.",
    "The transformer architecture revolutionized natural language processing.",
    "Large language models are trained on vast amounts of text data.",
    "Scientific research requires careful experimental design and analysis.",
]

# Held-out prompts — different domain mix from calibration to avoid
# overfitting the smooth scales.
HELD_OUT_PROMPTS = [
    "The Eiffel Tower, located in Paris, France, was constructed between 1887 and 1889 as the entrance arch to the 1889 World's Fair held to celebrate the centennial of the French Revolution.",
    "Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy stored in glucose, releasing oxygen as a byproduct.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n# Computes the nth Fibonacci number using simple recursion.",
    "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse equals the sum of squares of the other two sides, written as a^2 + b^2 = c^2.",
    "I think the best way to learn a new programming language is to build something practical with it rather than just reading documentation passively.",
    "Climate change is one of the most pressing issues of our time. Rising global temperatures, melting ice caps, and increasingly severe weather events are symptoms of a warming planet.",
    "If a train leaves the station at 9 AM traveling 60 mph and another leaves at 10 AM traveling 80 mph in the same direction, the second train will catch up at noon.",
    "Mitochondria are membrane-bound organelles found in the cells of most eukaryotic organisms, generating most of the cell's supply of adenosine triphosphate used as chemical energy.",
]


@torch.no_grad()
def compute_outputs(model, tokenizer, prompts, device="cuda"):
    """Run model and collect logits + last hidden state per prompt (CPU float32)."""
    outs = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        out = model(**inputs, output_hidden_states=True)
        outs.append({
            "logits": out.logits.float().cpu(),
            "hidden": out.hidden_states[-1].float().cpu(),
        })
    return outs


def measure_error(qnt_outs, ref_outs):
    """Per-prompt KL(ref||qnt) on logits, cos sim + rel RMSE on hidden state."""
    kls, cos_sims, rel_rmses = [], [], []
    for q, r in zip(qnt_outs, ref_outs):
        ref_logp = torch.log_softmax(r["logits"], dim=-1)
        q_logp = torch.log_softmax(q["logits"], dim=-1)
        ref_p = ref_logp.exp()
        kl = (ref_p * (ref_logp - q_logp)).sum(-1).mean().item()
        kls.append(kl)

        cos = torch.nn.functional.cosine_similarity(
            r["hidden"].view(-1, r["hidden"].shape[-1]),
            q["hidden"].view(-1, q["hidden"].shape[-1]),
            dim=-1,
        ).mean().item()
        cos_sims.append(cos)

        diff_sq = (r["hidden"] - q["hidden"]).pow(2).mean()
        ref_sq = r["hidden"].pow(2).mean()
        rel_rmses.append((diff_sq / ref_sq).sqrt().item())

    return {
        "kl_logits": sum(kls) / len(kls),
        "cos_hidden": sum(cos_sims) / len(cos_sims),
        "rel_rmse_hidden": sum(rel_rmses) / len(rel_rmses),
    }


def _set_inference_mode(model):
    """Equivalent to model.eval() — put model in inference mode."""
    model.train(False)


def build_quantized(model_path, smooth_alpha, scale_factor, weight_only=False,
                    device="cuda"):
    """Build a fresh quantized model. smooth_alpha=None → skip SmoothQuant."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    weight_cfg = HiFP8FakeQuantizeConfig(scale_factor=scale_factor)
    act_cfg = None if weight_only else HiFP8FakeQuantizeConfig(scale_factor=scale_factor)
    prepare_hifp8_fake_quant(model, weight_config=weight_cfg, activation_config=act_cfg)

    if smooth_alpha is not None:
        encoded = [tokenizer(p, return_tensors="pt") for p in CALIBRATION_PROMPTS]

        class _Loader:
            def __iter__(self):
                for batch in encoded:
                    yield {k: v.to(device) for k, v in batch.items()}

        calibrate_and_smooth(model, _Loader(), alpha=smooth_alpha,
                             num_batches=len(CALIBRATION_PROMPTS))
        fuse_smooth_into_norms(model)
        rollback_unfoldable_smooths(model)

    # Final calibration pass — gives any static-mode quantizers realistic ranges.
    model.train(True)
    with torch.no_grad():
        for p in CALIBRATION_PROMPTS:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            model(**inputs)
    _set_inference_mode(model)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/kailong/models/Qwen3-0.6B")
    parser.add_argument("--alphas", default="none,0.1,0.3,0.5,0.7,0.9",
                        help="Comma-separated; 'none' = skip SmoothQuant entirely")
    parser.add_argument("--scale-factors", default="1.0",
                        help="Comma-separated scale_factor values (one row per alpha x sf)")
    parser.add_argument("--include-weight-only", action="store_true",
                        help="Also measure w8a16 baseline (weight-only quantization)")
    parser.add_argument("--output", default=str(_OUTPUT_ROOT / "sweep_alpha_qerr.json"))
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"[Sweep] Model: {args.model}")
    print(f"[Sweep] GPU: {args.gpu}")
    print(f"[Sweep] Alphas: {args.alphas}")
    print(f"[Sweep] Scale factors: {args.scale_factors}")

    # --- Reference: BF16, no quantization ---
    print("\n[Ref] Loading reference (BF16) model...")
    t0 = time.time()
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True,
    )
    _set_inference_mode(ref_model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[Ref] Computing outputs on {len(HELD_OUT_PROMPTS)} held-out prompts...")
    ref_outs = compute_outputs(ref_model, tokenizer, HELD_OUT_PROMPTS)
    print(f"[Ref] Done in {time.time()-t0:.1f}s. Freeing reference model.")
    del ref_model
    torch.cuda.empty_cache()

    # --- Build config list ---
    alpha_strs = args.alphas.split(",")
    sf_strs = args.scale_factors.split(",")
    configs = []

    if args.include_weight_only:
        configs.append(("w8a16_weight_only", None, 1.0, True))

    for sf_s in sf_strs:
        sf = float(sf_s)
        for a_s in alpha_strs:
            alpha = None if a_s == "none" else float(a_s)
            label = ("no_sq" if alpha is None else f"a{alpha}") + f"_sf{sf}"
            configs.append((label, alpha, sf, False))

    # --- Sweep ---
    results = {}
    for label, alpha, sf, weight_only in configs:
        print(f"\n=== {label}: alpha={alpha} scale_factor={sf} weight_only={weight_only} ===")
        torch.cuda.empty_cache()
        t0 = time.time()
        try:
            qnt_model, tok = build_quantized(args.model, alpha, sf,
                                              weight_only=weight_only)
            qnt_outs = compute_outputs(qnt_model, tok, HELD_OUT_PROMPTS)
            err = measure_error(qnt_outs, ref_outs)
            err["build_seconds"] = time.time() - t0
            results[label] = err
            print(f"  KL={err['kl_logits']:.5f}  "
                  f"cos_h={err['cos_hidden']:.5f}  "
                  f"rel_rmse_h={err['rel_rmse_hidden']:.5f}  "
                  f"({err['build_seconds']:.1f}s)")
            del qnt_model
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
            results[label] = {"error": str(e)}
        torch.cuda.empty_cache()

    # --- Summary ---
    print("\n=== Summary (lower KL/rel_rmse, higher cos = better) ===")
    print(f"{'config':>22} {'KL':>10} {'cos_hidden':>12} {'rel_rmse_h':>12}")
    for label, r in results.items():
        if "error" in r:
            print(f"{label:>22}  ERROR: {r['error']}")
            continue
        print(f"{label:>22} {r['kl_logits']:>10.5f} "
              f"{r['cos_hidden']:>12.5f} {r['rel_rmse_hidden']:>12.5f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "calibration_prompts": CALIBRATION_PROMPTS,
            "held_out_prompts": HELD_OUT_PROMPTS,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
