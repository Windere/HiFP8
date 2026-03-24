"""
Export a SmoothQuant model with smooth scales absorbed into LayerNorm + weights.

For merged layers (qkv_proj, gate_up_proj), the smooth scale is computed on the
merged weight and absorbed:
  - W *= s  (bake into weight)
  - preceding_layernorm.weight /= s  (LayerNorm naturally outputs x/s)

For non-merged layers (o_proj, down_proj), smooth is skipped since their input
doesn't come from a LayerNorm.

Output: a standard HuggingFace model that can be loaded by any framework.
SmoothQuant is fully transparent -- no separate smooth_scale tensors needed.
"""

import argparse
import sys
sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.smooth import compute_smooth_scale
from datasets import load_dataset


def collect_activation_stats(model, tokenizer, num_batches=32, max_length=512):
    """Collect per-channel activation abs max for each Linear layer."""
    from torch import nn
    stats = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp):
            x = inp[0]
            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])
            cur = torch.amax(torch.abs(x), dim=0)
            if name not in stats:
                stats[name] = cur
            else:
                stats[name] = torch.maximum(stats[name], cur)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(s in name for s in ("embed_tokens", "lm_head")):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    model.eval()
    count = 0
    with torch.no_grad():
        for sample in dataset:
            text = sample["text"].strip()
            if len(text) < 50:
                continue
            tok = tokenizer(text, truncation=True, max_length=max_length,
                            return_tensors="pt").to(model.device)
            model(**tok)
            count += 1
            if count >= num_batches:
                break

    for h in hooks:
        h.remove()
    print(f"[Calibration] Collected stats for {len(stats)} layers over {count} batches")
    return stats


def apply_smooth_absorbed(model, act_stats, alpha=0.3, max_scale=5.0):
    """
    Apply SmoothQuant by absorbing scales into weights + LayerNorm.

    Only applies to merged layer groups (qkv, gate_up) where the preceding
    LayerNorm can absorb 1/s.
    """
    num_layers = model.config.num_hidden_layers

    total_groups = 0
    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # Group 1: input_layernorm -> q_proj, k_proj, v_proj
        ln_name = f"{prefix}.input_layernorm"
        sub_names = [f"{prefix}.self_attn.q_proj",
                     f"{prefix}.self_attn.k_proj",
                     f"{prefix}.self_attn.v_proj"]
        _apply_group(model, act_stats, ln_name, sub_names, alpha, max_scale)
        total_groups += 1

        # Group 2: post_attention_layernorm -> gate_proj, up_proj
        ln_name = f"{prefix}.post_attention_layernorm"
        sub_names = [f"{prefix}.mlp.gate_proj",
                     f"{prefix}.mlp.up_proj"]
        _apply_group(model, act_stats, ln_name, sub_names, alpha, max_scale)
        total_groups += 1

    print(f"[SmoothAbsorb] Applied smooth to {total_groups} merge groups "
          f"(alpha={alpha}, max_scale={max_scale})")


def _apply_group(model, act_stats, ln_name, sub_names, alpha, max_scale):
    """Apply smooth to one merge group."""
    def get_mod(name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p)
        return mod

    ln = get_mod(ln_name)

    available = [n for n in sub_names if n in act_stats]
    if not available:
        return

    # Activation stats from first sub-layer (all share same input from LayerNorm)
    x_abs_max = act_stats[available[0]]

    # Merge weights: concatenate along output dim
    sub_modules = [get_mod(n) for n in available]
    merged_w = torch.cat([m.weight.data for m in sub_modules], dim=0)

    # Compute unified smooth scale on merged weight
    s = compute_smooth_scale(x_abs_max, merged_w, alpha=alpha)
    if max_scale is not None:
        s = torch.clamp(s, min=1.0 / max_scale, max=max_scale)
    s = s.to(device=merged_w.device, dtype=merged_w.dtype)

    # Apply to weights: W *= s
    for mod in sub_modules:
        mod.weight.data = mod.weight.data * s.unsqueeze(0)

    # Absorb into LayerNorm: gamma /= s
    # RMSNorm: output = x / rms(x) * gamma
    # After: output = x / rms(x) * (gamma / s) = original_output / s
    ln.weight.data = ln.weight.data / s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="/tmp/qwen3-0.6b-smooth-absorbed")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-scale", type=float, default=5.0)
    parser.add_argument("--num-batches", type=int, default=32)
    args = parser.parse_args()

    print(f"[1] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[2] Collecting activation statistics...")
    act_stats = collect_activation_stats(model, tokenizer, num_batches=args.num_batches)

    print(f"[3] Applying SmoothQuant (alpha={args.alpha}, max_scale={args.max_scale})")
    apply_smooth_absorbed(model, act_stats, alpha=args.alpha, max_scale=args.max_scale)

    print(f"[4] Saving to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Quick sanity check
    print(f"[5] Sanity check...")
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"  Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
    print(f"\nDone! Model saved to {args.output}")
    print(f"Use with vLLM plugin for HiFloat8 quantization at runtime.")


if __name__ == "__main__":
    main()
