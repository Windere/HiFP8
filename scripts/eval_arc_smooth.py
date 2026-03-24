#!/usr/bin/env python3
"""
ARC benchmark: no-smooth vs smooth-absorbed (a=0.675, b=0.425), both with HiFP8.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_arc_smooth.py [--limit 0]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SMOOTH_DIR = Path("/tmp/qwen3-0.6b-smooth-a0675b0425")
BASELINE_DIR = Path("/tmp/qwen3-0.6b-baseline-copy")
PORT = 8020


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset-hub", default="modelscope")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def export_smooth():
    """Export smooth-absorbed model: s = x^0.675 / w^0.425, no clip."""
    if SMOOTH_DIR.exists() and (SMOOTH_DIR / "model.safetensors").exists():
        print("[Export] Smooth model exists, skipping")
        return

    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_name = "Qwen/Qwen3-0.6B"
    print(f"[Export] Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Collect activation abs_max
    act_abs_max = {}
    hooks = []
    def make_hook(name):
        def hook(module, inp):
            x = inp[0]
            if x.ndim > 2:
                x = x.view(-1, x.shape[-1])
            cur = torch.amax(torch.abs(x), dim=0)
            if name not in act_abs_max:
                act_abs_max[name] = cur
            else:
                act_abs_max[name] = torch.maximum(act_abs_max[name], cur)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not any(
            s in name for s in ("embed_tokens", "lm_head")):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    model.eval()
    count = 0
    with torch.no_grad():
        for sample in dataset:
            text = sample["text"].strip()
            if len(text) < 50:
                continue
            tok = tokenizer(text, truncation=True, max_length=512,
                            return_tensors="pt").to(model.device)
            model(**tok)
            count += 1
            if count >= 32:
                break
    for h in hooks:
        h.remove()

    # Apply s = x^0.675 / w^0.425 (KL-optimal from search_ab_by_kl.py)
    a, b = 0.675, 0.425
    print(f"[Export] Applying smooth a={a}, b={b}...")
    def get_mod(name):
        mod = model
        for p in name.split("."):
            mod = getattr(mod, p)
        return mod

    for i in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        for ln_name, sub_names in [
            (f"{prefix}.input_layernorm",
             [f"{prefix}.self_attn.q_proj",
              f"{prefix}.self_attn.k_proj",
              f"{prefix}.self_attn.v_proj"]),
            (f"{prefix}.post_attention_layernorm",
             [f"{prefix}.mlp.gate_proj",
              f"{prefix}.mlp.up_proj"]),
        ]:
            ln = get_mod(ln_name)
            available = [n for n in sub_names if n in act_abs_max]
            if not available:
                continue
            sub_modules = [get_mod(n) for n in available]
            merged_w = torch.cat([m.weight.data for m in sub_modules], dim=0)
            x_am = act_abs_max[available[0]].float().cuda().clamp(min=1e-7)
            w_am = merged_w.float().cuda().abs().amax(dim=0).clamp(min=1e-7)
            s = (x_am.pow(a) / w_am.pow(b)).clamp(min=1e-5)
            s = s.to(device=merged_w.device, dtype=merged_w.dtype)
            for mod in sub_modules:
                mod.weight.data = mod.weight.data * s.unsqueeze(0)
            ln.weight.data = ln.weight.data / s

    SMOOTH_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SMOOTH_DIR)
    tokenizer.save_pretrained(SMOOTH_DIR)
    del model
    torch.cuda.empty_cache()
    print("[Export] Done")


def export_baseline():
    """Copy baseline model."""
    if BASELINE_DIR.exists() and (BASELINE_DIR / "model.safetensors").exists():
        print("[Baseline] Exists, skipping")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[Baseline] Copying Qwen3-0.6B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BASELINE_DIR)
    tokenizer.save_pretrained(BASELINE_DIR)
    del model
    torch.cuda.empty_cache()
    print("[Baseline] Done")


def create_metadata(model_dir):
    """Create hifp8_metadata.json for v4 server."""
    path = Path(model_dir) / "hifp8_metadata.json"
    if path.exists():
        return
    with open(path, "w") as f:
        json.dump({
            "quantization_method": "hifp8",
            "export_format": "bf16_with_buffers",
            "layers": {},
        }, f, indent=2)


def start_server(model_path, name, use_hifp8=True):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:{env.get('PYTHONPATH', '')}"
    env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    if use_hifp8:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "start_vllm_hifp8_server_v4.py"),
            "--model", str(model_path),
            "--port", str(PORT),
            "--served-model-name", name,
            "--dtype", "bfloat16",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.4",
            "--trust-remote-code",
            "--disable-log-requests",
            "--enforce-eager",
        ]
    else:
        # Standard vLLM server without HiFP8 quantization
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_path),
            "--port", str(PORT),
            "--served-model-name", name,
            "--dtype", "bfloat16",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.4",
            "--trust-remote-code",
            "--disable-log-requests",
            "--enforce-eager",
        ]
    log_path = f"/tmp/vllm_{name}.log"
    log_file = open(log_path, "w")
    print(f"  Starting {name} on :{PORT} (hifp8={use_hifp8}, log: {log_path})")
    return subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid)


def wait_server(timeout=300, name=""):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3)
            if r.status == 200:
                print(f"  {name} ready ({time.time()-t0:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"  {name} FAILED to start!")
    return False


def kill_server(proc, name=""):
    if proc and proc.poll() is None:
        print(f"  Stopping {name}...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def run_arc(model_name, limit, dataset_hub):
    work_dir = f"/tmp/arc_results_{model_name}"
    cmd = [
        sys.executable, "-m", "evalscope.run",
        "--model", model_name,
        "--api-url", f"http://localhost:{PORT}/v1",
        "--api-key", "EMPTY",
        "--datasets", "arc",
        "--dataset-hub", dataset_hub,
        "--dataset-dir", "/home/data/.cache/modelscope/datasets",
        "--work-dir", work_dir,
        "--no-timestamp",
        "--seed", "42",
        "--generation-config", '{"max_tokens": 64, "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}',
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'ao'}:{env.get('PYTHONPATH', '')}"
    env["HF_HOME"] = "/home/data/.cache/huggingface"
    env["MODELSCOPE_CACHE"] = "/home/data/.cache/modelscope"

    print(f"  evalscope ARC limit={limit}...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)

    output = result.stdout + "\n" + result.stderr
    # Extract key lines
    lines = []
    for line in output.split("\n"):
        ll = line.lower()
        if any(k in ll for k in ("accuracy", "score", "weighted", "arc", "overall")):
            lines.append(line.strip())
    return "\n".join(lines) if lines else output[-1500:]


def main():
    args = parse_args()

    # Export models
    print("=" * 60)
    print("ARC Evaluation: Baseline vs Smooth (a=0.675, b=0.425)")
    print("=" * 60)

    export_smooth()
    export_baseline()

    create_metadata(SMOOTH_DIR)
    create_metadata(BASELINE_DIR)

    # Check port
    import subprocess as sp
    fuser = sp.run(["fuser", f"{PORT}/tcp"], capture_output=True, text=True)
    if fuser.stdout.strip():
        print(f"  WARNING: Port {PORT} in use: {fuser.stdout.strip()}")

    # Three configs: (name, model_dir, use_hifp8)
    configs = [
        ("no-quant", "Qwen/Qwen3-0.6B", False),
        ("hifp8-baseline", BASELINE_DIR, True),
        ("hifp8-smooth", SMOOTH_DIR, True),
    ]

    results = {}
    for name, model_dir, use_hifp8 in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name} (hifp8={use_hifp8})")
        print(f"{'='*60}")

        proc = start_server(model_dir, name, use_hifp8=use_hifp8)
        if not wait_server(name=name):
            log = f"/tmp/vllm_{name}.log"
            if os.path.exists(log):
                with open(log) as f:
                    print(f"  Log: {f.read()[-800:]}")
            kill_server(proc, name)
            results[name] = "FAILED"
            continue

        report = run_arc(name, args.limit, args.dataset_hub)
        results[name] = report
        print(f"  Result:\n{report}")

        kill_server(proc, name)
        time.sleep(5)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, report in results.items():
        print(f"\n--- {name} ---")
        print(report)


if __name__ == "__main__":
    main()
