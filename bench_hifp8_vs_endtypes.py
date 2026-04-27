"""
Head-to-head accuracy + latency bench: this repo's HiFP8 CUDA kernel vs
the reference `en-dtypes` PyPI package.

Coverage:
  Accuracy   — round-trip on a 1M-sample float32 sweep:
               (1) byte-equality between repo's Ascend-format encode
                   and en-dtypes' encode
               (2) decode equivalence on every 256 byte pattern
  Latency    — encode-only throughput on a 50M-sample input:
               * en-dtypes CPU encode  (numpy astype → hifloat8)
               * our CUDA kernel       (LUT-rank layout)
               * our CUDA kernel       (Ascend layout)
               * our CUDA kernel       (LUT-only, branchless)

Designed to be runnable in any env that has PyTorch+CUDA + en-dtypes +
the compiled HiFP8 extension. Self-contained, no model weights needed.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import en_dtypes as en

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "custom_ops"))
import hifp8_cuda_uint8 as hif8  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench_cuda(fn, x_cuda, n_runs=10, n_warmup=3):
    for _ in range(n_warmup):
        fn(x_cuda)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(x_cuda)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0  # ms


def _bench_cpu(fn, x_np, n_runs=3, n_warmup=1):
    for _ in range(n_warmup):
        fn(x_np)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(x_np)
    return (time.perf_counter() - t0) / n_runs * 1000.0  # ms


# ---------------------------------------------------------------------------
# Part A — accuracy
# ---------------------------------------------------------------------------

def check_byte_equality(n_samples=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    # log-uniform across full HiFloat8 dynamic range, signed
    e = rng.uniform(math.log(2 ** -25), math.log(2 ** 17), size=n_samples)
    s = rng.choice([-1.0, 1.0], size=n_samples)
    x_f32 = (s * np.exp(e)).astype(np.float32)

    x_cuda = torch.from_numpy(x_f32).cuda().contiguous()
    cuda_bytes = hif8.hif8_encode_ascend_cuda(x_cuda).cpu().numpy()
    en_bytes = x_f32.astype(en.hifloat8).view(np.uint8)

    diff = (cuda_bytes != en_bytes).sum()
    print(f"[A.1] Encode byte equality (Ascend layout, n={n_samples:,})")
    print(f"      mismatches = {diff}/{n_samples}  → {'OK' if diff == 0 else 'FAIL'}")
    return diff == 0


def check_decode_equality():
    bytes_all = np.arange(256, dtype=np.uint8)
    cuda_dec = hif8.hif8_decode_ascend_cuda(
        torch.from_numpy(bytes_all).cuda().contiguous()
    ).cpu().numpy().astype(np.float64)
    en_dec = bytes_all.view(en.hifloat8).astype(np.float64)

    bad = 0
    for b in range(256):
        a, e_ = cuda_dec[b], en_dec[b]
        if (np.isnan(a) and np.isnan(e_)) or a == e_ or \
           (np.isinf(a) and np.isinf(e_) and np.sign(a) == np.sign(e_)):
            continue
        bad += 1
    print(f"[A.2] Decode equality (all 256 byte patterns)")
    print(f"      mismatches = {bad}/256  → {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0


# ---------------------------------------------------------------------------
# Part B — latency
# ---------------------------------------------------------------------------

def latency_bench(n_samples=50_000_000, seed=1):
    rng = np.random.default_rng(seed)
    x_f32 = (rng.standard_normal(n_samples) * 100.0).astype(np.float32)
    x_cuda = torch.from_numpy(x_f32).cuda().contiguous()
    n_giga = n_samples / 1e9

    print(f"\n[B] Encode-only latency on {n_samples / 1e6:.0f} M float32 samples")
    print(f"    (CPU = single-threaded numpy.astype; CUDA = single H2H sync per run)")
    print(f"    {'path':<38}  {'ms/run':>10}  {'G-elems/s':>11}")
    print(f"    {'-'*38}  {'-'*10}  {'-'*11}")

    # CPU baseline (en-dtypes)
    ms_cpu = _bench_cpu(lambda a: a.astype(en.hifloat8), x_f32, n_runs=3)
    print(f"    {'en-dtypes CPU astype':<38}  {ms_cpu:>10.2f}  "
          f"{n_giga / (ms_cpu / 1000):>11.3f}")

    # CUDA: LUT-rank math + binary search
    ms_lr_math = _bench_cuda(lambda x: hif8.hif8_encode_cuda(x), x_cuda)
    print(f"    {'CUDA  LUT-rank  (math + binsearch)':<38}  {ms_lr_math:>10.3f}  "
          f"{n_giga / (ms_lr_math / 1000):>11.2f}")

    # CUDA: Ascend math + remap
    ms_as_math = _bench_cuda(lambda x: hif8.hif8_encode_ascend_cuda(x), x_cuda)
    print(f"    {'CUDA  Ascend    (math + binsearch + remap)':<38}  {ms_as_math:>10.3f}  "
          f"{n_giga / (ms_as_math / 1000):>11.2f}")

    # CUDA: LUT-only (Ascend layout)
    ms_as_lut = _bench_cuda(lambda x: hif8.hif8_encode_ascend_lut_only_cuda(x), x_cuda)
    print(f"    {'CUDA  Ascend    (LUT-only, branchless)':<38}  {ms_as_lut:>10.3f}  "
          f"{n_giga / (ms_as_lut / 1000):>11.2f}")

    # CUDA: LUT-only (LUT-rank layout)
    ms_lr_lut = _bench_cuda(lambda x: hif8.hif8_encode_lut_only_cuda(x), x_cuda)
    print(f"    {'CUDA  LUT-rank  (LUT-only, branchless)':<38}  {ms_lr_lut:>10.3f}  "
          f"{n_giga / (ms_lr_lut / 1000):>11.2f}")

    print()
    print(f"    Speedup vs en-dtypes CPU (best CUDA path):")
    best = min(ms_lr_math, ms_as_math, ms_lr_lut, ms_as_lut)
    print(f"      {ms_cpu / best:>7.0f}× faster   ({ms_cpu:.0f} ms → {best:.2f} ms)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"Device: {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)})")
    print(f"torch  : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"en-dtypes version: {getattr(en, '__version__', 'present')}")
    print()

    print("=" * 60)
    print("Part A — accuracy")
    print("=" * 60)
    ok_enc = check_byte_equality()
    ok_dec = check_decode_equality()
    if not (ok_enc and ok_dec):
        print("ACCURACY FAILED — aborting before bench")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Part B — latency")
    print("=" * 60)
    latency_bench()


if __name__ == "__main__":
    main()
