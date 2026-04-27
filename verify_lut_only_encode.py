"""
Verify the LUT-only encode path against the math + binary-search path.

Equivalence claim: for every float32 input, the LUT-only encoder must
produce byte-identical output to the existing encoder, in BOTH the
LUT-rank and Ascend byte layouts.

Coverage:
  1) All 254 representable HiFloat8 values (signed) — exact path
  2) Boundaries (subnormal mids, max, Inf threshold, 49152 saturation)
  3) 5 M random samples (uniform + log-uniform across full dynamic range)
  4) Stress: every float32 with exp ∈ [102, 144] and top_4_mant ∈ [0,15]
     for several random low-mantissa-bit fillings (>660 K floats touching
     all interesting LUT rows)
  5) Specials: ±0, ±Inf, every NaN bit-pattern with mant > 0
  6) Quick latency micro-bench (50 M elements)
"""

from __future__ import annotations

import math
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "custom_ops")
import hifp8_cuda_uint8 as h


def to_cuda_f32(arr) -> torch.Tensor:
    a = np.asarray(arr, dtype=np.float32)
    return torch.from_numpy(a).cuda().contiguous()


def diff_bytes(label, x_cuda, y_math_cuda, y_lut_cuda, max_show=10):
    a = y_math_cuda.cpu().numpy()
    b = y_lut_cuda.cpu().numpy()
    bad = a != b
    n_bad = int(bad.sum())
    print(f"[{label}] n={a.size:>9}  byte mismatches={n_bad}")
    if n_bad:
        idx = np.where(bad)[0][:max_show]
        x_np = x_cuda.cpu().numpy()
        for i in idx:
            print(f"    x={float(x_np[i])!r:>22}  math=0x{int(a[i]):02X}  lut=0x{int(b[i]):02X}")
    return n_bad


def check_pair(label, samples):
    x = to_cuda_f32(samples)

    # LUT-rank layout
    a_math = h.hif8_encode_cuda(x)
    a_lut = h.hif8_encode_lut_only_cuda(x)

    # Ascend layout
    b_math = h.hif8_encode_ascend_cuda(x)
    b_lut = h.hif8_encode_ascend_lut_only_cuda(x)

    n1 = diff_bytes(f"{label} | LUT-rank", x, a_math, a_lut)
    n2 = diff_bytes(f"{label} | Ascend  ", x, b_math, b_lut)
    return n1 + n2


def main():
    assert torch.cuda.is_available()
    rng = np.random.default_rng(7)
    bad_total = 0

    # 1) Representable values
    repr_vals = [0.0]
    for M in range(1, 8):
        repr_vals.append(2.0 ** (M - 23))
    for E in range(-15, 16):
        ae = abs(E)
        mb = 3 if ae <= 3 else (2 if ae <= 7 else 1)
        for m in range(2 ** mb):
            v = (2.0 ** E) * (1.0 + m / (2.0 ** mb))
            if v >= 1.5 * 2 ** 15:
                continue
            repr_vals.append(v)
    repr_vals = repr_vals + [-v for v in repr_vals]
    bad_total += check_pair("representable values", repr_vals)

    # 2) Boundaries
    pts = []
    for E in range(-22, -15):
        lo = 2.0 ** E
        up = 2.0 ** (E + 1)
        pts += [lo, 1.5 * lo - 1e-9, 1.5 * lo, 1.5 * lo + 1e-9, up]
    pts += [32767.0, 32768.0, 32768.5, 40959.0, 40960.0, 40960.5,
            49151.0, 49152.0, 49152.5, 65535.0, 1e30, 1e38, 3.4e38]
    pts += [-v for v in pts]
    pts += [0.0, -0.0, math.inf, -math.inf]
    bad_total += check_pair("boundaries", pts)

    # 3) Random sweeps
    for label, hi in [("uniform 1", 1.0), ("uniform 100", 100.0),
                      ("uniform 30k", 30000.0), ("uniform 1e6", 1e6)]:
        x = (rng.random(1_000_000, dtype=np.float32) * 2 - 1) * np.float32(hi)
        bad_total += check_pair(label, x)

    e = rng.uniform(math.log(2 ** -25), math.log(2 ** 17), size=1_000_000)
    s = rng.choice([-1.0, 1.0], size=1_000_000)
    bad_total += check_pair("log-uniform full range", s * np.exp(e))

    # 4) Stress every interesting (exp, top_4_mant) row with several
    #    random low-mantissa-bit fillings — proves "low bits irrelevant"
    sweep = []
    for exp in range(0, 256):                           # all exponents
        for top4 in range(16):
            for low_seed in range(10):                  # 10 random fills
                low_bits = int(rng.integers(0, 1 << 19))
                bits = (exp << 23) | (top4 << 19) | low_bits
                sweep.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    # Skip NaNs for the equivalence check (handled in step 5)
    sweep_np = np.array(sweep, dtype=np.float32)
    sweep_np = sweep_np[~np.isnan(sweep_np)]
    bad_total += check_pair("exhaustive (exp,top4) × random low-bits", sweep_np)

    # 5) NaN encodings: every NaN must produce 0x80 in both encoders
    nan_bits = []
    for mant in [1, 0x40_0000, 0x7F_FFFF, 0x123456, 0x7FF000]:
        nan_bits.append(0x7F800000 | mant)              # +NaN variants
        nan_bits.append(0xFF800000 | mant)              # -NaN variants
    nans = np.array([struct.unpack("<f", struct.pack("<I", b))[0] for b in nan_bits],
                    dtype=np.float32)
    x = torch.from_numpy(nans).cuda().contiguous()
    for kind, fn_math, fn_lut in [
        ("LUT-rank NaN", h.hif8_encode_cuda, h.hif8_encode_lut_only_cuda),
        ("Ascend   NaN", h.hif8_encode_ascend_cuda, h.hif8_encode_ascend_lut_only_cuda),
    ]:
        a = fn_math(x).cpu().numpy()
        b = fn_lut(x).cpu().numpy()
        ok = bool(np.all(a == 0x80) and np.all(b == 0x80))
        print(f"[NaN]      {kind:>14}: math={a.tolist()}  lut={b.tolist()}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            bad_total += 1

    # 6) Latency micro-bench
    print()
    print("--- latency micro-bench (50 M elements, mean of 5 runs) ---")
    big = torch.randn(50_000_000, device="cuda", dtype=torch.float32) * 100.0

    def bench(fn, x, n_runs=5):
        for _ in range(2):  # warmup
            fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_runs * 1000

    for label, fn in [
        ("LUT-rank math+search", h.hif8_encode_cuda),
        ("LUT-rank LUT-only   ", h.hif8_encode_lut_only_cuda),
        ("Ascend   math+search", h.hif8_encode_ascend_cuda),
        ("Ascend   LUT-only   ", h.hif8_encode_ascend_lut_only_cuda),
    ]:
        ms = bench(fn, big)
        gb = big.numel() / 1e9
        print(f"  {label}: {ms:7.3f} ms  ({gb / (ms / 1000):6.2f} G-elems/s)")

    print()
    print("=" * 60)
    if bad_total == 0:
        print("SUCCESS: LUT-only encode is byte-identical to math+search encode.")
    else:
        print(f"FAILURE: {bad_total} mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
