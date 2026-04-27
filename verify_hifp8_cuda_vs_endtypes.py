"""
CUDA-kernel level verification: repo's real HiFloat8 encode/decode CUDA
kernel vs en-dtypes (Ascend HiFloat8) round-trip.

Differs from verify_hifp8_vs_endtypes.py: that script tested a Python
re-implementation of the CPU rounding routine. This one runs the actual
CUDA kernel that ships in the repo.
"""

import math
import sys

import numpy as np
import torch
import en_dtypes as en

sys.path.insert(0, "custom_ops")
import hifp8_cuda_uint8 as hif8_cuda  # built from custom_ops/hifloat8_cuda


def cuda_roundtrip(x_np: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x_np.astype(np.float32)).cuda().contiguous()
    enc = hif8_cuda.hif8_encode_cuda(x)
    dec = hif8_cuda.hif8_decode_cuda(enc)
    return dec.cpu().numpy().astype(np.float64)


def en_roundtrip(x_np: np.ndarray) -> np.ndarray:
    return x_np.astype(np.float32).astype(en.hifloat8).astype(np.float64)


def diff(label, x, y_repo, y_en, ignore_nan=False, max_show=10):
    x = x.astype(np.float64)
    y_repo = y_repo.astype(np.float64)
    y_en = y_en.astype(np.float64)
    if ignore_nan:
        keep = ~(np.isnan(x))
        x, y_repo, y_en = x[keep], y_repo[keep], y_en[keep]
    both_nan = np.isnan(y_repo) & np.isnan(y_en)
    both_pinf = np.isposinf(y_repo) & np.isposinf(y_en)
    both_ninf = np.isneginf(y_repo) & np.isneginf(y_en)
    finite = np.isfinite(y_repo) & np.isfinite(y_en)
    eq_finite = finite & (y_repo == y_en)
    match = both_nan | both_pinf | both_ninf | eq_finite
    n_mis = int((~match).sum())
    print(f'[{label}] n={x.size:>7}  mismatches={n_mis}')
    if n_mis:
        idx = np.where(~match)[0][:max_show]
        for i in idx:
            print(f'    x={float(x[i])!r:>22}  cuda={float(y_repo[i])!r:>22}'
                  f'  en={float(y_en[i])!r:>22}')
    return n_mis


def check(label, samples, ignore_nan=False):
    arr = np.asarray(samples, dtype=np.float32)
    return diff(label, arr, cuda_roundtrip(arr), en_roundtrip(arr),
                ignore_nan=ignore_nan)


def main():
    assert torch.cuda.is_available(), "CUDA required for this verification"
    rng = np.random.default_rng(0)
    total = 0

    # 1) all 254 representable signed values (must be exact)
    pos_values = [0.0]
    for M in range(1, 8):
        pos_values.append(2.0 ** (M - 23))
    for E in range(-15, 16):
        ae = abs(E)
        mb = 3 if ae <= 3 else (2 if ae <= 7 else 1)
        for m in range(2 ** mb):
            v = (2.0 ** E) * (1.0 + m / (2.0 ** mb))
            if v >= 1.5 * 2 ** 15:
                continue
            pos_values.append(v)
    full = pos_values + [-v for v in pos_values]
    total += check('representable values (exact)', full)

    # 2) random uniform sweeps
    for hi in [1.0, 100.0, 30000.0]:
        x = (rng.random(200_000, dtype=np.float32) * 2 - 1) * np.float32(hi)
        total += check(f'uniform [-{hi}, {hi}] (n=200k)', x)

    # 3) log-uniform full range
    e = rng.uniform(math.log(2 ** -25), math.log(2 ** 17), size=500_000)
    s = rng.choice([-1.0, 1.0], size=500_000)
    total += check('log-uniform [2^-25, 2^17] (n=500k)', s * np.exp(e))

    # 4) boundaries
    pts = []
    for E in range(-22, -15):
        lo, up = 2.0 ** E, 2.0 ** (E + 1)
        mid = 1.5 * lo
        pts += [lo, mid - 1e-9, mid, mid + 1e-9, up]
    pts += [32767.0, 32768.0, 32768.5, 40959.0, 40960.0,
            40960.5, 49151.0, 49152.0, 49152.5, 65535.0]
    pts += [-v for v in pts]
    pts += [0.0, -0.0, math.inf, -math.inf]
    total += check('boundary points', pts)

    # 5) zone transitions (every 1/16 of the unit interval at each E)
    pts = []
    for E in range(-15, 16):
        for k in range(-3, 4):
            v = (2.0 ** E) * (1.0 + k * 0.0625)
            pts.append(v)
    total += check('zone transitions', pts)

    # 6) NaN handling probe (separate; we know they differ)
    print()
    print('--- NaN handling probe (informational) ---')
    nan_arr = np.array([math.nan, -math.nan], dtype=np.float32)
    print(f'  cuda(NaN) -> {cuda_roundtrip(nan_arr).tolist()}')
    print(f'  en  (NaN) -> {en_roundtrip(nan_arr).tolist()}')

    print()
    print('=' * 60)
    if total == 0:
        print('SUCCESS: real CUDA kernel matches en-dtypes on every finite sample.')
    else:
        print(f'FAILURE: {total} mismatches.')
        sys.exit(1)


if __name__ == '__main__':
    main()
