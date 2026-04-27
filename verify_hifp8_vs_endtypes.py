"""
Verify the repo's HiFP8 round-trip against en-dtypes (Ascend HiFloat8 reference).

Strategy:
  1. Port the repo's hif8_round_float_cpu algorithm to Python (1:1 from
     custom_ops/hifloat8_cuda/hif8_round_cpu.h). This is what the repo's
     `fake_quant` kernel and the encode-then-decode path both reduce to,
     since encode picks the nearest LUT value and decode reads it back.
  2. Build an en-dtypes round-trip: float32 -> hifloat8 -> float32.
  3. Compare on:
       - All 127 representable values (must be exact)
       - Random uniform in [-1, 1], [-100, 100]
       - Log-uniform sweep across the full dynamic range [2^-25, 2^17]
       - Boundary points: subnormal threshold, max threshold, NaN/Inf
  4. Report any mismatch with input value, repo result, en-dtypes result.
"""

import math
import struct
import sys

import numpy as np
import en_dtypes as en


# ---------------------------------------------------------------------------
# 1:1 Python port of hif8_round_float_cpu from
# custom_ops/hifloat8_cuda/hif8_round_cpu.h
# ---------------------------------------------------------------------------

UNDERFLOW_THRESH = 2.0 ** -23   # 1.1920928955078125e-07
MIN_DENORMAL = 2.0 ** -22       # 2.384185791015625e-07


def hif8_round_repo(x: float) -> float:
    """Port of hif8_round_float_cpu (the repo's CPU rounding routine)."""
    if math.isnan(x) or math.isinf(x) or x == 0.0:
        return x

    sign = math.copysign(1.0, x)
    ax = abs(x)

    if ax < UNDERFLOW_THRESH:
        return 0.0

    # Extract exponent
    _m, E_raw = math.frexp(ax)   # ax in [0.5, 1) * 2^E_raw  =>  ax = m * 2^E_raw
    E = E_raw - 1                # so ax in [1, 2) * 2^E

    # Denormal region
    if E <= -16:
        if E <= -23:
            return sign * MIN_DENORMAL
        lower = math.ldexp(1.0, E)
        upper = math.ldexp(1.0, E + 1)
        mid = 1.5 * lower
        result = upper if ax >= mid else lower
        if result < MIN_DENORMAL:
            return 0.0
        return sign * result

    if E > 15:
        return sign * math.inf

    abs_E = -E if E < 0 else E
    if abs_E <= 3:
        mantissa_bits = 3
    elif abs_E <= 7:
        mantissa_bits = 2
    else:
        mantissa_bits = 1

    shifted = math.ldexp(ax, mantissa_bits - E)
    rounded = math.floor(shifted + 0.5)

    carry_threshold = math.ldexp(1.0, mantissa_bits + 1)
    if rounded >= carry_threshold:
        new_E = E + 1
        if new_E > 15:
            return sign * math.inf
        return sign * math.ldexp(1.0, new_E)

    result = math.ldexp(rounded, E - mantissa_bits)
    if result >= 49152.0:
        return sign * math.inf
    return sign * result


# Vectorised wrapper
def hif8_round_repo_array(arr_f32: np.ndarray) -> np.ndarray:
    """Element-wise port; not for performance, just for correctness."""
    out = np.empty_like(arr_f32, dtype=np.float64)
    flat_in = arr_f32.astype(np.float64).reshape(-1)
    flat_out = out.reshape(-1)
    for i, v in enumerate(flat_in):
        flat_out[i] = hif8_round_repo(float(v))
    return out


# ---------------------------------------------------------------------------
# en-dtypes round-trip
# ---------------------------------------------------------------------------

def hif8_round_endtypes(arr_f32: np.ndarray) -> np.ndarray:
    """Round via en-dtypes: f32 -> hifloat8 -> f64."""
    return arr_f32.astype(en.hifloat8).astype(np.float64)


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------

def diff_arrays(label, x, y_repo, y_en, atol=0.0, rtol=0.0,
                ignore_nan_repr=True, max_show=10):
    """Compare repo vs en-dtypes element-wise. Returns mismatch count."""
    x = x.astype(np.float64)
    y_repo = y_repo.astype(np.float64)
    y_en = y_en.astype(np.float64)

    # Treat NaN==NaN, Inf same-sign==Inf
    both_nan = np.isnan(y_repo) & np.isnan(y_en)
    both_inf_pos = np.isposinf(y_repo) & np.isposinf(y_en)
    both_inf_neg = np.isneginf(y_repo) & np.isneginf(y_en)
    finite = np.isfinite(y_repo) & np.isfinite(y_en)
    if atol == 0 and rtol == 0:
        equal_finite = finite & (y_repo == y_en)
    else:
        equal_finite = finite & np.isclose(y_repo, y_en, atol=atol, rtol=rtol)

    match = both_nan | both_inf_pos | both_inf_neg | equal_finite
    mismatch = ~match

    n_total = x.size
    n_mis = int(mismatch.sum())
    print(f'[{label}] n={n_total}  mismatches={n_mis}')
    if n_mis:
        idx = np.where(mismatch)[0][:max_show]
        for i in idx:
            print(f'    x={x.flat[i]!r:>22}  repo={y_repo.flat[i]!r:>22}  en={y_en.flat[i]!r:>22}')
    return n_mis


def check_set(label, samples_f32):
    arr = np.asarray(samples_f32, dtype=np.float32)
    y_repo = hif8_round_repo_array(arr)
    y_en = hif8_round_endtypes(arr)
    return diff_arrays(label, arr, y_repo, y_en)


def main():
    rng = np.random.default_rng(0)
    total_mismatches = 0

    # 1) All 127 representable positive values + their negatives
    repo_values_pos = []
    for M in range(1, 8):
        repo_values_pos.append(2.0 ** (M - 23))
    for E in range(-15, 16):
        abs_E = abs(E)
        mb = 3 if abs_E <= 3 else (2 if abs_E <= 7 else 1)
        for m in range(2**mb):
            v = (2.0 ** E) * (1.0 + m / (2.0 ** mb))
            if v >= 1.5 * 2**15:
                continue
            repo_values_pos.append(v)
    repo_values_pos.append(0.0)
    full_repr = np.array(repo_values_pos + [-v for v in repo_values_pos],
                         dtype=np.float32)
    total_mismatches += check_set('representable values (exact)', full_repr)

    # 2) Random uniform [-1, 1], [-100, 100], [-30000, 30000]
    for hi in [1.0, 100.0, 30000.0]:
        x = (rng.random(100_000, dtype=np.float32) * 2 - 1) * np.float32(hi)
        total_mismatches += check_set(f'uniform [-{hi}, {hi}] (n=100k)', x)

    # 3) Log-uniform sweep
    log_lo, log_hi = math.log(2 ** -25), math.log(2 ** 17)
    e = rng.uniform(log_lo, log_hi, size=200_000)
    s = rng.choice([-1.0, 1.0], size=200_000)
    x = (s * np.exp(e)).astype(np.float32)
    total_mismatches += check_set('log-uniform [2^-25, 2^17] (n=200k)', x)

    # 4) Critical boundary points
    boundaries = []
    # around the underflow threshold 2^-23
    for k in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]:
        boundaries.append(2 ** (-23) * (1.0 + k * 1e-6))
    # around denormal midpoints
    for E in range(-22, -15):
        lower = 2.0 ** E
        upper = 2.0 ** (E + 1)
        mid = 1.5 * lower
        boundaries += [lower, mid - 1e-9, mid, mid + 1e-9, upper]
    # around max
    boundaries += [32767.0, 32768.0, 32768.5, 40959.0, 40960.0,
                   40961.0, 49151.0, 49152.0, 49152.5, 65535.0]
    # negatives
    boundaries += [-v for v in boundaries]
    # specials
    boundaries += [0.0, -0.0, math.inf, -math.inf, math.nan]
    total_mismatches += check_set('boundary points', boundaries)

    # 5) Quarter-step sweep around mantissa-zone transitions
    transition_pts = []
    for E in range(-15, 16):
        for k in range(-3, 4):
            v = (2.0 ** E) * (1.0 + k * 0.0625)
            transition_pts.append(v)
    total_mismatches += check_set('zone transitions', transition_pts)

    print()
    print('=' * 60)
    if total_mismatches == 0:
        print('SUCCESS: repo HiFP8 round-trip matches en-dtypes on every sample.')
    else:
        print(f'FAILURE: {total_mismatches} total mismatches.')
        sys.exit(1)


if __name__ == '__main__':
    main()
