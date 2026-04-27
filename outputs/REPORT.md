# HiFP8 QAT Pipeline — Qwen3-0.6B Four-way Eval
_Generated 2026-04-27 08:07:45_

## Configurations

| label | model path | quant_method |
| ----- | ---------- | ------------ |
| bf16 | `Qwen/Qwen3-0.6B` | — |
| ptq | `/home/kailong/Mem/workspace/quant-llm/HIFP8-VAL/outputs/qwen3_ptq_weightonly` | — |
| ptq_smooth | `/home/kailong/Mem/workspace/quant-llm/HIFP8-VAL/outputs/qwen3_ptq_smooth_fused` | — |
| qat | `/home/kailong/Mem/workspace/quant-llm/HIFP8-VAL/outputs/qwen3_qat` | — |

## Results

| benchmark      | bf16   | ptq    | ptq_smooth | qat    | Δ smooth-ptq | Δ qat-bf16 |
| -------------- | ------ | ------ | ---------- | ------ | ------------ | ---------- |
| arc-easy       |  0.775 |  0.760 |      0.795 |  0.645 |       +0.035 |     -0.130 |
| arc-challenge  |  0.630 |  0.670 |      0.635 |  0.455 |       -0.035 |     -0.175 |
| arc            |  0.703 |  0.715 |      0.715 |  0.550 |       +0.000 |     -0.152 |
| gsm8k          |  0.655 |  0.550 |      0.610 |  0.370 |       +0.060 |     -0.285 |

## Notes
- Benchmarks: arc, gsm8k (`--limit 200` per subset).
- All four models are served via stock vLLM (BF16 path) on unique ports (bf16: 8050, ptq: 8051, ptq_smooth: 8053, qat: 8052). The HiFP8-rounded weights are stored in BF16 storage, so no quant-method-aware loader is needed.
- evalscope client targets the OpenAI-compatible endpoint of each server.

### Method details
- **ptq** = BF16 → in-place HiFP8 fake-quant on every Linear weight (weight-only, no SmoothQuant). Per-row dynamic scale.
- **ptq_smooth** = naive SmoothQuant (alpha=0.5, 32 wikitext batches) → **fold-into-RMSNorm** fusion (q/k/v share input_layernorm, gate/up share post_attention_layernorm; o_proj/down_proj rolled back since no preceding norm to fold into) → HiFP8 fake-quant baked into Linear weights → plain nn.Linear save. **Zero runtime smooth_scale dependency** — any inference framework can serve it.
- **qat** = BF16 → 2 000 distillation steps (bs=1, grad-accum=4, seq=512, AdamW lr=1e-5, 0.5·CE + 0.5·KL with frozen BF16 teacher, T=2.0) on wikitext-103-raw, with HiFP8FakeQuantizedLinear(qat=True) wrapping every weight Linear. Trained from raw BF16 (not from PTQ).

### Reading the deltas
- **Δ smooth-ptq**: marginal benefit of fold-into-RMSNorm SmoothQuant on top of weight-only PTQ. Positive = SmoothQuant helped.
- **Δ qat-bf16**: total QAT loss vs lossless baseline. (QAT did not start from PTQ, so a fair PTQ-vs-QAT delta isn't well-defined.)

## Raw evalscope output

- `bf16` → `outputs/eval_results/bf16/`
- `ptq` → `outputs/eval_results/ptq/`
- `ptq_smooth` → `outputs/eval_results/ptq_smooth/`
- `qat` → `outputs/eval_results/qat/`

### Appendix A — what we tried that didn't work

We also tried a "full-coverage" fold variant that uses cross-layer
absorption to also smooth the unfoldable `o_proj` and `down_proj`:

  * `o_proj`'s 1/s is folded into `V_proj.weight` rows (GQA-aware,
    max-unified across attn-heads sharing each kv-head)
  * `down_proj`'s 1/s is folded into `up_proj.weight` rows
    (GLU path: `silu(gate) ⊙ (up/s)` ≡ `(silu(gate) ⊙ up)/s`)

Both folds are mathematically valid — pre-quant outputs are bit-identical
to vanilla SmoothQuant. But empirically on Qwen3-0.6B with HiFP8 per-row
weight quantization, the variant **regresses every benchmark** vs the
default norm-only fold (arc-easy −0.010, arc-challenge −0.025,
gsm8k −0.040).

Root cause: the smooth scales for these layers are 5-80 (median ~5-8).
Multiplying downstream Linear's columns by such scales inflates per-row
amax for outlier-heavy rows, forcing HiFP8's per-row LUT into its
coarse-precision extremes. The "push outliers into weight" assumption of
SmoothQuant assumes the weight quantizer has independent per-channel
scales (per-output-channel for INT8) — incompatible with our per-row
scheme.

The variant is preserved in code as `--full-fold` flag (see
`scripts/quantize_qwen3_ptq_smooth_fused.py`) and as
`fuse_crosslayer_smooths` in `quantization/smooth_fuse.py`. Use only if
you switch to a per-output-channel weight quantizer or AWQ-style scaling.

