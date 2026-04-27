#!/usr/bin/env python
"""
QAT fine-tune Qwen3-0.6B with HiFP8 fake-quantization + KL distillation.

  Teacher: frozen BF16 Qwen3-0.6B (HF weights)
  Student: copy of the same model with all nn.Linear (excluding lm_head and
           embeddings) wrapped in HiFP8FakeQuantizedLinear(qat=True).
           Initialised from the SmoothQuant'd PTQ checkpoint so QAT
           contribution is measured incrementally.
  Loss   : 0.5 * CE(student, labels) + 0.5 * KL(student || teacher, T=2.0)
  Steps  : 2 000   (logged every 50, validation every 200)
  Data   : wikitext-103-raw-v1, 1024-token sequences, batch=4, accum=2
  Output : outputs/qwen3_qat/  (HF-format student checkpoint)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# Make sibling packages (quantization/, custom_ops/) importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "custom_ops"))

from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear


def build_dataloaders(tokenizer, seq_len=1024, batch_size=4, n_train=4096, n_validation=200):
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_fn(rows):
        text = "\n".join(r for r in rows["text"] if r and len(r) > 50)
        ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        n = (ids.numel() // seq_len) * seq_len
        if n == 0:
            return {"input_ids": []}
        return {"input_ids": ids[:n].view(-1, seq_len).tolist()}

    train_ds = raw["train"].select(range(min(20_000, len(raw["train"])))) \
                           .map(tokenize_fn, batched=True, batch_size=2000,
                                remove_columns=raw["train"].column_names)
    val_ds = raw["validation"].map(tokenize_fn, batched=True, batch_size=2000,
                                    remove_columns=raw["validation"].column_names)

    train_ds = train_ds.with_format("torch")
    val_ds = val_ds.with_format("torch")
    train_ds = train_ds.shuffle(seed=0).select(range(min(n_train, len(train_ds))))
    val_ds = val_ds.select(range(min(n_validation, len(val_ds))))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))


def wrap_student_with_hifp8_qat(model: nn.Module) -> nn.Module:
    cfg = HiFP8FakeQuantizeConfig(qat=True)

    def _replace(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and "lm_head" not in full \
               and "embed" not in full.lower():
                replacement = HiFP8FakeQuantizedLinear.from_linear(
                    child, weight_config=cfg, activation_config=cfg,
                )
                setattr(parent, name, replacement)
            else:
                _replace(child, full)

    _replace(model)
    return model


def distill_loss(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
    ce = F.cross_entropy(
        student_logits[:, :-1].reshape(-1, student_logits.size(-1)).float(),
        labels[:, 1:].reshape(-1),
    )
    s_log = F.log_softmax(student_logits.float() / T, dim=-1)
    t_prob = F.softmax(teacher_logits.float() / T, dim=-1)
    kl = F.kl_div(s_log, t_prob, reduction="batchmean") * (T * T)
    return alpha * ce + (1 - alpha) * kl, ce.detach().item(), kl.detach().item()


@torch.no_grad()
def compute_validation_loss(student, validation_loader, device, max_batches=10):
    student.train(False)
    total, n = 0.0, 0
    for batch in validation_loader:
        ids = batch["input_ids"].to(device)
        out = student(ids).logits
        loss = F.cross_entropy(
            out[:, :-1].reshape(-1, out.size(-1)).float(),
            ids[:, 1:].reshape(-1),
        ).item()
        total += loss
        n += 1
        if n >= max_batches:
            break
    student.train(True)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptq-init", default="outputs/qwen3_ptq")
    ap.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output", default="outputs/qwen3_qat")
    ap.add_argument("--steps", type=int, default=2000)
    # Effective batch = batch-size × grad-accum (default 1×4=4).
    # Larger combos OOM on 32GB because Qwen3 vocab=151936 makes KL tensor huge.
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("outputs/logs/phase_4_qat.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log("loading teacher (frozen BF16)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16,
    ).to(device)
    teacher.train(False)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Start student from the same BF16 base as the teacher. We deliberately
    # do NOT chain from the PTQ+SmoothQuant checkpoint here: AutoModel.from_pretrained
    # discards the HiFP8FakeQuantizedLinear's smooth_scale buffers as "UNEXPECTED",
    # leaving the model with weights scaled by the smoothquant factor but no
    # corresponding activation divisor — that produced an untrained-level CE
    # (~13.6) at step 1 in the dry-run. Initialising student from BF16 keeps
    # the QAT-vs-PTQ comparison fair (both start from the same BF16 weights).
    log(f"loading student from {args.teacher} (BF16 base)...")
    student = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16,
    ).to(device)
    student = wrap_student_with_hifp8_qat(student)
    student.train(True)
    n_quant = sum(1 for m in student.modules() if isinstance(m, HiFP8FakeQuantizedLinear))
    log(f"wrapped {n_quant} Linear layers with HiFP8FakeQuantizedLinear(qat=True)")

    log("loading tokenizer + dataset...")
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_loader, validation_loader = build_dataloaders(
        tok, seq_len=args.seq_len, batch_size=args.batch_size,
    )

    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup,
        num_training_steps=args.steps,
    )

    log(f"training: {args.steps} steps, bs={args.batch_size}, accum={args.grad_accum}")
    losses = []
    step = 0
    train_iter = iter(train_loader)
    while step < args.steps:
        optim.zero_grad()
        accum_loss = 0.0
        ce_last = float("nan")
        kl_last = float("nan")
        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            ids = batch["input_ids"].to(device)
            with torch.no_grad():
                t_logits = teacher(ids).logits
            s_logits = student(ids).logits
            loss, ce_last, kl_last = distill_loss(s_logits, t_logits, ids)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optim.step()
        sched.step()
        losses.append(accum_loss)
        step += 1
        if step % 50 == 0 or step == 1:
            log(f"step {step:>5}/{args.steps}  loss={accum_loss:.4f}  "
                f"ce={ce_last:.4f}  kl={kl_last:.4f}  lr={sched.get_last_lr()[0]:.2e}")
        if step % 200 == 0:
            ev = compute_validation_loss(student, validation_loader, device)
            log(f"  >> validation_loss @ step {step} = {ev:.4f}")

    log(f"saving student to {out_dir}")
    student.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log("✅ Phase 4 done.")


if __name__ == "__main__":
    main()
