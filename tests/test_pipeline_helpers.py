import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from scripts.test_full_pipeline import parse_args


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_full_pipeline.py",
                                      "--model", "/tmp/model"])
    args = parse_args()
    assert args.model == "/tmp/model"
    assert args.output_dir == "./outputs/pipeline_test"
    assert args.arc_n == 100
    assert args.modes == ["baseline", "bf16", "uint8", "hif8"]
    assert args.port == 8010
    assert args.vllm_startup_timeout == 120


def test_parse_args_custom_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_full_pipeline.py",
                                      "--model", "/tmp/model",
                                      "--modes", "baseline,bf16"])
    args = parse_args()
    assert args.modes == ["baseline", "bf16"]


def test_calibration_prompts_nonempty():
    from scripts.test_full_pipeline import CALIBRATION_PROMPTS
    assert len(CALIBRATION_PROMPTS) >= 4
    assert all(isinstance(p, str) and len(p) > 10 for p in CALIBRATION_PROMPTS)
