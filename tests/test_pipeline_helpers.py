import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
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


def test_export_dir_names():
    from scripts.test_full_pipeline import _export_dir
    assert _export_dir("/tmp/out", "baseline") == Path("/tmp/out/baseline")
    assert _export_dir("/tmp/out", "bf16") == Path("/tmp/out/bf16")
    assert _export_dir("/tmp/out", "hif8") == Path("/tmp/out/hif8")


def test_build_vllm_cmd_baseline():
    from scripts.test_full_pipeline import _build_vllm_cmd
    cmd = _build_vllm_cmd("baseline", "/tmp/model", 8010, "0")
    assert "-m" in cmd
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model" in cmd
    assert "/tmp/model" in cmd
    assert "start_vllm_hifp8_server_v4.py" not in " ".join(cmd)


def test_build_vllm_cmd_bf16_uses_v4_server():
    from scripts.test_full_pipeline import _build_vllm_cmd
    cmd = _build_vllm_cmd("bf16", "/tmp/model", 8010, "0")
    joined = " ".join(cmd)
    assert "start_vllm_hifp8_server_v4.py" in joined
    assert "/tmp/model" in joined


def test_build_vllm_cmd_hif8_uses_quantization_flag():
    from scripts.test_full_pipeline import _build_vllm_cmd
    cmd = _build_vllm_cmd("hif8", "/tmp/model", 8010, "0")
    assert "--quantization" in cmd
    idx = cmd.index("--quantization")
    assert cmd[idx + 1] == "hif8"
    assert "start_vllm_hifp8_server_v4.py" not in " ".join(cmd)


def test_build_vllm_cmd_uint8_uses_v4_server():
    from scripts.test_full_pipeline import _build_vllm_cmd
    cmd = _build_vllm_cmd("uint8", "/tmp/model", 8010, "0")
    joined = " ".join(cmd)
    assert "start_vllm_hifp8_server_v4.py" in joined


import tempfile
from pathlib import Path as _Path


def _write_json(path, data):
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(path, "w") as f:
        _json.dump(data, f)


def test_parse_arc_results_finds_accuracy(tmp_path):
    from scripts.test_full_pipeline import _parse_arc_results
    _write_json(tmp_path / "reports" / "arc_e.json",
                {"accuracy": 0.623, "num_examples": 100})
    result = _parse_arc_results(str(tmp_path))
    assert "accuracy" in result
    assert abs(result["accuracy"] - 0.623) < 1e-6


def test_parse_arc_results_empty_dir(tmp_path):
    from scripts.test_full_pipeline import _parse_arc_results
    result = _parse_arc_results(str(tmp_path))
    assert result == {}
