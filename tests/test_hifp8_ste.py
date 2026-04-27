"""Unit tests for the HiFP8 STE wrapper (Phase 2 of QAT pipeline)."""
import sys
import pytest
import torch

sys.path.insert(0, "custom_ops")

from quantization.hifp8_ste import hifp8_fake_quantize_ste
from custom_ops.hifp8_ops import hifp8_fake_quantize
from custom_ops.hifp8_uint8_ops import HIF8_MAX


@pytest.fixture
def x_cuda():
    torch.manual_seed(0)
    return torch.randn(4, 64, device="cuda", dtype=torch.float32) * 5.0


def test_forward_matches_inference_path(x_cuda):
    """STE wrapper's forward output must equal the inference-only fake_quant."""
    y_ste = hifp8_fake_quantize_ste(x_cuda)
    y_ref = hifp8_fake_quantize(x_cuda)
    assert torch.equal(y_ste, y_ref), "STE forward diverged from inference path"


def test_gradient_flows_through_ste(x_cuda):
    """After backward, x.grad must be non-zero (gradient flows through quant)."""
    x = x_cuda.clone().requires_grad_(True)
    y = hifp8_fake_quantize_ste(x)
    loss = y.pow(2).sum()
    loss.backward()
    assert x.grad is not None, "no gradient produced"
    nonzero_frac = (x.grad != 0).float().mean().item()
    assert nonzero_frac > 0.99, f"too few non-zero gradients: {nonzero_frac:.3f}"


def test_clipped_ste_zeros_saturated():
    """Gradients for inputs above HIF8_MAX must be zeroed (clipped STE)."""
    huge = HIF8_MAX * 10.0
    small = 0.5
    x = torch.tensor([huge, small, -huge, -small], device="cuda", dtype=torch.float32,
                     requires_grad=True)
    y = hifp8_fake_quantize_ste(x, scale_factor=1.0)
    y.sum().backward()
    assert x.grad[0].item() == 0.0, "saturated +x grad should be 0"
    assert x.grad[2].item() == 0.0, "saturated -x grad should be 0"
    assert x.grad[1].item() == 1.0, f"in-range +x grad should pass through: {x.grad[1]}"
    assert x.grad[3].item() == 1.0, f"in-range -x grad should pass through: {x.grad[3]}"


def test_qat_linear_optimizer_step():
    """A wrapped Linear must accept an Adam step and reduce a contrived loss."""
    torch.manual_seed(0)
    x = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    target = torch.randn(8, 16, device="cuda", dtype=torch.float32)

    weight = torch.nn.Parameter(
        torch.randn(16, 32, device="cuda", dtype=torch.float32) * 0.1
    )
    optim = torch.optim.AdamW([weight], lr=1e-2)

    def step():
        wq = hifp8_fake_quantize_ste(weight)
        out = torch.nn.functional.linear(x, wq)
        return (out - target).pow(2).mean()

    loss0 = step().item()
    for _ in range(20):
        optim.zero_grad()
        loss = step()
        loss.backward()
        optim.step()
    loss1 = step().item()
    assert loss1 < loss0, f"loss did not decrease: {loss0:.4f} -> {loss1:.4f}"


def test_fake_quantizer_qat_flag():
    """HiFP8FakeQuantizer with qat=True flows gradients."""
    from quantization.hifp8_config import HiFP8FakeQuantizeConfig
    from quantization.hifp8_fake_quantizer import HiFP8FakeQuantizer

    cfg_on = HiFP8FakeQuantizeConfig(qat=True)
    qon = HiFP8FakeQuantizer(cfg_on).cuda()
    x = torch.randn(2, 16, device="cuda", requires_grad=True)
    y = qon(x)
    y.sum().backward()
    assert x.grad is not None, "qat=True must produce gradients"
    assert (x.grad != 0).float().mean().item() > 0.5
