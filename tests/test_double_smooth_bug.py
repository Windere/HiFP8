#!/usr/bin/env python3
"""Test if smooth_scale is applied twice during calibration."""

import torch
import torch.nn as nn
from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant
from quantization.smooth import apply_smooth_scale
from quantization.calibration import HiFP8ActivationObserver

# Create model
model = nn.Sequential(
    nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16),
).cuda()

# Apply HiFP8 quantization
model = prepare_hifp8_fake_quant(
    model,
    weight_config=HiFP8FakeQuantizeConfig(),
    activation_config=HiFP8FakeQuantizeConfig(),
)

# Apply smooth scale
smooth_scale = torch.ones(64, device="cuda", dtype=torch.bfloat16) * 2.0
model[0].set_smooth_scale(smooth_scale)

print(f"Smooth scale: {smooth_scale[0].item()}")

# Test input
x = torch.ones(4, 64, device="cuda", dtype=torch.bfloat16) * 10.0
print(f"Original input: {x[0, 0].item()}")

# Without hook - normal forward
with torch.no_grad():
    output_normal = model(x.clone())

# Expected: x / smooth_scale = 10.0 / 2.0 = 5.0
print(f"After normal forward (expected x/2=5.0): input should be divided once")

# With hook - simulate calibration
observer = HiFP8ActivationObserver(
    granularity=model[0].activation_fake_quantizer.config.granularity,
    target_dtype=torch.float8_e4m3fn,
).cuda()

observed_values = []

def make_hook(observer):
    def hook(mod, input_tuple):
        x_hook = input_tuple[0]
        print(f"  Hook sees input: {x_hook[0, 0].item()}")

        # Apply smooth_scale (THIS IS THE BUG!)
        if mod.smooth_scale is not None:
            x_hook = x_hook / mod.smooth_scale
            print(f"  Hook after smooth: {x_hook[0, 0].item()}")

        observed_values.append(x_hook[0, 0].item())
        observer(x_hook)
    return hook

hook_handle = model[0].register_forward_pre_hook(make_hook(observer))

print("\nWith calibration hook:")
with torch.no_grad():
    output_with_hook = model(x.clone())

hook_handle.remove()

print(f"\nObserved value in hook: {observed_values[0]}")
print(f"Expected if divided once: 5.0")
print(f"Expected if divided twice: 2.5")
print(f"Actual observed: {observed_values[0]}")

if abs(observed_values[0] - 2.5) < 0.1:
    print("\n❌ BUG CONFIRMED: Input is divided by smooth_scale TWICE!")
    print("   Hook divides once, then forward divides again.")
elif abs(observed_values[0] - 5.0) < 0.1:
    print("\n✅ Correct: Input is divided by smooth_scale only once")
