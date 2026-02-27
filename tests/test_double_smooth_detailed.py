#!/usr/bin/env python3
"""Detailed test of smooth_scale application."""

import torch
import torch.nn as nn
from quantization import HiFP8FakeQuantizeConfig, prepare_hifp8_fake_quant

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

# Patch forward to track what it receives
original_forward = model[0].forward
forward_received = []

def patched_forward(self, x_in):
    forward_received.append(x_in[0, 0].item())
    print(f"  Forward receives: {x_in[0, 0].item()}")

    # Show value after smooth scaling
    if self.smooth_scale is not None:
        x_after_smooth = x_in / self.smooth_scale
        print(f"  Forward after smooth: {x_after_smooth[0, 0].item()}")

    return original_forward(x_in)

model[0].forward = lambda x: patched_forward(model[0], x)

# With hook - simulate calibration
from quantization.calibration import HiFP8ActivationObserver

observer = HiFP8ActivationObserver(
    granularity=model[0].activation_fake_quantizer.config.granularity,
    target_dtype=torch.float8_e4m3fn,
).cuda()

observed_values = []

def make_hook(observer):
    def hook(mod, input_tuple):
        x_hook = input_tuple[0]
        print(f"  Hook receives: {x_hook[0, 0].item()}")

        # Apply smooth_scale (如果这里应用了，forward也会应用)
        if mod.smooth_scale is not None:
            x_hook = x_hook / mod.smooth_scale
            print(f"  Hook after smooth: {x_hook[0, 0].item()}")

        observed_values.append(x_hook[0, 0].item())
        observer(x_hook)
    return hook

hook_handle = model[0].register_forward_pre_hook(make_hook(observer))

print("\nWith calibration hook:")
with torch.no_grad():
    output = model(x.clone())

hook_handle.remove()

print(f"\n" + "="*60)
print("Summary:")
print(f"Original input: 10.0")
print(f"Hook observed (after hook's smooth): {observed_values[0]}")
print(f"Forward received: {forward_received[0]}")
print(f"Forward will smooth again: {forward_received[0] / 2.0}")
print("="*60)

if forward_received[0] == 10.0:
    print("\n✅ Hook doesn't modify input to forward")
    print("   But observer sees smooth-scaled value (5.0)")
    print("   And forward will smooth-scale again (to 5.0)")
    print("\n💡 This is CORRECT for calibration:")
    print("   Observer should see the value that will be quantized")
    print("   (i.e., after smooth scaling)")
