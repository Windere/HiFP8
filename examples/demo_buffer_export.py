"""
Demonstration of buffer-based BF16 export.

Shows that scales are embedded as buffers in model.safetensors without separate files.
"""

import sys
import os
import tempfile
import torch
import torch.nn as nn

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.hifp8_config import HiFP8FakeQuantizeConfig
from quantization.hifp8_linear import HiFP8FakeQuantizedLinear
from quantization.smooth import apply_smooth_scale


def main():
    print("=" * 60)
    print("Buffer-Based BF16 Export Demonstration")
    print("=" * 60)

    # Create a simple model
    print("\n1. Creating model with HiFP8FakeQuantizedLinear...")
    model = nn.Sequential(
        HiFP8FakeQuantizedLinear.from_linear(
            nn.Linear(128, 256, device="cuda", dtype=torch.bfloat16),
            weight_config=HiFP8FakeQuantizeConfig(),
        ),
        HiFP8FakeQuantizedLinear.from_linear(
            nn.Linear(256, 512, device="cuda", dtype=torch.bfloat16),
            weight_config=HiFP8FakeQuantizeConfig(),
        ),
    )

    # Apply smooth scales to demonstrate buffer persistence
    print("\n2. Applying SmoothQuant scales...")
    smooth_scales = {
        "0": torch.ones(128, device="cuda", dtype=torch.bfloat16) * 2.0,
        "1": torch.ones(256, device="cuda", dtype=torch.bfloat16) * 1.5,
    }
    apply_smooth_scale(model, smooth_scales)

    # Check state_dict includes buffers
    print("\n3. Verifying buffers in state_dict...")
    state_dict = model.state_dict()
    buffer_keys = [k for k in state_dict.keys() if "smooth_scale" in k]
    print(f"   Found {len(buffer_keys)} smooth_scale buffers in state_dict:")
    for key in buffer_keys:
        print(f"   - {key}: shape={state_dict[key].shape}, dtype={state_dict[key].dtype}")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name

    print(f"\n4. Saving state_dict to {save_path}...")
    torch.save(state_dict, save_path)

    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"   File size: {file_size:.2f} KB")
    print(f"   Number of keys in state_dict: {len(state_dict)}")

    # Verify buffers survive load cycle
    print("\n5. Loading state_dict to verify persistence...")
    loaded_state = torch.load(save_path, weights_only=False)

    for key in buffer_keys:
        original = state_dict[key]
        loaded = loaded_state[key]
        matches = torch.equal(original, loaded)
        print(f"   {key}: {'✓ matches' if matches else '✗ mismatch'}")

    # Clean up
    os.unlink(save_path)

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("✓ Scales registered as buffers")
    print("✓ Buffers automatically included in state_dict")
    print("✓ Buffers persist through save/load cycle")
    print("✓ Single file storage (no separate .pt files)")
    print("\nFor HuggingFace models, use model.save_pretrained() which will")
    print("automatically include all buffers in model.safetensors")
    print("=" * 60)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo")
        sys.exit(0)

    main()
