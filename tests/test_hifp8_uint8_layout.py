"""
Tests for HiFloat8 uint8 Layout and TensorImpl.

These tests verify the torchao integration for HiFloat8 uint8 encoding.
"""

import unittest
import torch

from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
from torchao.quantization.granularity import PerRow

from custom_ops.hifp8_uint8_layout import (
    HiFloat8Uint8Layout,
    HiFloat8Uint8AQTTensorImpl,
    quantize_to_hifloat8_uint8,
)


def _requires_cuda(test_func):
    """Decorator to skip tests if CUDA is not available."""
    return unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA not available"
    )(test_func)


class TestHiFloat8Uint8Layout(unittest.TestCase):
    """Tests for HiFloat8Uint8Layout class."""

    def test_layout_creation(self):
        """Test creating a HiFloat8Uint8Layout."""
        layout = HiFloat8Uint8Layout(param1=1, param2=2)
        self.assertEqual(layout.param1, 1)
        self.assertEqual(layout.param2, 2)

    def test_layout_default_params(self):
        """Test default parameters."""
        layout = HiFloat8Uint8Layout()
        self.assertEqual(layout.param1, 0)
        self.assertEqual(layout.param2, 0)

    def test_layout_repr(self):
        """Test layout string representation."""
        layout = HiFloat8Uint8Layout(param1=1, param2=2)
        repr_str = repr(layout)
        self.assertIn("HiFloat8Uint8Layout", repr_str)
        self.assertIn("param1=1", repr_str)
        self.assertIn("param2=2", repr_str)


@_requires_cuda
class TestHiFloat8Uint8AQTTensorImpl(unittest.TestCase):
    """Tests for HiFloat8Uint8AQTTensorImpl class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda")
        self.layout = HiFloat8Uint8Layout()

    def test_tensor_impl_creation(self):
        """Test creating a HiFloat8Uint8AQTTensorImpl."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data,
            scale,
            zero_point=None,
            _layout=self.layout,
        )

        self.assertEqual(tensor_impl.uint8_data.shape, (4, 8))
        self.assertEqual(tensor_impl.scale.shape, (4,))
        self.assertIsNone(tensor_impl.zero_point)
        self.assertIsInstance(tensor_impl._layout, HiFloat8Uint8Layout)

    def test_tensor_impl_dtype_validation(self):
        """Test that tensor impl validates dtypes."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        # Should work with correct dtypes
        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data, scale, None, self.layout
        )
        self.assertIsNotNone(tensor_impl)

        # Should fail with wrong uint8_data dtype
        with self.assertRaises(AssertionError):
            HiFloat8Uint8AQTTensorImpl(
                uint8_data.to(torch.int8),  # Wrong dtype
                scale,
                None,
                self.layout
            )

        # Should fail with wrong scale dtype (int32 is not a float type)
        with self.assertRaises(AssertionError):
            HiFloat8Uint8AQTTensorImpl(
                uint8_data,
                scale.to(torch.int32),  # Wrong dtype
                None,
                self.layout
            )

    def test_tensor_flatten_unflatten(self):
        """Test tensor flattening/unflattening for serialization."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data, scale, None, self.layout
        )

        # Flatten
        tensors, attrs = tensor_impl.__tensor_flatten__()
        self.assertIn("uint8_data", tensors)
        self.assertIn("scale", tensors)
        self.assertNotIn("zero_point", tensors)  # Should be absent when None
        self.assertEqual(len(attrs), 1)
        self.assertIsInstance(attrs[0], HiFloat8Uint8Layout)

        # Unflatten
        tensor_dict = {
            "uint8_data": uint8_data,
            "scale": scale,
        }
        reconstructed = HiFloat8Uint8AQTTensorImpl.__tensor_unflatten__(
            tensor_dict, attrs, None, None
        )

        self.assertTrue(torch.equal(reconstructed.uint8_data, uint8_data))
        self.assertTrue(torch.equal(reconstructed.scale, scale))
        self.assertIsNone(reconstructed.zero_point)

    def test_tensor_to_device(self):
        """Test moving tensor to different device."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data, scale, None, self.layout
        )

        # Move to CPU
        cpu_impl = tensor_impl.to("cpu")
        self.assertEqual(cpu_impl.uint8_data.device.type, "cpu")
        self.assertEqual(cpu_impl.scale.device.type, "cpu")

        # Move back to CUDA
        cuda_impl = cpu_impl.to("cuda")
        self.assertEqual(cuda_impl.uint8_data.device.type, "cuda")
        self.assertEqual(cuda_impl.scale.device.type, "cuda")

    def test_tensor_clone(self):
        """Test cloning tensor."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data, scale, None, self.layout
        )

        cloned = tensor_impl.clone()
        self.assertTrue(torch.equal(cloned.uint8_data, tensor_impl.uint8_data))
        self.assertTrue(torch.equal(cloned.scale, tensor_impl.scale))
        self.assertIsNot(cloned.uint8_data, tensor_impl.uint8_data)  # Different objects

    def test_tensor_transpose(self):
        """Test transposing tensor."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data, scale, None, self.layout
        )

        transposed = tensor_impl.t()
        self.assertEqual(transposed.uint8_data.shape, (8, 4))
        self.assertEqual(transposed.scale.shape, (4,))  # Scale unchanged

    def test_get_plain(self):
        """Test get_plain() method."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data, scale, None, self.layout
        )

        data, s, zp = tensor_impl.get_plain()
        self.assertTrue(torch.equal(data, uint8_data))
        self.assertTrue(torch.equal(s, scale))
        self.assertIsNone(zp)

    def test_from_plain(self):
        """Test from_plain() class method."""
        uint8_data = torch.randint(0, 256, (4, 8), dtype=torch.uint8, device=self.device)
        scale = torch.randn(4, dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl.from_plain(
            uint8_data, scale, None, self.layout
        )

        self.assertTrue(torch.equal(tensor_impl.uint8_data, uint8_data))
        self.assertTrue(torch.equal(tensor_impl.scale, scale))
        self.assertIsNone(tensor_impl.zero_point)

    def test_dequantize(self):
        """Test dequantize() produces correct shape and dtype."""
        # Create simple quantized data
        uint8_data = torch.tensor([[100, 200], [50, 150]], dtype=torch.uint8, device=self.device)
        scale = torch.tensor([2.0, 1.0], dtype=torch.float32, device=self.device)

        tensor_impl = HiFloat8Uint8AQTTensorImpl(
            uint8_data, scale, None, self.layout
        )

        # Dequantize (may use CUDA kernels or fallback depending on availability)
        dequant = tensor_impl.dequantize(output_dtype=torch.float32)

        # Check shape and dtype
        self.assertEqual(dequant.shape, (2, 2))
        self.assertEqual(dequant.dtype, torch.float32)


@_requires_cuda
class TestQuantizeToHiFloat8Uint8(unittest.TestCase):
    """Tests for quantize_to_hifloat8_uint8() helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda")

    def test_basic_quantization(self):
        """Test basic quantization workflow."""
        weight = torch.randn(4, 8, dtype=torch.float32, device=self.device)

        aqt = quantize_to_hifloat8_uint8(weight)

        # Check it's an AffineQuantizedTensor
        self.assertIsInstance(aqt, AffineQuantizedTensor)

        # Check tensor impl
        self.assertIsInstance(aqt.tensor_impl, HiFloat8Uint8AQTTensorImpl)

        # Check layout via tensor_impl
        self.assertIsInstance(aqt.tensor_impl.get_layout(), HiFloat8Uint8Layout)

        # Check dtypes
        self.assertEqual(aqt.tensor_impl.uint8_data.dtype, torch.uint8)
        self.assertEqual(aqt.tensor_impl.scale.dtype, torch.float32)

        # Check shapes
        self.assertEqual(aqt.tensor_impl.uint8_data.shape, (4, 8))
        self.assertEqual(aqt.tensor_impl.scale.shape, (4,))  # Per-row

    def test_quantization_with_custom_params(self):
        """Test quantization with custom parameters."""
        weight = torch.randn(4, 8, dtype=torch.float32, device=self.device)

        aqt = quantize_to_hifloat8_uint8(
            weight,
            param1=1,
            param2=2,
        )

        # Check layout parameters
        layout = aqt.tensor_impl.get_layout()
        self.assertEqual(layout.param1, 1)
        self.assertEqual(layout.param2, 2)

    def test_quantization_bfloat16_input(self):
        """Test quantization with bfloat16 input."""
        weight = torch.randn(4, 8, dtype=torch.bfloat16, device=self.device)

        aqt = quantize_to_hifloat8_uint8(weight)

        # Should work and produce uint8
        self.assertEqual(aqt.tensor_impl.uint8_data.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
