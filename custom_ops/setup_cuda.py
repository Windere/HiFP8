"""
Setup script for HiFP8 CUDA extensions (uint8 encoding/decoding).
This builds the CUDA kernels as a PyTorch extension.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Bypass CUDA version check if there's a mismatch
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

# Get paths
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_dir = os.path.join(current_dir, "hifloat8_cuda")
parent_dir = os.path.dirname(current_dir)  # custom_ops/

# Sources (only encode_decode.cu, rounding is now inline in header)
sources = [
    os.path.join(cuda_dir, "hifloat8_encode_decode.cu"),
]

setup(
    name="hifp8_cuda_uint8",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="hifp8_cuda_uint8",
            sources=sources,
            include_dirs=[cuda_dir],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
