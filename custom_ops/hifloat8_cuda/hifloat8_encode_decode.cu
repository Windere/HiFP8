// HiFloat8 uint8 encoding/decoding implementation for HiFP8 project
// Adapted from reference implementation

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "hifloat8_lut.h"
#include "hif8_round.cuh"  // Inline rounding function

// Encoding: float → uint8
__device__ __forceinline__ uint8_t hif8_encode_uint8_device(float x) {
    float rounded = hif8_round_float(x);

    if (rounded == 0.0f || rounded == -0.0f)
        return 0x00;

    if (__isinff(rounded))
        return (rounded > 0) ? 0x7F : 0xFF;

    uint8_t sign_bit = (rounded < 0.0f) ? 0x80 : 0x00;
    float magnitude = fabsf(rounded);
    uint8_t index = hif8_find_index(magnitude);

    return sign_bit | index;
}

// Decoding: uint8 → float
__device__ __forceinline__ float hif8_decode_uint8_device(uint8_t encoded) {
    bool is_negative = (encoded & 0x80) != 0;
    uint8_t index = encoded & 0x7F;
    float magnitude = hif8_lookup_value(index);
    return is_negative ? -magnitude : magnitude;
}

// Kernels
__global__ void hif8_encode_kernel(uint8_t* output, const float* input, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = hif8_encode_uint8_device(input[idx]);
    }
}

__global__ void hif8_decode_kernel(float* output, const uint8_t* input, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = hif8_decode_uint8_device(input[idx]);
    }
}

__global__ void compute_row_max_kernel(float* row_max, const float* input,
                                        int64_t num_rows, int64_t row_size) {
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        const float* row_data = input + row * row_size;
        float max_val = 0.0f;
        for (int64_t i = 0; i < row_size; i++) {
            float val = fabsf(row_data[i]);
            if (val > max_val) max_val = val;
        }
        row_max[row] = max_val;
    }
}

__global__ void hif8_encode_with_scale_kernel(uint8_t* output, float* scales,
                                                const float* input,
                                                int64_t num_rows, int64_t row_size) {
    int64_t row = blockIdx.x;
    int64_t col = threadIdx.x + blockIdx.y * blockDim.x;

    if (row < num_rows && col < row_size) {
        float scale = scales[row];
        float scale_factor = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
        int64_t idx = row * row_size + col;
        float val = input[idx] * scale_factor;
        output[idx] = hif8_encode_uint8_device(val);
    }
}

__global__ void hif8_decode_with_scale_kernel(float* output, const uint8_t* input,
                                                const float* scales,
                                                int64_t num_rows, int64_t row_size) {
    int64_t row = blockIdx.x;
    int64_t col = threadIdx.x + blockIdx.y * blockDim.x;

    if (row < num_rows && col < row_size) {
        float scale = scales[row];
        int64_t idx = row * row_size + col;
        float val = hif8_decode_uint8_device(input[idx]);
        output[idx] = val * scale;
    }
}

// Host launchers
torch::Tensor hif8_encode_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32,
                "Expected float32 CUDA tensor");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig, torch::dtype(torch::kUInt8));
    int64_t n = input_contig.numel();
    if (n == 0) return output;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    hif8_encode_kernel<<<blocks, threads>>>(
        output.data_ptr<uint8_t>(),
        input_contig.data_ptr<float>(),
        n
    );

    return output;
}

torch::Tensor hif8_decode_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kUInt8,
                "Expected uint8 CUDA tensor");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig, torch::dtype(torch::kFloat32));
    int64_t n = input_contig.numel();
    if (n == 0) return output;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    hif8_decode_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        input_contig.data_ptr<uint8_t>(),
        n
    );

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> hif8_encode_with_scale_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32,
                "Expected float32 CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Expected 2D tensor");

    auto input_contig = input.contiguous();
    int64_t num_rows = input_contig.size(0);
    int64_t row_size = input_contig.size(1);

    auto uint8_data = torch::empty_like(input_contig, torch::dtype(torch::kUInt8));
    auto scales = torch::empty({num_rows},
                               torch::dtype(torch::kFloat32).device(input.device()));

    // Compute scales
    const int threads = 256;
    const int blocks = (num_rows + threads - 1) / threads;
    compute_row_max_kernel<<<blocks, threads>>>(
        scales.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        num_rows, row_size
    );

    // Encode
    dim3 blocks2(num_rows, (row_size + threads - 1) / threads);
    hif8_encode_with_scale_kernel<<<blocks2, threads>>>(
        uint8_data.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        num_rows, row_size
    );

    return std::make_tuple(uint8_data, scales);
}

torch::Tensor hif8_decode_with_scale_cuda(torch::Tensor data, torch::Tensor scales) {
    TORCH_CHECK(data.is_cuda() && data.scalar_type() == torch::kUInt8,
                "Expected uint8 CUDA tensor");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kFloat32,
                "Expected float32 scale tensor");
    TORCH_CHECK(data.dim() == 2, "Expected 2D data");

    auto data_contig = data.contiguous();
    auto scales_contig = scales.contiguous();

    int64_t num_rows = data_contig.size(0);
    int64_t row_size = data_contig.size(1);

    auto output = torch::empty_like(data_contig, torch::dtype(torch::kFloat32));

    const int threads = 256;
    dim3 blocks(num_rows, (row_size + threads - 1) / threads);

    hif8_decode_with_scale_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        data_contig.data_ptr<uint8_t>(),
        scales_contig.data_ptr<float>(),
        num_rows, row_size
    );

    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hif8_encode_cuda", &hif8_encode_cuda, "HiFloat8 encode to uint8 (CUDA)");
    m.def("hif8_decode_cuda", &hif8_decode_cuda, "HiFloat8 decode from uint8 (CUDA)");
    m.def("hif8_encode_with_scale_cuda", &hif8_encode_with_scale_cuda,
          "HiFloat8 encode with per-row scaling (CUDA)");
    m.def("hif8_decode_with_scale_cuda", &hif8_decode_with_scale_cuda,
          "HiFloat8 decode with per-row scaling (CUDA)");
}
