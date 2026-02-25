// Standalone implementation of hif8_round_float for uint8 encoding
// This provides independence from the main fake_quant kernel

#include <cuda.h>
#include <cuda_runtime.h>

// HiFloat8 rounding: rounds to nearest HiFloat8-representable value
__device__ float hif8_round_float(float x) {
    // Handle special values
    if (__isnanf(x) || __isinff(x) || x == 0.0f)
        return x;

    float sign = copysignf(1.0f, x);
    float ax = fabsf(x);

    // Underflow threshold
    if (ax < 1.1920928955078125e-07f)  // 2^(-23)
        return 0.0f;

    // Extract exponent
    unsigned int bits = __float_as_uint(ax);
    int E = (int)((bits >> 23) & 0xFF) - 127;

    // Denormal region: E <= -16
    if (E <= -16) {
        if (E <= -23)
            return sign * 2.384185791015625e-07f;  // 2^(-22)

        float lower = ldexpf(1.0f, E);
        float upper = ldexpf(1.0f, E + 1);
        float mid = 1.5f * lower;
        float result = (ax >= mid) ? upper : lower;

        if (result < 2.384185791015625e-07f)
            return 0.0f;
        return sign * result;
    }

    // Overflow: E > 15
    if (E > 15)
        return sign * __int_as_float(0x7f800000);  // Inf

    // Normal region: E in [-15, 15]
    int abs_E = (E < 0) ? -E : E;
    int mantissa_bits;
    if (abs_E <= 3)
        mantissa_bits = 3;
    else if (abs_E <= 7)
        mantissa_bits = 2;
    else
        mantissa_bits = 1;

    // Round with TA (round half away from zero)
    float shifted = ldexpf(ax, mantissa_bits - E);
    float rounded = floorf(shifted + 0.5f);

    // Handle carry
    float carry_threshold = ldexpf(1.0f, mantissa_bits + 1);
    if (rounded >= carry_threshold) {
        int new_E = E + 1;
        if (new_E > 15)
            return sign * __int_as_float(0x7f800000);
        return sign * ldexpf(1.0f, new_E);
    }

    float result = ldexpf(rounded, E - mantissa_bits);

    // Check Inf encoding
    if (result >= 49152.0f)
        return sign * __int_as_float(0x7f800000);

    return sign * result;
}
