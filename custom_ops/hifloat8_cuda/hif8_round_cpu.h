// HiFloat8 CPU rounding functions (float + double)
// CPU equivalent of hif8_round.cuh, using std:: math instead of CUDA intrinsics

#ifndef HIF8_ROUND_CPU_H
#define HIF8_ROUND_CPU_H

#include <cmath>
#include <limits>

// HiFloat8 rounding: rounds to nearest HiFloat8-representable value (float)
static inline float hif8_round_float_cpu(float x) {
    if (std::isnan(x) || std::isinf(x) || x == 0.0f)
        return x;

    float sign = std::copysign(1.0f, x);
    float ax = std::fabs(x);

    // Underflow: below 2^(-23)
    constexpr float UNDERFLOW_THRESH = 1.1920928955078125e-07f;  // 2^(-23)
    constexpr float MIN_DENORMAL = 2.384185791015625e-07f;       // 2^(-22)
    if (ax < UNDERFLOW_THRESH)
        return 0.0f;

    // Extract exponent
    int E_raw;
    std::frexp(ax, &E_raw);
    int E = E_raw - 1;

    // Denormal region: E <= -16
    if (E <= -16) {
        if (E <= -23)
            return sign * MIN_DENORMAL;

        float lower = std::ldexp(1.0f, E);
        float upper = std::ldexp(1.0f, E + 1);
        float mid = 1.5f * lower;
        float result = (ax >= mid) ? upper : lower;
        if (result < MIN_DENORMAL)
            return 0.0f;
        return sign * result;
    }

    // Overflow: E > 15
    if (E > 15)
        return sign * std::numeric_limits<float>::infinity();

    // Normal region: E in [-15, 15]
    int abs_E = (E < 0) ? -E : E;
    int mantissa_bits;
    if (abs_E <= 3)
        mantissa_bits = 3;
    else if (abs_E <= 7)
        mantissa_bits = 2;
    else
        mantissa_bits = 1;

    float shifted = std::ldexp(ax, mantissa_bits - E);
    float rounded = std::floor(shifted + 0.5f);

    float carry_threshold = std::ldexp(1.0f, mantissa_bits + 1);
    if (rounded >= carry_threshold) {
        int new_E = E + 1;
        if (new_E > 15)
            return sign * std::numeric_limits<float>::infinity();
        return sign * std::ldexp(1.0f, new_E);
    }

    float result = std::ldexp(rounded, E - mantissa_bits);
    if (result >= 49152.0f)
        return sign * std::numeric_limits<float>::infinity();
    return sign * result;
}

// HiFloat8 rounding: rounds to nearest HiFloat8-representable value (double)
static inline double hif8_round_double_cpu(double x) {
    if (std::isnan(x) || std::isinf(x) || x == 0.0)
        return x;

    double sign = std::copysign(1.0, x);
    double ax = std::fabs(x);

    constexpr double UNDERFLOW_THRESH = 1.1920928955078125e-07;
    constexpr double MIN_DENORMAL = 2.384185791015625e-07;
    if (ax < UNDERFLOW_THRESH)
        return 0.0;

    int E_raw;
    std::frexp(ax, &E_raw);
    int E = E_raw - 1;

    if (E <= -16) {
        if (E <= -23)
            return sign * MIN_DENORMAL;

        double lower = std::ldexp(1.0, E);
        double upper = std::ldexp(1.0, E + 1);
        double mid = 1.5 * lower;
        double result = (ax >= mid) ? upper : lower;
        if (result < MIN_DENORMAL)
            return 0.0;
        return sign * result;
    }

    if (E > 15)
        return sign * std::numeric_limits<double>::infinity();

    int abs_E = (E < 0) ? -E : E;
    int mantissa_bits = (abs_E <= 3) ? 3 : (abs_E <= 7) ? 2 : 1;

    double shifted = std::ldexp(ax, mantissa_bits - E);
    double rounded = std::floor(shifted + 0.5);

    double carry_threshold = std::ldexp(1.0, mantissa_bits + 1);
    if (rounded >= carry_threshold) {
        int new_E = E + 1;
        if (new_E > 15)
            return sign * std::numeric_limits<double>::infinity();
        return sign * std::ldexp(1.0, new_E);
    }

    double result = std::ldexp(rounded, E - mantissa_bits);
    if (result >= 49152.0)
        return sign * std::numeric_limits<double>::infinity();
    return sign * result;
}

#endif // HIF8_ROUND_CPU_H
