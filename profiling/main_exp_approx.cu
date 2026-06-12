#include <cuda_runtime.h>

#include <cstdint>

#define TA_EXP_BENCH_CUSTOM_EX2 1
#define TA_EXP_BENCH_IMPL_LABEL "int-rn-pow2-ftz"

__device__ __forceinline__
float ex2_approx_ftz(float x)
{
    float r = x + 12582912.0f;
    uint32_t b = (__float_as_uint(r) << 23) + 0x3F800000u;
    return __uint_as_float(b);
}

#include "main_exp.cu"
