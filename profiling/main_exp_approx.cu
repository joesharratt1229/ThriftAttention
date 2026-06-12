#include <cuda_runtime.h>

#include <cstdint>

#define TA_EXP_BENCH_CUSTOM_EX2 1
#define TA_EXP_BENCH_IMPL_LABEL "int-rn-pow2-ftz"

__device__ __forceinline__
float ex2_approx_ftz(float x)
{
    uint32_t bits;
    asm volatile(
        "{\n\t"
        ".reg .s32 exp_int;\n\t"
        "cvt.rni.s32.f32 exp_int, %1;\n\t"
        "add.s32 exp_int, exp_int, 127;\n\t"
        "shl.b32 %0, exp_int, 23;\n\t"
        "}\n"
        : "=r"(bits)
        : "f"(x));
    return __uint_as_float(bits);
}

#include "main_exp.cu"
