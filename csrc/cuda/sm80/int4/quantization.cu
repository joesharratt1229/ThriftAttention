#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "thriftattention/sm80/cuda_common.cuh"

template <typename T>
__device__ inline float sm80_to_float(T value);

template <>
__device__ inline float sm80_to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ inline float sm80_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ inline uint8_t pack_s4_pair(int low, int high) {
    return static_cast<uint8_t>((low & 0xf) | ((high & 0xf) << 4));
}

template <typename T, int HEAD_DIM>
__global__ void int4_quantize_kernel(
    const T* X,
    uint8_t* X_packed,
    float* X_scale,
    int rows)
{
    constexpr int SCALE_DIM = HEAD_DIM / 64;
    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    __shared__ float reduce[64];

    #pragma unroll
    for (int group = 0; group < SCALE_DIM; group++) {
        const int group_start = group * 64;
        float value = 0.0f;
        if (tid < 64) {
            value = fabsf(sm80_to_float<T>(X[row * HEAD_DIM + group_start + tid]));
        }
        reduce[tid] = value;
        __syncthreads();

        for (int stride = 32; stride > 0; stride /= 2) {
            if (tid < stride) {
                reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
            }
            __syncthreads();
        }

        const float scale = fmaxf(reduce[0], 1.0e-6f) / 7.0f;
        if (tid == 0) {
            X_scale[row * SCALE_DIM + group] = scale;
        }
        __syncthreads();

        if (tid < 32) {
            const int d0 = group_start + tid * 2;
            const int d1 = d0 + 1;
            int q0 = static_cast<int>(rintf(sm80_to_float<T>(X[row * HEAD_DIM + d0]) / scale));
            int q1 = static_cast<int>(rintf(sm80_to_float<T>(X[row * HEAD_DIM + d1]) / scale));
            q0 = max(-8, min(7, q0));
            q1 = max(-8, min(7, q1));
            X_packed[row * HEAD_DIM_2 + group * 32 + tid] = pack_s4_pair(q0, q1);
        }
        __syncthreads();
    }
}

template <typename T>
static void int4_quantize_typed(
    const void* X_raw,
    void* X_packed_raw,
    void* X_scale_raw,
    int rows,
    int head_dim)
{
    auto X = reinterpret_cast<const T*>(X_raw);
    auto X_packed = reinterpret_cast<uint8_t*>(X_packed_raw);
    auto X_scale = reinterpret_cast<float*>(X_scale_raw);

    if (head_dim == 64) {
        int4_quantize_kernel<T, 64><<<rows, 64>>>(X, X_packed, X_scale, rows);
    } else if (head_dim == 128) {
        int4_quantize_kernel<T, 128><<<rows, 64>>>(X, X_packed, X_scale, rows);
    }
}

void int4_quantize(
    const void* X_raw,
    void* X_packed_raw,
    void* X_scale_raw,
    int rows,
    int head_dim,
    bool is_bf16)
{
    if (is_bf16) {
        int4_quantize_typed<__nv_bfloat16>(X_raw, X_packed_raw, X_scale_raw, rows, head_dim);
    } else {
        int4_quantize_typed<half>(X_raw, X_packed_raw, X_scale_raw, rows, head_dim);
    }
}
