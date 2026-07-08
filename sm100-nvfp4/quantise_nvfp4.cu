#include <cstdint>
#include <cstdio>
#include <float.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int ELEMENTS_PER_THREAD = 16;

__host__ __device__ inline int ta_cdiv(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int ta_sage_perm32(int x) {
    return (x / 8) * 2 + ((x % 8) / 2) * 8 + (x % 2);
}

__host__ __device__ inline int ta_sage_perm32_inv(int x) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (ta_sage_perm32(i) == x) {
            return i;
        }
    }
    return x;
}

__host__ __device__ inline int ta_sage_perm_seq(int x, bool inverse = false) {
    const int base = (x / 32) * 32;
    const int local = x & 31;
    return base + (inverse ? ta_sage_perm32_inv(local) : ta_sage_perm32(local));
}

__device__ inline uint32_t ta_cvt_8xf32_to_e2m1_packed(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7) {
    uint32_t packed;
    asm volatile(
        "{\n\t"
        ".reg .b8 a0, a1, a2, a3;\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a0, %1, %2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a1, %3, %4;\n\t"
        "mov.b16 lo, {a0, a1};\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a2, %5, %6;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 a3, %7, %8;\n\t"
        "mov.b16 hi, {a2, a3};\n\t"
        "mov.b32 %0, {lo, hi};\n\t"
        "}"
        : "=r"(packed)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3),
          "f"(f4), "f"(f5), "f"(f6), "f"(f7));
    return packed;
}

template <typename T>
struct PrecisionTraits;

template <>
struct PrecisionTraits<half> {
    using scalar = half;
    using vec2 = __half2;

    static __device__ __forceinline__ float to_float(scalar x) {
        return __half2float(x);
    }

    static __device__ __forceinline__ vec2 make_vec2(scalar a, scalar b) {
        return __halves2half2(a, b);
    }

    static __device__ __forceinline__ vec2 abs2(vec2 x) {
        return __habs2(x);
    }

    static __device__ __forceinline__ vec2 max2(vec2 a, vec2 b) {
        return __hmax2(a, b);
    }

    static __device__ __forceinline__ scalar low(vec2 x) {
        return __low2half(x);
    }

    static __device__ __forceinline__ scalar high(vec2 x) {
        return __high2half(x);
    }

    static __device__ __forceinline__ float2 to_float2(vec2 x) {
        return __half22float2(x);
    }
};

template <>
struct PrecisionTraits<__nv_bfloat16> {
    using scalar = __nv_bfloat16;
    using vec2 = __nv_bfloat162;

    static __device__ __forceinline__ float to_float(scalar x) {
        return __bfloat162float(x);
    }

    static __device__ __forceinline__ vec2 make_vec2(scalar a, scalar b) {
        return __halves2bfloat162(a, b);
    }

    static __device__ __forceinline__ vec2 abs2(vec2 x) {
        return __habs2(x);
    }

    static __device__ __forceinline__ vec2 max2(vec2 a, vec2 b) {
        return __hmax2(a, b);
    }

    static __device__ __forceinline__ scalar low(vec2 x) {
        return __low2bfloat16(x);
    }

    static __device__ __forceinline__ scalar high(vec2 x) {
        return __high2bfloat16(x);
    }

    static __device__ __forceinline__ float2 to_float2(vec2 x) {
        return __bfloat1622float2(x);
    }
};

template<typename T, int SEQ_PER_BLOCK, int THREADS_PER_HEAD>
__global__
void nvfp4_quantise_kernel(const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale, int bs, int seq_len, int head_dim)
{
    using Traits = PrecisionTraits<T>;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_seq_blocks = ta_cdiv(seq_len, SEQ_PER_BLOCK);
    const int batch_id = bid/num_seq_blocks;

    const int seq_block_id = bid % num_seq_blocks;
    const int seq_thread_id = tid / THREADS_PER_HEAD;
    const int head_id = tid % THREADS_PER_HEAD;

    X += (batch_id * seq_len *head_dim) + (seq_block_id * SEQ_PER_BLOCK + seq_thread_id) * head_dim + head_id * ELEMENTS_PER_THREAD;

    const int seq_id = seq_block_id * SEQ_PER_BLOCK + seq_thread_id;

    uint32_t X_reg[2][4];

    if (seq_id < seq_len) {
        asm volatile(
            "ld.global.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
            "ld.global.v4.b32 {%4, %5, %6, %7}, [%8+16];\n\t"
            : "=r"(X_reg[0][0]), "=r"(X_reg[0][1]), "=r"(X_reg[0][2]), "=r"(X_reg[0][3]),
              "=r"(X_reg[1][0]), "=r"(X_reg[1][1]), "=r"(X_reg[1][2]), "=r"(X_reg[1][3])
            : "l"(X)
        );
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            X_reg[0][i] = 0;
            X_reg[1][i] = 0;
        }
    }

    typename Traits::vec2* X_h2 = reinterpret_cast<typename Traits::vec2*>(X_reg);

    typename Traits::vec2 local_max = Traits::abs2(X_h2[0]);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        local_max = Traits::max2(local_max, Traits::abs2(X_h2[i]));
    }

    float vec_max = max(Traits::to_float(Traits::low(local_max)),
                        Traits::to_float(Traits::high(local_max)));

    float sf = vec_max / 6.0f;
    uint8_t sf_fp8;
    reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8) = __nv_fp8_e4m3(sf);
    sf = float(reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8));

    float sf_inv = (sf == 0.0f) ? 0.0f : 1.0f / sf;

    float2 X_f2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X_f2[i] = Traits::to_float2(X_h2[i]);
        X_f2[i].x *= sf_inv;
        X_f2[i].y *= sf_inv;
    }

    uint32_t fp4_packed[2];
    fp4_packed[0] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[0].y, X_f2[0].x, X_f2[1].y, X_f2[1].x,
        X_f2[2].y, X_f2[2].x, X_f2[3].y, X_f2[3].x);
    fp4_packed[1] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[4].y, X_f2[4].x, X_f2[5].y, X_f2[5].x,
        X_f2[6].y, X_f2[6].x, X_f2[7].y, X_f2[7].x);

    if (seq_id < seq_len) {
        __nv_fp4x2_e2m1* out = X_fp4 + batch_id * seq_len * (head_dim / 2)
                              + seq_id * (head_dim / 2)
                              + head_id * (ELEMENTS_PER_THREAD / 2);
        reinterpret_cast<uint64_t*>(out)[0] = reinterpret_cast<uint64_t*>(fp4_packed)[0];

        X_scale[batch_id * seq_len * (head_dim / 16) + seq_id * (head_dim / 16) + head_id] =
            reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8);
    }
}

template<typename T, int HEAD_DIM>
static void nvfp4_quantise_launch(
    const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale,
    int bs, int seq_len) {

    constexpr int SEQ_PER_BLOCK = 128;
    constexpr int THREADS_PER_HEAD = HEAD_DIM / ELEMENTS_PER_THREAD;
    constexpr int TB_SIZE = SEQ_PER_BLOCK * THREADS_PER_HEAD;

    const int num_blocks = bs * ta_cdiv(seq_len, SEQ_PER_BLOCK);

    nvfp4_quantise_kernel<T, SEQ_PER_BLOCK, THREADS_PER_HEAD>
        <<<num_blocks, TB_SIZE>>>(X, X_fp4, X_scale, bs, seq_len, HEAD_DIM);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "nvfp4_quantise_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
static void dispatch_nvfp4_quantise(
    const void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim) {

    auto X = reinterpret_cast<const T*>(X_raw);
    auto X_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(X_fp4_raw);
    auto X_scale = reinterpret_cast<__nv_fp8_e4m3*>(X_scale_raw);

    if (head_dim == 64)
        nvfp4_quantise_launch<T, 64>(X, X_fp4, X_scale, bs, seq_len);
    else if (head_dim == 128)
        nvfp4_quantise_launch<T, 128>(X, X_fp4, X_scale, bs, seq_len);
    else
        fprintf(stderr, "nvfp4_quantise: unsupported head_dim=%d (must be 64 or 128)\n", head_dim);
}

void nvfp4_quantise(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_nvfp4_quantise<__nv_bfloat16>(X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim);
    } else {
        dispatch_nvfp4_quantise<half>(X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim);
    }
}

template<typename T, int SEQ_PER_BLOCK, int THREADS_PER_HEAD>
__global__
void nvfp4_quantise_permute_seq_kernel(
    const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale,
    int bs, int seq_len, int head_dim, bool inverse)
{
    using Traits = PrecisionTraits<T>;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_seq_blocks = ta_cdiv(seq_len, SEQ_PER_BLOCK);
    const int batch_id = bid / num_seq_blocks;
    const int seq_block_id = bid % num_seq_blocks;
    const int seq_thread_id = tid / THREADS_PER_HEAD;
    const int head_id = tid % THREADS_PER_HEAD;

    const int phys_seq_id = seq_block_id * SEQ_PER_BLOCK + seq_thread_id;
    const int logical_local = ta_sage_perm_seq(seq_thread_id, inverse);
    const int logical_seq_id = seq_block_id * SEQ_PER_BLOCK + logical_local;

    const T* X_ptr = X + batch_id * seq_len * head_dim
                    + logical_seq_id * head_dim
                    + head_id * ELEMENTS_PER_THREAD;

    uint32_t X_reg[2][4];
    if (logical_seq_id < seq_len && phys_seq_id < seq_len) {
        asm volatile(
            "ld.global.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
            "ld.global.v4.b32 {%4, %5, %6, %7}, [%8+16];\n\t"
            : "=r"(X_reg[0][0]), "=r"(X_reg[0][1]), "=r"(X_reg[0][2]), "=r"(X_reg[0][3]),
              "=r"(X_reg[1][0]), "=r"(X_reg[1][1]), "=r"(X_reg[1][2]), "=r"(X_reg[1][3])
            : "l"(X_ptr)
        );
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            X_reg[0][i] = 0;
            X_reg[1][i] = 0;
        }
    }

    typename Traits::vec2* X_h2 = reinterpret_cast<typename Traits::vec2*>(X_reg);

    typename Traits::vec2 local_max = Traits::abs2(X_h2[0]);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        local_max = Traits::max2(local_max, Traits::abs2(X_h2[i]));
    }

    float vec_max = max(Traits::to_float(Traits::low(local_max)),
                        Traits::to_float(Traits::high(local_max)));

    float sf = vec_max / 6.0f;
    uint8_t sf_fp8;
    reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8) = __nv_fp8_e4m3(sf);
    sf = float(reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8));

    float sf_inv = (sf == 0.0f) ? 0.0f : 1.0f / sf;

    float2 X_f2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X_f2[i] = Traits::to_float2(X_h2[i]);
        X_f2[i].x *= sf_inv;
        X_f2[i].y *= sf_inv;
    }

    uint32_t fp4_packed[2];
    fp4_packed[0] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[0].y, X_f2[0].x, X_f2[1].y, X_f2[1].x,
        X_f2[2].y, X_f2[2].x, X_f2[3].y, X_f2[3].x);
    fp4_packed[1] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[4].y, X_f2[4].x, X_f2[5].y, X_f2[5].x,
        X_f2[6].y, X_f2[6].x, X_f2[7].y, X_f2[7].x);

    if (phys_seq_id < seq_len) {
        __nv_fp4x2_e2m1* out = X_fp4 + batch_id * seq_len * (head_dim / 2)
                              + phys_seq_id * (head_dim / 2)
                              + head_id * (ELEMENTS_PER_THREAD / 2);
        reinterpret_cast<uint64_t*>(out)[0] = reinterpret_cast<uint64_t*>(fp4_packed)[0];

        X_scale[batch_id * seq_len * (head_dim / 16) + phys_seq_id * (head_dim / 16) + head_id] =
            reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8);
    }
}

template<typename T, int HEAD_DIM>
static void nvfp4_quantise_permute_seq_launch(
    const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale,
    int bs, int seq_len, bool inverse) {

    constexpr int SEQ_PER_BLOCK = 128;
    constexpr int THREADS_PER_HEAD = HEAD_DIM / ELEMENTS_PER_THREAD;
    constexpr int TB_SIZE = SEQ_PER_BLOCK * THREADS_PER_HEAD;

    const int num_blocks = bs * ta_cdiv(seq_len, SEQ_PER_BLOCK);

    nvfp4_quantise_permute_seq_kernel<T, SEQ_PER_BLOCK, THREADS_PER_HEAD>
        <<<num_blocks, TB_SIZE>>>(X, X_fp4, X_scale, bs, seq_len, HEAD_DIM, inverse);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "nvfp4_quantise_permute_seq_kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

template<typename T>
static void dispatch_nvfp4_quantise_permute_seq(
    const void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool inverse) {

    auto X = reinterpret_cast<const T*>(X_raw);
    auto X_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(X_fp4_raw);
    auto X_scale = reinterpret_cast<__nv_fp8_e4m3*>(X_scale_raw);

    if (head_dim == 64)
        nvfp4_quantise_permute_seq_launch<T, 64>(X, X_fp4, X_scale, bs, seq_len, inverse);
    else if (head_dim == 128)
        nvfp4_quantise_permute_seq_launch<T, 128>(X, X_fp4, X_scale, bs, seq_len, inverse);
    else
        fprintf(stderr, "nvfp4_quantise_permute_seq: unsupported head_dim=%d (must be 64 or 128)\n", head_dim);
}

void nvfp4_quantise_permute_seq(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool inverse,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_nvfp4_quantise_permute_seq<__nv_bfloat16>(
            X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim, inverse);
    } else {
        dispatch_nvfp4_quantise_permute_seq<half>(
            X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim, inverse);
    }
}

template<typename T, int HEAD_DIM, int SEQ_PER_BLOCK>
__global__
void nvfp4_quantise_transpose_kernel(
    const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale,
    int bs, int seq_len, int padded_seq)
{
    using Traits = PrecisionTraits<T>;
    constexpr int THREADS_PER_HEAD = HEAD_DIM / ELEMENTS_PER_THREAD;
    constexpr int THREADS_PER_SEQ  = SEQ_PER_BLOCK / ELEMENTS_PER_THREAD;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_seq_blocks = ta_cdiv(seq_len, SEQ_PER_BLOCK);
    const int batch_id = bid / num_seq_blocks;
    const int seq_block_id = bid % num_seq_blocks;

    const int load_seq_local  = tid / THREADS_PER_HEAD;
    const int load_head_chunk = tid % THREADS_PER_HEAD;
    const int load_seq_global = seq_block_id * SEQ_PER_BLOCK + load_seq_local;

    const T* X_ptr = X + batch_id * seq_len * HEAD_DIM
                    + load_seq_global * HEAD_DIM
                    + load_head_chunk * ELEMENTS_PER_THREAD;

    uint32_t X_reg[2][4];
    if (load_seq_global < seq_len) {
        asm volatile(
            "ld.global.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
            "ld.global.v4.b32 {%4, %5, %6, %7}, [%8+16];\n\t"
            : "=r"(X_reg[0][0]), "=r"(X_reg[0][1]), "=r"(X_reg[0][2]), "=r"(X_reg[0][3]),
              "=r"(X_reg[1][0]), "=r"(X_reg[1][1]), "=r"(X_reg[1][2]), "=r"(X_reg[1][3])
            : "l"(X_ptr)
        );
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            X_reg[0][i] = 0;
            X_reg[1][i] = 0;
        }
    }

    __shared__ T smem[SEQ_PER_BLOCK * HEAD_DIM];

    T* smem_row = smem + load_seq_local * HEAD_DIM + load_head_chunk * ELEMENTS_PER_THREAD;
    *reinterpret_cast<uint4*>(smem_row)     = *reinterpret_cast<uint4*>(X_reg[0]);
    *reinterpret_cast<uint4*>(smem_row + 8) = *reinterpret_cast<uint4*>(X_reg[1]);

    __syncthreads();

    const int dim_id    = tid / THREADS_PER_SEQ;
    const int seq_chunk = tid % THREADS_PER_SEQ;

    typename Traits::vec2 vals_h2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int s0 = seq_chunk * ELEMENTS_PER_THREAD + 2 * i;
        int s1 = s0 + 1;
        vals_h2[i] = Traits::make_vec2(smem[s0 * HEAD_DIM + dim_id],
                                       smem[s1 * HEAD_DIM + dim_id]);
    }

    typename Traits::vec2 local_max = Traits::abs2(vals_h2[0]);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        local_max = Traits::max2(local_max, Traits::abs2(vals_h2[i]));
    }
    float vec_max = max(Traits::to_float(Traits::low(local_max)),
                        Traits::to_float(Traits::high(local_max)));

    float sf = vec_max / 6.0f;
    uint8_t sf_fp8;
    reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8) = __nv_fp8_e4m3(sf);
    sf = float(reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8));
    float sf_inv = (sf == 0.0f) ? 0.0f : 1.0f / sf;

    float2 X_f2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X_f2[i] = Traits::to_float2(vals_h2[i]);
        X_f2[i].x *= sf_inv;
        X_f2[i].y *= sf_inv;
    }

    uint32_t fp4_packed[2];
    fp4_packed[0] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[0].y, X_f2[0].x, X_f2[1].y, X_f2[1].x,
        X_f2[2].y, X_f2[2].x, X_f2[3].y, X_f2[3].x);
    fp4_packed[1] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[4].y, X_f2[4].x, X_f2[5].y, X_f2[5].x,
        X_f2[6].y, X_f2[6].x, X_f2[7].y, X_f2[7].x);

    const int seq_offset = seq_block_id * SEQ_PER_BLOCK + seq_chunk * ELEMENTS_PER_THREAD;

    __nv_fp4x2_e2m1* out = X_fp4 + batch_id * HEAD_DIM * (padded_seq / 2)
                          + dim_id * (padded_seq / 2)
                          + seq_offset / 2;
    reinterpret_cast<uint64_t*>(out)[0] = reinterpret_cast<uint64_t*>(fp4_packed)[0];

    X_scale[batch_id * HEAD_DIM * (padded_seq / 16)
            + dim_id * (padded_seq / 16)
            + seq_block_id * (SEQ_PER_BLOCK / 16) + seq_chunk] =
        reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8);
}

template<typename T, int HEAD_DIM>
static void nvfp4_quantise_transpose_launch(
    const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale,
    int bs, int seq_len) {

    constexpr int SEQ_PER_BLOCK = 128;
    constexpr int TB_SIZE = SEQ_PER_BLOCK * HEAD_DIM / ELEMENTS_PER_THREAD;

    const int num_blocks = bs * ta_cdiv(seq_len, SEQ_PER_BLOCK);
    const int padded_seq = ta_cdiv(seq_len, SEQ_PER_BLOCK) * SEQ_PER_BLOCK;

    nvfp4_quantise_transpose_kernel<T, HEAD_DIM, SEQ_PER_BLOCK>
        <<<num_blocks, TB_SIZE>>>(X, X_fp4, X_scale, bs, seq_len, padded_seq);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "nvfp4_quantise_transpose_kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

template<typename T>
static void dispatch_nvfp4_quantise_transpose(
    const void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim) {

    auto X = reinterpret_cast<const T*>(X_raw);
    auto X_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(X_fp4_raw);
    auto X_scale = reinterpret_cast<__nv_fp8_e4m3*>(X_scale_raw);

    if (head_dim == 64)
        nvfp4_quantise_transpose_launch<T, 64>(X, X_fp4, X_scale, bs, seq_len);
    else if (head_dim == 128)
        nvfp4_quantise_transpose_launch<T, 128>(X, X_fp4, X_scale, bs, seq_len);
    else
        fprintf(stderr, "nvfp4_quantise_transpose: unsupported head_dim=%d (must be 64 or 128)\n", head_dim);
}

void nvfp4_quantise_transpose(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_nvfp4_quantise_transpose<__nv_bfloat16>(
            X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim);
    } else {
        dispatch_nvfp4_quantise_transpose<half>(
            X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim);
    }
}

template<typename T, int HEAD_DIM, int SEQ_PER_BLOCK>
__global__
void nvfp4_quantise_transpose_permute_seq_kernel(
    const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale,
    int bs, int seq_len, int padded_seq, bool inverse)
{
    using Traits = PrecisionTraits<T>;
    constexpr int THREADS_PER_HEAD = HEAD_DIM / ELEMENTS_PER_THREAD;
    constexpr int THREADS_PER_SEQ  = SEQ_PER_BLOCK / ELEMENTS_PER_THREAD;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_seq_blocks = ta_cdiv(seq_len, SEQ_PER_BLOCK);
    const int batch_id = bid / num_seq_blocks;
    const int seq_block_id = bid % num_seq_blocks;

    const int load_phys_local = tid / THREADS_PER_HEAD;
    const int load_head_chunk = tid % THREADS_PER_HEAD;
    const int load_logical_local = ta_sage_perm_seq(load_phys_local, inverse);
    const int load_logical_global = seq_block_id * SEQ_PER_BLOCK + load_logical_local;
    const int load_phys_global = seq_block_id * SEQ_PER_BLOCK + load_phys_local;

    const T* X_ptr = X + batch_id * seq_len * HEAD_DIM
                    + load_logical_global * HEAD_DIM
                    + load_head_chunk * ELEMENTS_PER_THREAD;

    uint32_t X_reg[2][4];
    if (load_logical_global < seq_len && load_phys_global < seq_len) {
        asm volatile(
            "ld.global.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
            "ld.global.v4.b32 {%4, %5, %6, %7}, [%8+16];\n\t"
            : "=r"(X_reg[0][0]), "=r"(X_reg[0][1]), "=r"(X_reg[0][2]), "=r"(X_reg[0][3]),
              "=r"(X_reg[1][0]), "=r"(X_reg[1][1]), "=r"(X_reg[1][2]), "=r"(X_reg[1][3])
            : "l"(X_ptr)
        );
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            X_reg[0][i] = 0;
            X_reg[1][i] = 0;
        }
    }

    __shared__ T smem[SEQ_PER_BLOCK * HEAD_DIM];
    T* smem_row = smem + load_phys_local * HEAD_DIM + load_head_chunk * ELEMENTS_PER_THREAD;
    *reinterpret_cast<uint4*>(smem_row)     = *reinterpret_cast<uint4*>(X_reg[0]);
    *reinterpret_cast<uint4*>(smem_row + 8) = *reinterpret_cast<uint4*>(X_reg[1]);

    __syncthreads();

    const int dim_id    = tid / THREADS_PER_SEQ;
    const int seq_chunk = tid % THREADS_PER_SEQ;

    typename Traits::vec2 vals_h2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int s0 = seq_chunk * ELEMENTS_PER_THREAD + 2 * i;
        int s1 = s0 + 1;
        vals_h2[i] = Traits::make_vec2(smem[s0 * HEAD_DIM + dim_id],
                                       smem[s1 * HEAD_DIM + dim_id]);
    }

    typename Traits::vec2 local_max = Traits::abs2(vals_h2[0]);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        local_max = Traits::max2(local_max, Traits::abs2(vals_h2[i]));
    }
    float vec_max = max(Traits::to_float(Traits::low(local_max)),
                        Traits::to_float(Traits::high(local_max)));

    float sf = vec_max / 6.0f;
    uint8_t sf_fp8;
    reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8) = __nv_fp8_e4m3(sf);
    sf = float(reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8));
    float sf_inv = (sf == 0.0f) ? 0.0f : 1.0f / sf;

    float2 X_f2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X_f2[i] = Traits::to_float2(vals_h2[i]);
        X_f2[i].x *= sf_inv;
        X_f2[i].y *= sf_inv;
    }

    uint32_t fp4_packed[2];
    fp4_packed[0] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[0].y, X_f2[0].x, X_f2[1].y, X_f2[1].x,
        X_f2[2].y, X_f2[2].x, X_f2[3].y, X_f2[3].x);
    fp4_packed[1] = ta_cvt_8xf32_to_e2m1_packed(
        X_f2[4].y, X_f2[4].x, X_f2[5].y, X_f2[5].x,
        X_f2[6].y, X_f2[6].x, X_f2[7].y, X_f2[7].x);

    const int seq_offset = seq_block_id * SEQ_PER_BLOCK + seq_chunk * ELEMENTS_PER_THREAD;

    __nv_fp4x2_e2m1* out = X_fp4 + batch_id * HEAD_DIM * (padded_seq / 2)
                          + dim_id * (padded_seq / 2)
                          + seq_offset / 2;
    reinterpret_cast<uint64_t*>(out)[0] = reinterpret_cast<uint64_t*>(fp4_packed)[0];

    X_scale[batch_id * HEAD_DIM * (padded_seq / 16)
            + dim_id * (padded_seq / 16)
            + seq_block_id * (SEQ_PER_BLOCK / 16) + seq_chunk] =
        reinterpret_cast<__nv_fp8_e4m3&>(sf_fp8);
}

template<typename T, int HEAD_DIM>
static void nvfp4_quantise_transpose_permute_seq_launch(
    const T* X, __nv_fp4x2_e2m1* X_fp4, __nv_fp8_e4m3* X_scale,
    int bs, int seq_len, bool inverse) {

    constexpr int SEQ_PER_BLOCK = 128;
    constexpr int TB_SIZE = SEQ_PER_BLOCK * HEAD_DIM / ELEMENTS_PER_THREAD;

    const int num_blocks = bs * ta_cdiv(seq_len, SEQ_PER_BLOCK);
    const int padded_seq = ta_cdiv(seq_len, SEQ_PER_BLOCK) * SEQ_PER_BLOCK;

    nvfp4_quantise_transpose_permute_seq_kernel<T, HEAD_DIM, SEQ_PER_BLOCK>
        <<<num_blocks, TB_SIZE>>>(X, X_fp4, X_scale, bs, seq_len, padded_seq, inverse);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "nvfp4_quantise_transpose_permute_seq_kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

template<typename T>
static void dispatch_nvfp4_quantise_transpose_permute_seq(
    const void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool inverse) {

    auto X = reinterpret_cast<const T*>(X_raw);
    auto X_fp4 = reinterpret_cast<__nv_fp4x2_e2m1*>(X_fp4_raw);
    auto X_scale = reinterpret_cast<__nv_fp8_e4m3*>(X_scale_raw);

    if (head_dim == 64)
        nvfp4_quantise_transpose_permute_seq_launch<T, 64>(X, X_fp4, X_scale, bs, seq_len, inverse);
    else if (head_dim == 128)
        nvfp4_quantise_transpose_permute_seq_launch<T, 128>(X, X_fp4, X_scale, bs, seq_len, inverse);
    else
        fprintf(stderr, "nvfp4_quantise_transpose_permute_seq: unsupported head_dim=%d (must be 64 or 128)\n", head_dim);
}

void nvfp4_quantise_transpose_permute_seq(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool inverse,
    bool is_bf16) {
    if (is_bf16) {
        dispatch_nvfp4_quantise_transpose_permute_seq<__nv_bfloat16>(
            X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim, inverse);
    } else {
        dispatch_nvfp4_quantise_transpose_permute_seq<half>(
            X_raw, X_fp4_raw, X_scale_raw, bs, seq_len, head_dim, inverse);
    }
}
