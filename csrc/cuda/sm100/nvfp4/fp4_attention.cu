// SM100 NVFP4 packed attention kernel.

#include <cstdint>
#include <float.h>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <c10/cuda/CUDAException.h>

#include "thriftattention/sm120/cuda_common.cuh"

namespace {

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t value) {
    float result = 0.0f;
    switch (value & 0x7) {
        case 1: result = 0.5f; break;
        case 2: result = 1.0f; break;
        case 3: result = 1.5f; break;
        case 4: result = 2.0f; break;
        case 5: result = 3.0f; break;
        case 6: result = 4.0f; break;
        case 7: result = 6.0f; break;
        default: break;
    }
    return (value & 0x8) ? -result : result;
}

__device__ __forceinline__ float fp8_e4m3_to_float(__nv_fp8_e4m3 value) {
    return float(value);
}

__device__ __forceinline__ float load_transposed_nvfp4(
    const __nv_fp4x2_e2m1* values,
    const __nv_fp8_e4m3* scales,
    int dim,
    int seq,
    int padded_seq) {
    const int packed_stride = padded_seq / 2;
    const int scale_stride = padded_seq / 16;
    const uint8_t packed = reinterpret_cast<const uint8_t*>(values)[dim * packed_stride + seq / 2];
    const uint8_t nibble = (seq & 1) ? (packed >> 4) : (packed & 0xf);
    const float scale = fp8_e4m3_to_float(scales[dim * scale_stride + seq / 16]);
    return fp4_e2m1_to_float(nibble) * scale;
}

__device__ __forceinline__ uint32_t shared_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 shared_addr64; cvta.to.shared.u64 shared_addr64, %1; cvt.u32.u64 %0, shared_addr64; }\n"
        : "=r"(addr)
        : "l"(ptr));
    return addr;
}

__device__ __forceinline__ uint32_t pack_scale_word(const __nv_fp8_e4m3* ptr) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(ptr);
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

__device__ __forceinline__ int swizzle_k_sw32(int logical_byte) {
    return logical_byte ^ (((logical_byte >> 7) & 1) << 4);
}

template<typename T, bool CAUSAL, bool K_PERMUTED, int HEAD_DIM>
__launch_bounds__(1024)
__global__
void sm100_fp4_attention_qk_tcgen05_kernel(
    const __nv_fp4x2_e2m1* __restrict__ Q,
    const __nv_fp4x2_e2m1* __restrict__ K,
    const __nv_fp4x2_e2m1* __restrict__ V,
    const __nv_fp8_e4m3* __restrict__ S_Q,
    const __nv_fp8_e4m3* __restrict__ S_K,
    const __nv_fp8_e4m3* __restrict__ S_V,
    T* __restrict__ O,
    int flat_q_heads,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads) {
    constexpr int BLOCK_Q = 128;
    constexpr int BLOCK_KV = 128;
    constexpr int HEAD_DIM_PACKED = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_PACKED = MMA_K / 2;
    constexpr int MMA_SCALE_DIM = MMA_K / 16;
    constexpr int MMA_CHUNKS = HEAD_DIM / MMA_K;
    constexpr int SCORE_PARTS = 8;
    constexpr int THREADS = 1024;
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "HEAD_DIM must be 64 or 128");
    constexpr uint32_t IDESC_M128N128K64_NVFP4 =
        (1u << 7) |
        (1u << 10) |
        (16u << 17) |
        (1u << 27);
    constexpr uint64_t SMEM_DESC_BASE_K_SW32 =
        (1ull << 16) |
        (16ull << 32) |
        (1ull << 46) |
        (6ull << 61);

    const int q_tile = blockIdx.x;
    const int q_bid = blockIdx.y;
    if (q_bid >= flat_q_heads) {
        return;
    }

    const int q_start = q_tile * BLOCK_Q;
    const int batch_id = q_bid / num_q_heads;
    const int q_head = q_bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;
    const int padded_kv = ta_cdiv(kv_capacity, 128) * 128;
    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));

    const __nv_fp4x2_e2m1* q_values = Q + (q_bid * q_len + q_start) * HEAD_DIM_PACKED;
    const __nv_fp8_e4m3* q_scales = S_Q + (q_bid * q_len + q_start) * SCALE_DIM;
    const __nv_fp4x2_e2m1* k_values = K + kv_bid * kv_capacity * HEAD_DIM_PACKED;
    const __nv_fp8_e4m3* k_scales = S_K + kv_bid * kv_capacity * SCALE_DIM;
    const __nv_fp4x2_e2m1* v_values = V + kv_bid * HEAD_DIM * (padded_kv / 2);
    const __nv_fp8_e4m3* v_scales = S_V + kv_bid * HEAD_DIM * (padded_kv / 16);
    T* out = O + (q_bid * q_len + q_start) * HEAD_DIM;

    __shared__ __align__(8) uint64_t mma_barrier;
    __shared__ uint32_t tmem_addr;
    extern __shared__ uint8_t shared_raw[];

    const uint32_t shared_base = shared_u32(shared_raw);
    const uint32_t q_smem_addr = (shared_base + 1023u) & ~1023u;
    const uint32_t k_smem_addr = q_smem_addr + BLOCK_Q * MMA_PACKED;
    uint8_t* q_smem = shared_raw + (q_smem_addr - shared_base);
    uint8_t* k_smem = q_smem + BLOCK_Q * MMA_PACKED;
    half* probs_half = reinterpret_cast<half*>(k_smem + BLOCK_KV * MMA_PACKED);
    half* v_half = probs_half + BLOCK_Q * BLOCK_KV;
    float* output_acc = reinterpret_cast<float*>(v_half + BLOCK_KV * HEAD_DIM);
    float* row_max = output_acc + BLOCK_Q * HEAD_DIM;
    float* row_sum = row_max + BLOCK_Q;
    float* row_rescale = row_sum + BLOCK_Q;
    float* row_partials = row_rescale + BLOCK_Q;

    for (int idx = threadIdx.x; idx < BLOCK_Q * HEAD_DIM; idx += THREADS) {
        output_acc[idx] = 0.0f;
    }
    if (threadIdx.x < BLOCK_Q) {
        row_max[threadIdx.x] = -FLT_MAX;
        row_sum[threadIdx.x] = 0.0f;
        row_rescale[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x == 0) {
        mma_barrier = 0;
    }
    __syncthreads();

    const uint32_t tmem_addr_shared = shared_u32(&tmem_addr);
    if ((threadIdx.x >> 5) == 0) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;\n"
            :
            : "r"(tmem_addr_shared)
            : "memory");
    }
    __syncthreads();

    const uint32_t d_tmem = tmem_addr;
    const uint32_t scale_q_tmem = d_tmem + 128;
    const uint32_t scale_k_tmem = d_tmem + 160;

    const int kv_tile_limit = CAUSAL ? min(kv_len, q_start + BLOCK_Q) : kv_len;
    for (int k_start = 0; k_start < kv_tile_limit; k_start += BLOCK_KV) {
        const int valid_kv = min(BLOCK_KV, kv_len - k_start);
        #pragma unroll
        for (int mma_chunk = 0; mma_chunk < MMA_CHUNKS; ++mma_chunk) {
            for (int idx = threadIdx.x; idx < BLOCK_Q * MMA_PACKED; idx += THREADS) {
                const int row = idx / MMA_PACKED;
                const int byte_col = idx - row * MMA_PACKED;
                const int q_row = q_start + row;
                const int smem_idx = swizzle_k_sw32(row * MMA_PACKED + byte_col);
                const int src_idx = row * HEAD_DIM_PACKED + mma_chunk * MMA_PACKED + byte_col;
                q_smem[smem_idx] = q_row < q_len ? reinterpret_cast<const uint8_t*>(q_values)[src_idx] : 0;
            }
            for (int idx = threadIdx.x; idx < BLOCK_KV * MMA_PACKED; idx += THREADS) {
                const int row = idx / MMA_PACKED;
                const int byte_col = idx - row * MMA_PACKED;
                const int logical_k_row = k_start + row;
                const int physical_k_row = K_PERMUTED ? ta_kv_physical_from_logical(logical_k_row) : logical_k_row;
                const int smem_idx = swizzle_k_sw32(row * MMA_PACKED + byte_col);
                const int src_idx = physical_k_row * HEAD_DIM_PACKED + mma_chunk * MMA_PACKED + byte_col;
                k_smem[smem_idx] = logical_k_row < kv_len && physical_k_row < kv_capacity
                    ? reinterpret_cast<const uint8_t*>(k_values)[src_idx]
                    : 0;
            }
            __syncthreads();

            const int warp = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            const int row = warp * 32 + lane;
            const uint32_t lane_base = static_cast<uint32_t>(warp * 32) << 16;

            if (warp < 4) {
                const int q_row = q_start + row;
                const uint32_t q_scale_word = q_row < q_len
                    ? pack_scale_word(q_scales + row * SCALE_DIM + mma_chunk * MMA_SCALE_DIM)
                    : 0;
                for (int col = 0; col < 32; ++col) {
                    const uint32_t q_addr = scale_q_tmem + static_cast<uint32_t>(col) + lane_base;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                        :
                        : "r"(q_addr), "r"(q_scale_word)
                        : "memory");

                    const int n_group = col & 3;
                    const int logical_k_row = k_start + n_group * 32 + lane;
                    const int physical_k_row = K_PERMUTED ? ta_kv_physical_from_logical(logical_k_row) : logical_k_row;
                    const uint32_t k_scale_word = logical_k_row < kv_len && physical_k_row < kv_capacity
                        ? pack_scale_word(k_scales + physical_k_row * SCALE_DIM + mma_chunk * MMA_SCALE_DIM)
                        : 0;
                    const uint32_t k_addr = scale_k_tmem + static_cast<uint32_t>(col) + lane_base;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                        :
                        : "r"(k_addr), "r"(k_scale_word)
                        : "memory");
                }
                asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
            }
            __syncthreads();

            const uint64_t desc_q = SMEM_DESC_BASE_K_SW32 | ((static_cast<uint64_t>(q_smem_addr) & 0x3ffffull) >> 4);
            const uint64_t desc_k = SMEM_DESC_BASE_K_SW32 | ((static_cast<uint64_t>(k_smem_addr) & 0x3ffffull) >> 4);
            const uint32_t desc_q_lo = static_cast<uint32_t>(desc_q);
            const uint32_t desc_q_hi = static_cast<uint32_t>(desc_q >> 32);
            const uint32_t desc_k_lo = static_cast<uint32_t>(desc_k);
            const uint32_t desc_k_hi = static_cast<uint32_t>(desc_k >> 32);
            const uint32_t barrier_addr = shared_u32(&mma_barrier);
            const uint64_t barrier_generic = reinterpret_cast<uint64_t>(&mma_barrier);

            if (threadIdx.x == 0) {
                asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"(barrier_addr) : "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    ".reg .b64 desc_a;\n\t"
                    ".reg .b64 desc_b;\n\t"
                    "mov.b64 desc_a, {%1, %2};\n\t"
                    "mov.b64 desc_b, {%3, %4};\n\t"
                    "setp.ne.u32 p, %7, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
                    "[%0], desc_a, desc_b, %5, [%6], [%8], p;\n\t"
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%9];\n\t"
                    "wait_mma_done:\n\t"
                    ".reg .pred done;\n\t"
                    "mbarrier.try_wait.parity.b64 done, [%9], 0;\n\t"
                    "@!done bra wait_mma_done;\n\t"
                    "tcgen05.fence::after_thread_sync;\n\t"
                    "}\n"
                    :
                    : "r"(d_tmem),
                      "r"(desc_q_lo),
                      "r"(desc_q_hi),
                      "r"(desc_k_lo),
                      "r"(desc_k_hi),
                      "r"(IDESC_M128N128K64_NVFP4),
                      "r"(scale_q_tmem),
                      "r"(static_cast<uint32_t>(mma_chunk)),
                      "r"(scale_k_tmem),
                      "l"(barrier_generic)
                    : "memory");
            }
            __syncthreads();
        }

        const int score_warp = threadIdx.x >> 5;
        const int score_lane = threadIdx.x & 31;
        const int score_group = score_warp & 3;
        const int score_part = score_warp >> 2;
        const int score_row = score_group * 32 + score_lane;
        float partial_max = -FLT_MAX;
        if (score_warp < 4 * SCORE_PARTS) {
            const int q_row = q_start + score_row;
            const uint32_t score_lane_base = static_cast<uint32_t>(score_group * 32) << 16;
            float tile_max = -FLT_MAX;
            for (int col = score_part; col < BLOCK_KV; col += SCORE_PARTS) {
                uint32_t loaded = 0;
                const uint32_t addr = d_tmem + static_cast<uint32_t>(col) + score_lane_base;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
                    : "=r"(loaded)
                    : "r"(addr)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
                const int logical_k = k_start + col;
                const bool valid_score = q_row < q_len &&
                    logical_k < kv_len &&
                    (!CAUSAL || logical_k <= q_row);
                const float score = valid_score
                    ? __uint_as_float(loaded) * softmax_scale
                    : -FLT_MAX;
                tile_max = fmaxf(tile_max, score);
            }
            partial_max = tile_max;
            if (score_part > 0) {
                row_partials[(score_part - 1) * BLOCK_Q + score_row] = tile_max;
            }
        }
        __syncthreads();

        if (score_warp < 4) {
            float tile_max = partial_max;
            #pragma unroll
            for (int part = 1; part < SCORE_PARTS; ++part) {
                tile_max = fmaxf(tile_max, row_partials[(part - 1) * BLOCK_Q + score_row]);
            }
            const float new_max = fmaxf(row_max[score_row], tile_max);
            const float rescale = row_max[score_row] > -FLT_MAX * 0.5f
                ? __expf(row_max[score_row] - new_max)
                : 0.0f;
            row_rescale[score_row] = rescale;
            row_max[score_row] = new_max;
        }
        __syncthreads();

        if (threadIdx.x < BLOCK_Q) {
            row_sum[threadIdx.x] *= row_rescale[threadIdx.x];
        }
        for (int idx = threadIdx.x; idx < BLOCK_Q * HEAD_DIM; idx += THREADS) {
            const int out_row = idx / HEAD_DIM;
            output_acc[idx] *= row_rescale[out_row];
        }
        __syncthreads();

        float partial_sum = 0.0f;
        if (score_warp < 4 * SCORE_PARTS) {
            float tile_sum = 0.0f;
            const int q_row = q_start + score_row;
            const float max_value = row_max[score_row];
            const uint32_t score_lane_base = static_cast<uint32_t>(score_group * 32) << 16;
            for (int col = score_part; col < BLOCK_KV; col += SCORE_PARTS) {
                uint32_t loaded = 0;
                const uint32_t addr = d_tmem + static_cast<uint32_t>(col) + score_lane_base;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
                    : "=r"(loaded)
                    : "r"(addr)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
                const int logical_k = k_start + col;
                const bool valid_score = q_row < q_len &&
                    logical_k < kv_len &&
                    (!CAUSAL || logical_k <= q_row) &&
                    max_value > -FLT_MAX * 0.5f;
                const float score = valid_score
                    ? __uint_as_float(loaded) * softmax_scale
                    : -FLT_MAX;
                const int idx = score_row * BLOCK_KV + col;
                const float prob = valid_score ? __expf(score - max_value) : 0.0f;
                probs_half[idx] = __float2half(prob);
                tile_sum += prob;
            }
            partial_sum = tile_sum;
            if (score_part > 0) {
                row_partials[(score_part - 1) * BLOCK_Q + score_row] = tile_sum;
            }
        }
        __syncthreads();

        if (score_warp < 4) {
            float tile_sum = partial_sum;
            #pragma unroll
            for (int part = 1; part < SCORE_PARTS; ++part) {
                tile_sum += row_partials[(part - 1) * BLOCK_Q + score_row];
            }
            row_sum[score_row] += tile_sum;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < BLOCK_KV * HEAD_DIM; idx += THREADS) {
            const int kv_row = idx / HEAD_DIM;
            const int dim = idx - kv_row * HEAD_DIM;
            const float v = kv_row < valid_kv
                ? load_transposed_nvfp4(v_values, v_scales, dim, k_start + kv_row, padded_kv)
                : 0.0f;
            v_half[idx] = __float2half(v);
        }
        __syncthreads();

        const int warp_id = threadIdx.x >> 5;
        for (int tile = warp_id; tile < (BLOCK_Q / 16) * (HEAD_DIM / 16); tile += THREADS / 32) {
            const int m_tile = (tile / (HEAD_DIM / 16)) * 16;
            const int n_tile = (tile % (HEAD_DIM / 16)) * 16;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
            nvcuda::wmma::load_matrix_sync(
                c_frag,
                output_acc + m_tile * HEAD_DIM + n_tile,
                HEAD_DIM,
                nvcuda::wmma::mem_row_major);
            for (int k_tile = 0; k_tile < BLOCK_KV; k_tile += 16) {
                nvcuda::wmma::load_matrix_sync(a_frag, probs_half + m_tile * BLOCK_KV + k_tile, BLOCK_KV);
                nvcuda::wmma::load_matrix_sync(b_frag, v_half + k_tile * HEAD_DIM + n_tile, HEAD_DIM);
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            nvcuda::wmma::store_matrix_sync(
                output_acc + m_tile * HEAD_DIM + n_tile,
                c_frag,
                HEAD_DIM,
                nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < BLOCK_Q * HEAD_DIM; idx += THREADS) {
        const int row = idx / HEAD_DIM;
        const int dim = idx - row * HEAD_DIM;
        const int q_row = q_start + row;
        if (q_row < q_len) {
            const float denom = row_sum[row];
            const float value = denom > 0.0f ? output_acc[idx] / denom : 0.0f;
            out[row * HEAD_DIM + dim] = PrecisionTraits<T>::from_float(value);
        }
    }
    __syncthreads();

    if ((threadIdx.x >> 5) == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;\n"
            :
            : "r"(d_tmem)
            : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
}

template<typename T, bool CAUSAL, bool K_PERMUTED, int HEAD_DIM>
void sm100_launch_fp4_attention_qk_tcgen05(
    const void* Q,
    const void* K,
    const void* V,
    const void* S_Q,
    const void* S_K,
    const void* S_V,
    void* O,
    int flat_q_heads,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads) {
    constexpr int block_q = 128;
    constexpr int block_kv = 128;
    constexpr int head_dim = HEAD_DIM;
    constexpr int mma_packed = 64 / 2;
    constexpr int score_parts = 8;
    constexpr int threads = 1024;
    const dim3 grid(ta_cdiv(q_len, block_q), flat_q_heads);
    const int smem_bytes = 1024
        + block_q * mma_packed
        + block_kv * mma_packed
        + block_q * block_kv * static_cast<int>(sizeof(half))
        + block_kv * head_dim * static_cast<int>(sizeof(half))
        + block_q * head_dim * static_cast<int>(sizeof(float))
        + (3 + score_parts - 1) * block_q * static_cast<int>(sizeof(float));

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sm100_fp4_attention_qk_tcgen05_kernel<T, CAUSAL, K_PERMUTED, HEAD_DIM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    sm100_fp4_attention_qk_tcgen05_kernel<T, CAUSAL, K_PERMUTED, HEAD_DIM><<<grid, threads, smem_bytes>>>(
        reinterpret_cast<const __nv_fp4x2_e2m1*>(Q),
        reinterpret_cast<const __nv_fp4x2_e2m1*>(K),
        reinterpret_cast<const __nv_fp4x2_e2m1*>(V),
        reinterpret_cast<const __nv_fp8_e4m3*>(S_Q),
        reinterpret_cast<const __nv_fp8_e4m3*>(S_K),
        reinterpret_cast<const __nv_fp8_e4m3*>(S_V),
        reinterpret_cast<T*>(O),
        flat_q_heads,
        q_len,
        kv_len,
        kv_capacity,
        num_q_heads,
        num_kv_heads);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T, bool CAUSAL, bool K_PERMUTED>
void sm100_dispatch_fp4_attention(
    const void* Q,
    const void* K,
    const void* V,
    const void* S_Q,
    const void* S_K,
    const void* S_V,
    void* O,
    int flat_q_heads,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
    if (head_dim == 64) {
        sm100_launch_fp4_attention_qk_tcgen05<T, CAUSAL, K_PERMUTED, 64>(
            Q, K, V, S_Q, S_K, S_V, O, flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads);
    } else if (head_dim == 128) {
        sm100_launch_fp4_attention_qk_tcgen05<T, CAUSAL, K_PERMUTED, 128>(
            Q, K, V, S_Q, S_K, S_V, O, flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads);
    } else {
        C10_THROW_ERROR(ValueError, "sm100_fp4_attention supports only head_dim 64 or 128");
    }
}

template<bool CAUSAL, bool K_PERMUTED>
void sm100_dispatch_fp4_attention_dtype(
    const void* Q,
    const void* K,
    const void* V,
    const void* S_Q,
    const void* S_K,
    const void* S_V,
    void* O,
    int flat_q_heads,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        sm100_dispatch_fp4_attention<__nv_bfloat16, CAUSAL, K_PERMUTED>(
            Q, K, V, S_Q, S_K, S_V, O, flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim);
    } else {
        sm100_dispatch_fp4_attention<half, CAUSAL, K_PERMUTED>(
            Q, K, V, S_Q, S_K, S_V, O, flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim);
    }
}

}  // namespace

void sm100_fp4_attention_causal_nvfp4(
    const void* Q,
    const void* K,
    const void* V,
    const void* S_Q,
    const void* S_K,
    const void* S_V,
    void* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    sm100_dispatch_fp4_attention_dtype<true, true>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity,
        num_q_heads, num_kv_heads, head_dim, is_bf16);
}

void sm100_fp4_attention_noncausal_nvfp4(
    const void* Q,
    const void* K,
    const void* V,
    const void* S_Q,
    const void* S_K,
    const void* S_V,
    void* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    sm100_dispatch_fp4_attention_dtype<false, true>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity,
        num_q_heads, num_kv_heads, head_dim, is_bf16);
}

void sm100_fp4_attention_single_query_nvfp4(
    const void* Q,
    const void* K,
    const void* V,
    const void* S_Q,
    const void* S_K,
    const void* S_V,
    void* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    sm100_dispatch_fp4_attention_dtype<false, false>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity,
        num_q_heads, num_kv_heads, head_dim, is_bf16);
}
