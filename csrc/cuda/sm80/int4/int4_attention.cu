#include <cstdint>
#include <float.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "thriftattention/sm80/cuda_common.cuh"

#define FULL_MASK 0xffffffff

template <typename T>
__device__ inline T int4_attention_from_float(float value);

template <>
__device__ inline half int4_attention_from_float<half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 int4_attention_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

__device__ inline int8_t unpack_s4(uint8_t packed, int index) {
    const int nibble = (packed >> (4 * index)) & 0xf;
    return static_cast<int8_t>(nibble >= 8 ? nibble - 16 : nibble);
}

__device__ inline uint32_t load_packed_s4x8(const uint8_t* ptr) {
    return *reinterpret_cast<const uint32_t*>(ptr);
}

template <int HEAD_DIM, int SCALE_DIM, int KV_CHUNK>
__device__ void compute_int4_scores_mma_tile(
    const uint8_t* q_chunk,
    const uint8_t* k_chunk,
    const float* sq_chunk,
    const float* sk_chunk,
    float* scores,
    int kv_tile,
    int q_block_start,
    int q_len,
    int kv_start,
    int kv_len,
    bool causal)
{
    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;

    const int lane = threadIdx.x;
    const int tid4 = lane & 3;
    const int groupID = lane >> 2;
    const int row0 = groupID;
    const int row1 = groupID + 8;
    const int col0 = tid4 * 2;
    const int col1 = tid4 * 2 + 1;

    float score0 = 0.0f;
    float score1 = 0.0f;
    float score2 = 0.0f;
    float score3 = 0.0f;

    #pragma unroll
    for (int group = 0; group < SCALE_DIM; group++) {
        const int byte_offset = group * 32;
        uint32_t a_frag[4];
        uint32_t b_frag[2];
        int32_t acc[4] = {0, 0, 0, 0};

        a_frag[0] = load_packed_s4x8(q_chunk + row0 * HEAD_DIM_2 + byte_offset + tid4 * 4);
        a_frag[1] = load_packed_s4x8(q_chunk + row1 * HEAD_DIM_2 + byte_offset + tid4 * 4);
        a_frag[2] = load_packed_s4x8(q_chunk + row0 * HEAD_DIM_2 + byte_offset + 16 + tid4 * 4);
        a_frag[3] = load_packed_s4x8(q_chunk + row1 * HEAD_DIM_2 + byte_offset + 16 + tid4 * 4);

        const int k_row = kv_tile + groupID;
        b_frag[0] = load_packed_s4x8(k_chunk + k_row * HEAD_DIM_2 + byte_offset + tid4 * 4);
        b_frag[1] = load_packed_s4x8(k_chunk + k_row * HEAD_DIM_2 + byte_offset + 16 + tid4 * 4);

        ta_mma_m16n8k64_s4(a_frag, b_frag, acc);

        score0 += static_cast<float>(acc[0]) * sq_chunk[row0 * SCALE_DIM + group] * sk_chunk[(kv_tile + col0) * SCALE_DIM + group];
        score1 += static_cast<float>(acc[1]) * sq_chunk[row0 * SCALE_DIM + group] * sk_chunk[(kv_tile + col1) * SCALE_DIM + group];
        score2 += static_cast<float>(acc[2]) * sq_chunk[row1 * SCALE_DIM + group] * sk_chunk[(kv_tile + col0) * SCALE_DIM + group];
        score3 += static_cast<float>(acc[3]) * sq_chunk[row1 * SCALE_DIM + group] * sk_chunk[(kv_tile + col1) * SCALE_DIM + group];
    }

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
    const int q0 = q_block_start + row0;
    const int q1 = q_block_start + row1;
    const int k0 = kv_start + kv_tile + col0;
    const int k1 = kv_start + kv_tile + col1;

    const bool q0_valid = q0 < q_len;
    const bool q1_valid = q1 < q_len;
    const bool k0_valid = k0 < kv_len;
    const bool k1_valid = k1 < kv_len;

    scores[row0 * KV_CHUNK + kv_tile + col0] = (q0_valid && k0_valid && (!causal || k0 <= q0)) ? score0 * softmax_scale : -INFINITY;
    scores[row0 * KV_CHUNK + kv_tile + col1] = (q0_valid && k1_valid && (!causal || k1 <= q0)) ? score1 * softmax_scale : -INFINITY;
    scores[row1 * KV_CHUNK + kv_tile + col0] = (q1_valid && k0_valid && (!causal || k0 <= q1)) ? score2 * softmax_scale : -INFINITY;
    scores[row1 * KV_CHUNK + kv_tile + col1] = (q1_valid && k1_valid && (!causal || k1 <= q1)) ? score3 * softmax_scale : -INFINITY;
}

template <typename T, bool CAUSAL, int HEAD_DIM, int SCALE_DIM, int BLOCK_Q>
__global__ void int4_attention_kernel_mma_qk(
    const uint8_t* Q,
    const uint8_t* K,
    const uint8_t* V,
    const float* S_Q,
    const float* S_K,
    const float* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int KV_CHUNK = 64;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int q_blocks = ta_cdiv(q_len, BLOCK_Q);
    const int q_block_idx = blockIdx.x % q_blocks;
    const int q_head = (blockIdx.x / q_blocks) % num_q_heads;
    const int batch = blockIdx.x / (q_blocks * num_q_heads);
    const int kv_head = q_head * num_kv_heads / num_q_heads;
    const int q_block_start = q_block_idx * BLOCK_Q;
    const int out_d = tid;
    const bool has_output = out_d < HEAD_DIM;
    const int v_group = out_d / 64;

    __shared__ uint8_t q_chunk[BLOCK_Q][HEAD_DIM_2];
    __shared__ uint8_t k_chunk[KV_CHUNK][HEAD_DIM_2];
    __shared__ uint8_t v_chunk[KV_CHUNK][HEAD_DIM_2];
    __shared__ float sq_chunk[BLOCK_Q][SCALE_DIM];
    __shared__ float sk_chunk[KV_CHUNK][SCALE_DIM];
    __shared__ float sv_chunk[KV_CHUNK][SCALE_DIM];
    __shared__ float scores[BLOCK_Q][KV_CHUNK];
    __shared__ float reduce_mem[4];

    float running_max[BLOCK_Q];
    float running_denom[BLOCK_Q];
    float running_acc[BLOCK_Q];

    #pragma unroll
    for (int q_i = 0; q_i < BLOCK_Q; q_i++) {
        running_max[q_i] = -INFINITY;
        running_denom[q_i] = 0.0f;
        running_acc[q_i] = 0.0f;
    }

    for (int idx = tid; idx < BLOCK_Q * HEAD_DIM_2; idx += blockDim.x) {
        const int q_i = idx / HEAD_DIM_2;
        const int d = idx % HEAD_DIM_2;
        const int q_token = q_block_start + q_i;
        if (q_token < q_len) {
            const int src = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM_2 + d;
            q_chunk[q_i][d] = Q[src];
        } else {
            q_chunk[q_i][d] = 0;
        }
    }

    for (int idx = tid; idx < BLOCK_Q * SCALE_DIM; idx += blockDim.x) {
        const int q_i = idx / SCALE_DIM;
        const int g = idx % SCALE_DIM;
        const int q_token = q_block_start + q_i;
        if (q_token < q_len) {
            const int src = ((batch * q_len + q_token) * num_q_heads + q_head) * SCALE_DIM + g;
            sq_chunk[q_i][g] = S_Q[src];
        } else {
            sq_chunk[q_i][g] = 0.0f;
        }
    }

    __syncthreads();

    for (int kv_start = 0; kv_start < kv_len; kv_start += KV_CHUNK) {
        if constexpr (CAUSAL) {
            if (kv_start > q_block_start + BLOCK_Q - 1) {
                break;
            }
        }

        for (int idx = tid; idx < KV_CHUNK * HEAD_DIM_2; idx += blockDim.x) {
            const int local_token = idx / HEAD_DIM_2;
            const int d = idx % HEAD_DIM_2;
            const int kv_token = kv_start + local_token;
            if (kv_token < kv_len) {
                const int src = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * HEAD_DIM_2 + d;
                k_chunk[local_token][d] = K[src];
                v_chunk[local_token][d] = V[src];
            } else {
                k_chunk[local_token][d] = 0;
                v_chunk[local_token][d] = 0;
            }
        }

        for (int idx = tid; idx < KV_CHUNK * SCALE_DIM; idx += blockDim.x) {
            const int local_token = idx / SCALE_DIM;
            const int g = idx % SCALE_DIM;
            const int kv_token = kv_start + local_token;
            if (kv_token < kv_len) {
                const int src = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * SCALE_DIM + g;
                sk_chunk[local_token][g] = S_K[src];
                sv_chunk[local_token][g] = S_V[src];
            } else {
                sk_chunk[local_token][g] = 0.0f;
                sv_chunk[local_token][g] = 0.0f;
            }
        }

        __syncthreads();

        if (tid < 32) {
            #pragma unroll
            for (int kv_tile = 0; kv_tile < KV_CHUNK; kv_tile += 8) {
                compute_int4_scores_mma_tile<HEAD_DIM, SCALE_DIM, KV_CHUNK>(
                    &q_chunk[0][0], &k_chunk[0][0], &sq_chunk[0][0], &sk_chunk[0][0],
                    &scores[0][0], kv_tile, q_block_start, q_len, kv_start, kv_len, CAUSAL);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int q_i = 0; q_i < BLOCK_Q; q_i++) {
            const int q_token = q_block_start + q_i;
            const bool valid_q = q_token < q_len;
            float local_max = -INFINITY;

            for (int local_token = tid; local_token < KV_CHUNK; local_token += blockDim.x) {
                local_max = fmaxf(local_max, scores[q_i][local_token]);
            }

            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
            }
            if (lane == 0) {
                reduce_mem[warp] = local_max;
            }
            __syncthreads();

            float chunk_max = -INFINITY;
            if (warp == 0) {
                chunk_max = lane < 4 ? reduce_mem[lane] : -INFINITY;
                for (int offset = 16; offset > 0; offset /= 2) {
                    chunk_max = fmaxf(chunk_max, __shfl_down_sync(FULL_MASK, chunk_max, offset));
                }
                if (lane == 0) {
                    reduce_mem[0] = chunk_max;
                }
            }
            __syncthreads();
            chunk_max = reduce_mem[0];

            float local_denom = 0.0f;
            for (int local_token = tid; local_token < KV_CHUNK; local_token += blockDim.x) {
                const float score = scores[q_i][local_token];
                if (isfinite(score) && isfinite(chunk_max)) {
                    local_denom += expf(score - chunk_max);
                }
            }

            for (int offset = 16; offset > 0; offset /= 2) {
                local_denom += __shfl_down_sync(FULL_MASK, local_denom, offset);
            }
            if (lane == 0) {
                reduce_mem[warp] = local_denom;
            }
            __syncthreads();

            float chunk_denom = 0.0f;
            if (warp == 0) {
                chunk_denom = lane < 4 ? reduce_mem[lane] : 0.0f;
                for (int offset = 16; offset > 0; offset /= 2) {
                    chunk_denom += __shfl_down_sync(FULL_MASK, chunk_denom, offset);
                }
                if (lane == 0) {
                    reduce_mem[0] = chunk_denom;
                }
            }
            __syncthreads();
            chunk_denom = reduce_mem[0];

            float chunk_acc = 0.0f;
            if (valid_q && has_output && chunk_denom > 0.0f) {
                for (int local_token = 0; local_token < KV_CHUNK; local_token++) {
                    const float score = scores[q_i][local_token];
                    if (!isfinite(score)) {
                        continue;
                    }
                    const float weight = expf(score - chunk_max);
                    const uint8_t packed_v = v_chunk[local_token][out_d / 2];
                    const float v_real = static_cast<float>(unpack_s4(packed_v, out_d & 1)) * sv_chunk[local_token][v_group];
                    chunk_acc += weight * v_real;
                }
            }

            if (chunk_denom > 0.0f) {
                if (running_denom[q_i] == 0.0f) {
                    running_max[q_i] = chunk_max;
                    running_denom[q_i] = chunk_denom;
                    running_acc[q_i] = chunk_acc;
                } else {
                    const float new_max = fmaxf(running_max[q_i], chunk_max);
                    const float old_scale = expf(running_max[q_i] - new_max);
                    const float chunk_scale = expf(chunk_max - new_max);
                    running_acc[q_i] = running_acc[q_i] * old_scale + chunk_acc * chunk_scale;
                    running_denom[q_i] = running_denom[q_i] * old_scale + chunk_denom * chunk_scale;
                    running_max[q_i] = new_max;
                }
            }

            __syncthreads();
        }
    }

    #pragma unroll
    for (int q_i = 0; q_i < BLOCK_Q; q_i++) {
        const int q_token = q_block_start + q_i;
        const int o_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
        if (q_token < q_len && has_output) {
            O[o_offset + out_d] = int4_attention_from_float<T>(running_acc[q_i] / running_denom[q_i]);
        }
    }
}

template <typename T, bool CAUSAL, int HEAD_DIM>
static void launch_int4_attention(
    const uint8_t *Q, const uint8_t *K, const uint8_t *V,
    const float *S_Q, const float *S_K, const float *S_V,
    T *O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads)
{
    constexpr int SCALE_DIM = HEAD_DIM / 64;
    constexpr int BLOCK_Q = 16;

    const int q_blocks = ta_cdiv(q_len, BLOCK_Q);
    const int num_blocks = bs * num_q_heads * q_blocks;
    auto kernel = int4_attention_kernel_mma_qk<T, CAUSAL, HEAD_DIM, SCALE_DIM, BLOCK_Q>;

    kernel<<<num_blocks, 128>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

template <typename T, bool CAUSAL>
static void dispatch_int4_attention(
    const uint8_t *Q,
    const uint8_t *K,
    const uint8_t *V,
    const float *S_Q,
    const float *S_K,
    const float *S_V,
    T *O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    if (head_dim == 64) {
        launch_int4_attention<T, CAUSAL, 64>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    } else if (head_dim == 128) {
        launch_int4_attention<T, CAUSAL, 128>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    } else {
        fprintf(stderr, "int4_attention: unsupported head_dim=%d\n", head_dim);
    }
}

template <typename T, bool CAUSAL>
static void int4_attention_typed(
    const void *Q_raw,
    const void *K_raw,
    const void *V_raw,
    const void *S_Q_raw,
    const void *S_K_raw,
    const void *S_V_raw,
    void *O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    auto Q = reinterpret_cast<const uint8_t *>(Q_raw);
    auto K = reinterpret_cast<const uint8_t *>(K_raw);
    auto V = reinterpret_cast<const uint8_t *>(V_raw);
    auto S_Q = reinterpret_cast<const float*>(S_Q_raw);
    auto S_K = reinterpret_cast<const float*>(S_K_raw);
    auto S_V = reinterpret_cast<const float*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    dispatch_int4_attention<T, CAUSAL>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
        kv_capacity, num_q_heads, num_kv_heads, head_dim);
}

void int4_attention_noncausal(
    const void *Q_raw,
    const void *K_raw,
    const void *V_raw,
    const void *S_Q_raw,
    const void *S_K_raw,
    const void *S_V_raw,
    void *O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16)
{
    if (is_bf16) {
        int4_attention_typed<__nv_bfloat16, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    } else {
        int4_attention_typed<half, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}

void int4_attention_causal(
    const void *Q_raw,
    const void *K_raw,
    const void *V_raw,
    const void *S_Q_raw,
    const void *S_K_raw,
    const void *S_V_raw,
    void *O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16)
{
    if (is_bf16) {
        int4_attention_typed<__nv_bfloat16, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    } else {
        int4_attention_typed<half, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}
