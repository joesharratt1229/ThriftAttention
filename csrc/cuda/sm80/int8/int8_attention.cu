#include <cstdint>
#include <float.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "thriftattention/sm80/cuda_common.cuh"

template <typename T>
__device__ inline T int8_attention_from_float(float value);

template <>
__device__ inline half int8_attention_from_float<half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 int8_attention_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <int HEAD_DIM, int SCALE_DIM>
__device__ float compute_int8_score(
    const int8_t* Q,
    const int8_t* K,
    const float* S_Q,
    const float* S_K,
    const int q_offset,
    const int sq_offset,
    const int batch,
    const int kv_capacity,
    const int kv_token,
    const int num_kv_heads,
    const int kv_head)
{
    const int k_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * HEAD_DIM;
    const int sk_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * SCALE_DIM;

    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
        const int group = d / 32;
        const int q_val = int(Q[q_offset + d]);
        const int k_val = int(K[k_offset + d]);
        const float scale = S_Q[sq_offset + group] * S_K[sk_offset + group];
        score += float(q_val * k_val) * scale;
    }

    return score * rsqrtf(float(HEAD_DIM));
}

template <typename T, bool CAUSAL, int HEAD_DIM, int INT8_HEAD_DIM, int SCALE_DIM>
__global__ void int8_attention_kernel(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    const float* S_Q,
    const float* S_K,
    const float* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads
) {
    const int q_idx = blockIdx.x;
    const int q_token = q_idx % q_len;
    const int q_head = (q_idx / q_len) % num_q_heads;
    const int batch = q_idx / (q_len * num_q_heads);
    const int kv_head = q_head * num_kv_heads / num_q_heads;

    const int q_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
    const int o_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
    const int sq_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * SCALE_DIM;

    __shared__ float score_mem[128];
    __shared__ float denom_mem[128];

    // Phase 1: find the largest QK score for numerical stability.
    float local_max = -INFINITY;

    for (int kv_token = threadIdx.x; kv_token < kv_len; kv_token += blockDim.x) {
        float score = -INFINITY;
        if (kv_token < kv_len) {
            score = compute_int8_score<HEAD_DIM, SCALE_DIM>(
                Q, K, S_Q, S_K, q_offset, sq_offset, batch, kv_capacity, kv_token,
                num_kv_heads, kv_head);

            if constexpr (CAUSAL) {
                if (kv_token > q_token) {
                    score = -INFINITY;
                }
            }
        } 
        local_max = fmaxf(local_max, score);
    }
 
    __syncthreads();
    score_mem[threadIdx.x] = local_max;
 
    for (int stride=blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            score_mem[threadIdx.x] = fmaxf(score_mem[threadIdx.x], score_mem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_score = score_mem[0];

    // Phase 2: compute the softmax denominator.
    float local_denom = 0.0f;
    for (int kv_token = threadIdx.x; kv_token < kv_len; kv_token += blockDim.x) {
        float score = compute_int8_score<HEAD_DIM, SCALE_DIM>(
            Q, K, S_Q, S_K, q_offset, sq_offset, batch, kv_capacity, kv_token,
            num_kv_heads, kv_head);

        if constexpr (CAUSAL) {
            if (kv_token > q_token) {
                score = -INFINITY;
            }
        }

        local_denom += expf(score - max_score);
    }

    denom_mem[threadIdx.x] = local_denom;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            denom_mem[threadIdx.x] += denom_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float denom = denom_mem[0];

    // Phase 3: use softmax probabilities to mix V into each output channel.
    int out_d = threadIdx.x;
    if (out_d < HEAD_DIM) {
        const int v_group = out_d / 32;
        float acc = 0.0f;

        for (int kv_token = 0; kv_token < kv_len; kv_token++) {
            const int v_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * HEAD_DIM;
            const int sv_offset = ((batch * kv_capacity + kv_token) * num_kv_heads + kv_head) * SCALE_DIM;

            float score = compute_int8_score<HEAD_DIM, SCALE_DIM>(
                Q, K, S_Q, S_K, q_offset, sq_offset, batch, kv_capacity, kv_token,
                num_kv_heads, kv_head);

            if constexpr (CAUSAL) {
                if (kv_token > q_token) {
                    score = -INFINITY;
                }
            }
            const float p = expf(score - max_score) / denom;
            const float v_real = float(V[v_offset + out_d]) * S_V[sv_offset + v_group];
            acc += p * v_real;
        }

        O[o_offset + out_d] = int8_attention_from_float<T>(acc);
    }
}

template <typename T, bool CAUSAL, int HEAD_DIM>
static void launch_int8_attention(
    const int8_t *Q, const int8_t *K, const int8_t *V,
    const float *S_Q, const float *S_K, const float *S_V,
    T *O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads)
{
    constexpr int INT8_HEAD_DIM = HEAD_DIM;
    constexpr int SCALE_DIM = HEAD_DIM / 32;

    const int num_blocks = bs * num_q_heads * q_len;

    auto kernel = int8_attention_kernel<T, CAUSAL, HEAD_DIM, INT8_HEAD_DIM, SCALE_DIM>;

    kernel<<<num_blocks, 128>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

template <typename T, bool CAUSAL>
static void dispatch_int8_attention(
    const int8_t *Q,
    const int8_t *K,
    const int8_t *V,
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
    if (head_dim == 64)
    {
        launch_int8_attention<T, CAUSAL, 64>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    }
    else if (head_dim == 128)
    {
        launch_int8_attention<T, CAUSAL, 128>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    }
    else
    {
        fprintf(stderr, "int8_attention: unsupported head_dim=%d\n", head_dim);
    }
}

template <typename T, bool CAUSAL>
static void int8_attention_typed(
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
    auto Q = reinterpret_cast<const int8_t *>(Q_raw);
    auto K = reinterpret_cast<const int8_t *>(K_raw);
    auto V = reinterpret_cast<const int8_t *>(V_raw);
    auto S_Q = reinterpret_cast<const float*>(S_Q_raw);
    auto S_K = reinterpret_cast<const float*>(S_K_raw);
    auto S_V = reinterpret_cast<const float*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    dispatch_int8_attention<T, CAUSAL>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
        kv_capacity, num_q_heads, num_kv_heads, head_dim);
}

void int8_attention_noncausal(
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
    if (is_bf16)
    {
        int8_attention_typed<__nv_bfloat16, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
    else
    {
        int8_attention_typed<half, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}

void int8_attention_causal(
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
    if (is_bf16)
    {
        int8_attention_typed<__nv_bfloat16, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
    else
    {
        int8_attention_typed<half, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}