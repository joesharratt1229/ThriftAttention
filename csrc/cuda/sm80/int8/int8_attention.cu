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
    constexpr int KV_CHUNK = 128;

    const int tid = threadIdx.x;
    const int q_idx = blockIdx.x;
    const int q_token = q_idx % q_len;
    const int q_head = (q_idx / q_len) % num_q_heads;
    const int batch = q_idx / (q_len * num_q_heads);
    const int kv_head = q_head * num_kv_heads / num_q_heads;

    const int q_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
    const int o_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
    const int sq_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * SCALE_DIM;

    __shared__ float scores[KV_CHUNK];
    __shared__ float reduce_mem[KV_CHUNK];

    const int out_d = tid;
    const bool has_output = out_d < HEAD_DIM;
    const int v_group = out_d / 32;

    float running_max = -INFINITY;
    float running_denom = 0.0f;
    float running_acc = 0.0f;

    for (int kv_start = 0; kv_start < kv_len; kv_start += KV_CHUNK) {
        const int kv_token = kv_start + tid;
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

        scores[tid] = score;
        reduce_mem[tid] = score;
        __syncthreads();

        for (int stride = KV_CHUNK / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                reduce_mem[tid] = fmaxf(reduce_mem[tid], reduce_mem[tid + stride]);
            }
            __syncthreads();
        }

        const float chunk_max = reduce_mem[0];
        __syncthreads();

        float local_chunk_denom = 0.0f;
        if (isfinite(score) && isfinite(chunk_max)) {
            local_chunk_denom = expf(score - chunk_max);
        }

        reduce_mem[tid] = local_chunk_denom;
        __syncthreads();

        for (int stride = KV_CHUNK / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                reduce_mem[tid] += reduce_mem[tid + stride];
            }
            __syncthreads();
        }

        const float chunk_denom = reduce_mem[0];
        float chunk_acc = 0.0f;

        if (has_output && chunk_denom > 0.0f) {
            const int chunk_end = min(kv_start + KV_CHUNK, kv_len);
            for (int token = kv_start; token < chunk_end; token++) {
                const int score_idx = token - kv_start;
                const float token_score = scores[score_idx];
                if (!isfinite(token_score)) {
                    continue;
                }

                const int v_offset = ((batch * kv_capacity + token) * num_kv_heads + kv_head) * HEAD_DIM;
                const int sv_offset = ((batch * kv_capacity + token) * num_kv_heads + kv_head) * SCALE_DIM;
                const float weight = expf(token_score - chunk_max);
                const float v_real = float(V[v_offset + out_d]) * S_V[sv_offset + v_group];
                chunk_acc += weight * v_real;
            }
        }

        if (chunk_denom > 0.0f) {
            if (running_denom == 0.0f) {
                running_max = chunk_max;
                running_denom = chunk_denom;
                running_acc = chunk_acc;
            } else {
                const float new_max = fmaxf(running_max, chunk_max);
                const float old_scale = expf(running_max - new_max);
                const float chunk_scale = expf(chunk_max - new_max);

                running_acc = running_acc * old_scale + chunk_acc * chunk_scale;
                running_denom = running_denom * old_scale + chunk_denom * chunk_scale;
                running_max = new_max;
            }
        }

        __syncthreads();
    }

    if (has_output) {
        O[o_offset + out_d] = int8_attention_from_float<T>(running_acc / running_denom);
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