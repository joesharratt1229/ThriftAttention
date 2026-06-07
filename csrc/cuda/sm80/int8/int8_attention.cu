#include <cstdint>
#include <float.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "thriftattention/sm80/cuda_common.cuh"

template <typename T, bool CAUSAL, int BLOCK_Q, int BLOCK_KV, int HEAD_DIM, int INT8_HEAD_DIM, int SCALE_DIM, int NUM_WARPS, int WARP_Q>
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

    float score = 0.0f;

    int q_scale_offset = ((batch * q_len + q_token) * num_q_heads + q_head) * HEAD_DIM;
    int k_scale_offset = ();

    for (int d = 0; d < HEAD_DIM; d++) {
        int q_val = int(Q[q_offset + d]);
        int k_val = int(K[k_offset + d]);

        int group = d / 32;
        float scale = S_Q[q_scale_offset + group] * S_K[k_scale_offset + group];
        score += float(q_val * k_val) * scale;
    }

    score *= rsqrtf(float(HEAD_DIM));

    float max_score = -INFINITY;
    for (int d = 0; d < HEAD_DIM; d++) {

    }
    O
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
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int WARP_Q = 16;
    constexpr int NUM_WARPS = BLOCK_Q / WARP_Q;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;

    const int num_blocks = bs * num_q_heads * ta_cdiv(q_len, BLOCK_Q);

    constexpr int q_phase_smem = BLOCK_Q * INT8_HEAD_DIM * sizeof(int8_t) + BLOCK_Q * SCALE_DIM * sizeof(float);
    constexpr int v_phase_smem = BLOCK_KV * INT8_HEAD_DIM * sizeof(int8_t) + BLOCK_KV * SCALE_DIM * sizeof(float);
    constexpr int k_phase_smem = BLOCK_KV * INT8_HEAD_DIM * sizeof(int8_t) + BLOCK_KV * SCALE_DIM * sizeof(float);

    constexpr int smem_size = q_phase_smem + v_phase_smem;

    auto kernel = int8_attention_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM, INT8_HEAD_DIM, SCALE_DIM, NUM_WARPS, WARP_Q>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<num_blocks, TB_SIZE, smem_size>>>(
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