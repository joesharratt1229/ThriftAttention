// SM120 NVFP4 attention baseline

#include <cstdint>
#include <float.h>
#include <type_traits>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "thriftattention/sm120/cuda_common.cuh"
#include "thriftattention/sm120/paged_bf16_prefix.h"

// QK^T -> causal mask -> online softmax -> FP4-pack P
// VARLEN: keys at or past kv_valid (the end of the sequence) are masked, and key_bias (if not null) adds one value
// per key to every query's logits before scaling, indexed by the key's position in the sequence.
template<bool CAUSAL, int BLOCK_KV, int HEAD_DIM, int HEAD_DIM_2,
         int SCALE_DIM, int WARP_Q, bool VARLEN = false>
__device__ __forceinline__
void fp4_qk_softmax_pack(
    float rowmax[WARP_Q / 16][2],
    float rowsum[WARP_Q / 16][2],
    float O_rmem[WARP_Q / 16][HEAD_DIM / 8][4],
    uint32_t Q_rmem[WARP_Q / 16][HEAD_DIM / 64][4],
    uint32_t sfQ_rmem[WARP_Q / 16][HEAD_DIM / 64],
    uint32_t S_fp4_rmem[WARP_Q / 16][BLOCK_KV / 64][4],
    uint32_t S_fp4_s_rmem[WARP_Q / 16][BLOCK_KV / 64],
    uint32_t K_ld_base,
    uint32_t K_sf_smem,
    int k_block_start,
    bool needs_causal_mask,
    int lane_id,
    int k_col_base,
    int q_row_upper_global,
    int q_row_lower_global,
    float softmax_scale,
    int kv_valid = 0,
    const float* key_bias = nullptr)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    uint32_t K_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t sfK_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K];
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    if constexpr (HEAD_DIM / MMA_K == 2) {
        // ldmatrix_x4: lanes 0-15 cover mma_id_d=0, lanes 16-31 cover mma_id_d=1
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            uint32_t addr = K_ld_base + mma_id_kv * MMA_N * HEAD_DIM_2;
            ta_ldmatrix_x4(K_rmem[mma_id_kv][0], addr);
        }
    } else if constexpr (HEAD_DIM / MMA_K == 4) {
        // second call covers mma_id_d={2,3} 
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            uint32_t addr = K_ld_base + mma_id_kv * MMA_N * HEAD_DIM_2;
            ta_ldmatrix_x4(K_rmem[mma_id_kv][0], addr);
            ta_ldmatrix_x4(K_rmem[mma_id_kv][2], addr ^ (2 * (MMA_K / 2)));
        }
    } else {
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
                uint32_t addr = K_ld_base;
                addr += mma_id_kv * MMA_N * HEAD_DIM_2;
                addr ^= mma_id_d * (MMA_K / 2);
                ta_ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
            }
    }

    const int sf_row_k = (lane_id / 4);
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            const int row = mma_id_kv * MMA_N + sf_row_k;
            const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
            asm volatile("ld.shared.u32 %0, [%1];"
                : "=r"(sfK_rmem[mma_id_kv][mma_id_d])
                : "r"(K_sf_smem + offset));
        }
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++)
                ta_mma_m16n8k64_nvfp4(
                    Q_rmem[mma_id_q][mma_id_d],
                    K_rmem[mma_id_kv][mma_id_d],
                    sfQ_rmem[mma_id_q][mma_id_d],
                    sfK_rmem[mma_id_kv][mma_id_d],
                    S_rmem[mma_id_q][mma_id_kv]);

    // ---- Varlen: add the per-key bias and mask the keys past the end of the sequence ----
    if constexpr (VARLEN) {
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            const int k_phys_0 = mma_id_kv * MMA_N + k_col_base;
            const int k_col_0 = k_block_start + ta_kv_logical_from_physical(k_phys_0);
            const int k_col_1 = k_block_start + ta_kv_logical_from_physical(k_phys_0 + 1);
            const bool valid_0 = k_col_0 < kv_valid;
            const bool valid_1 = k_col_1 < kv_valid;
            const float bias_0 = (key_bias != nullptr && valid_0) ? key_bias[k_col_0] : 0.0f;
            const float bias_1 = (key_bias != nullptr && valid_1) ? key_bias[k_col_1] : 0.0f;
            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                // regs[0], regs[2]: key k_col_0; regs[1], regs[3]: key k_col_1 (upper and lower query rows)
                float *regs = S_rmem[mma_id_q][mma_id_kv];
                regs[0] = valid_0 ? regs[0] + bias_0 : -INFINITY;
                regs[1] = valid_1 ? regs[1] + bias_1 : -INFINITY;
                regs[2] = valid_0 ? regs[2] + bias_0 : -INFINITY;
                regs[3] = valid_1 ? regs[3] + bias_1 : -INFINITY;
            }
        }
    }

    // ---- Apply causal mask on blocks overlapping the query tile ----
    if constexpr (CAUSAL) {
        if (needs_causal_mask) {
            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    const int k_phys_0 = mma_id_kv * MMA_N + k_col_base;
                    const int k_phys_1 = k_phys_0 + 1;
                    const int k_col_0 = k_block_start + ta_kv_logical_from_physical(k_phys_0);
                    const int k_col_1 = k_block_start + ta_kv_logical_from_physical(k_phys_1);

                    // regs[0]: (q_row_upper, k_col_0), regs[1]: (q_row_upper, k_col_1)
                    // regs[2]: (q_row_lower, k_col_0), regs[3]: (q_row_lower, k_col_1)
                    if (k_col_0 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][0] = -INFINITY;
                    if (k_col_1 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][1] = -INFINITY;
                    if (k_col_0 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][2] = -INFINITY;
                    if (k_col_1 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][3] = -INFINITY;
                }
            }
        }
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        // apply softmax scale
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int reg_id = 0; reg_id < 4; reg_id++)
                S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

        // rowmax through the logit maxima of the groups of 16 that share a P scale: scale 2g+h covers the
        // values that lanes 2h and 2h+1 of each quad hold for the 8-key blocks 4g..4g+3
        float group_max[BLOCK_KV / MMA_N / 4][2];
        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            group_max[g][0] = -FLT_MAX;
            group_max[g][1] = -FLT_MAX;
            for (int t = 4 * g; t < 4 * g + 4; t++) {
                const float *regs = S_rmem[mma_id_q][t];
                group_max[g][0] = max(group_max[g][0], max(regs[0], regs[1]));
                group_max[g][1] = max(group_max[g][1], max(regs[2], regs[3]));
            }
            group_max[g][0] = max(group_max[g][0], __shfl_xor_sync(0xFFFFFFFF, group_max[g][0], 1));
            group_max[g][1] = max(group_max[g][1], __shfl_xor_sync(0xFFFFFFFF, group_max[g][1], 1));
            this_rowmax[0] = max(this_rowmax[0], group_max[g][0]);
            this_rowmax[1] = max(this_rowmax[1], group_max[g][1]);
        }
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

        // new rowmax
        this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
        this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

        // rescale previous O for new rowmax
        float rescale[2];
        rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
        rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
            O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
            O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
            O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
        }

        // save new rowmax
        rowmax[mma_id_q][0] = this_rowmax[0];
        rowmax[mma_id_q][1] = this_rowmax[1];

        // softmax numerator and rowsumexp
        float this_rowsumexp[2] = {};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_q][mma_id_kv];
            regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
            regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
            regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
            regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

            this_rowsumexp[0] += regs[0] + regs[1];
            this_rowsumexp[1] += regs[2] + regs[3];
        }

        // butterfly reduction within 4 threads
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

        // accumulate to running rowsum
        rowsum[mma_id_q][0] = rowsum[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
        rowsum[mma_id_q][1] = rowsum[mma_id_q][1] * rescale[1] + this_rowsumexp[1];

        constexpr float FP4_RANGE = 448.0f * 6.0f;
        constexpr float FP4_MAX = 6.0f;
        float sf_P_upper[BLOCK_KV / MMA_N / 2];
        float sf_P_lower[BLOCK_KV / MMA_N / 2];

        // One FP8 scale per 16 probabilities, from their largest (one A-operand register of the P.V MMA)
        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            // the largest probability of the group, formed as the probabilities are
            const float pmax_upper = __expf(group_max[g][0] - rowmax[mma_id_q][0]);
            const float pmax_lower = __expf(group_max[g][1] - rowmax[mma_id_q][1]);

            // lanes 0, 1 hold the maximum of scale 2g and lanes 2, 3 that of 2g+1; every lane packs both
            const float other_upper = __shfl_xor_sync(0xFFFFFFFF, pmax_upper, 2);
            const float other_lower = __shfl_xor_sync(0xFFFFFFFF, pmax_lower, 2);
            const bool first = (lane_id % 4) < 2;
            const float upper_first = first ? pmax_upper : other_upper;
            const float upper_second = first ? other_upper : pmax_upper;
            const float lower_first = first ? pmax_lower : other_lower;
            const float lower_second = first ? other_lower : pmax_lower;
            sf_P_upper[2 * g] = upper_first > 0.0f ? (FP4_RANGE / FP4_MAX * upper_first) : 1.0f;
            sf_P_upper[2 * g + 1] = upper_second > 0.0f ? (FP4_RANGE / FP4_MAX * upper_second) : 1.0f;
            sf_P_lower[2 * g] = lower_first > 0.0f ? (FP4_RANGE / FP4_MAX * lower_first) : 1.0f;
            sf_P_lower[2 * g + 1] = lower_second > 0.0f ? (FP4_RANGE / FP4_MAX * lower_second) : 1.0f;

            const float inv_upper = pmax_upper > 0.0f ? (FP4_MAX / pmax_upper) : 0.0f;
            const float inv_lower = pmax_lower > 0.0f ? (FP4_MAX / pmax_lower) : 0.0f;
            for (int t = 4 * g; t < 4 * g + 4; t++) {
                S_rmem[mma_id_q][t][0] *= inv_upper;
                S_rmem[mma_id_q][t][1] *= inv_upper;
                S_rmem[mma_id_q][t][2] *= inv_lower;
                S_rmem[mma_id_q][t][3] *= inv_lower;
            }
        }

        for (int g = 0; g < BLOCK_KV / MMA_N / 4; g++) {
            int blk = 4 * g;
            float *r0 = S_rmem[mma_id_q][blk];
            float *r1 = S_rmem[mma_id_q][blk + 1];
            float *r2 = S_rmem[mma_id_q][blk + 2];
            float *r3 = S_rmem[mma_id_q][blk + 3];

            S_fp4_rmem[mma_id_q][0][2 * g] = ta_cvt_8xf32_to_e2m1_packed(
                r0[1], r0[0], r1[1], r1[0],
                r2[1], r2[0], r3[1], r3[0]);
            S_fp4_rmem[mma_id_q][0][2 * g + 1] = ta_cvt_8xf32_to_e2m1_packed(
                r0[3], r0[2], r1[3], r1[2],
                r2[3], r2[2], r3[3], r3[2]);
        }

        for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV / MMA_K; mma_sc_id++) {
            int base = mma_sc_id * 4;
            uint32_t sfP_upper_packed = ta_cvt_4xf32_to_e4m3_packed(
                sf_P_upper[base + 1], sf_P_upper[base + 0],
                sf_P_upper[base + 3], sf_P_upper[base + 2]);

            uint32_t sfP_lower_packed = ta_cvt_4xf32_to_e4m3_packed(
                sf_P_lower[base + 1], sf_P_lower[base + 0],
                sf_P_lower[base + 3], sf_P_lower[base + 2]);

            S_fp4_s_rmem[mma_id_q][mma_sc_id] =
                (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
        }
    }
}

// O += P @ V
template<int BLOCK_KV, int HEAD_DIM, int WARP_Q>
__device__ __forceinline__
void fp4_pv(
    float O_rmem[WARP_Q / 16][HEAD_DIM / 8][4],
    uint32_t S_fp4_rmem[WARP_Q / 16][BLOCK_KV / 64][4],
    uint32_t S_fp4_s_rmem[WARP_Q / 16][BLOCK_KV / 64],
    uint32_t V_smem,
    uint32_t V_sf_smem,
    int lane_id)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    uint32_t V_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
    uint32_t sfV_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N];

    // V layout in smem: [HEAD_DIM, BLOCK_KV/2], stride = BLOCK_KV/2 bytes.
    // ldmatrix_x4: lanes 0-15 address tile mma_id_d, lanes 16-31 address tile mma_id_d+1.
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d += 2) {
            const int n_idx = mma_id_d * MMA_N + (lane_id / 16) * MMA_N + (lane_id % 8);
            const int k_byte_offset = mma_id_kv * 32 + ((lane_id % 16) / 8) * 16;
            uint32_t addr = ta_swizzle<BLOCK_KV / 2>(
                V_smem + n_idx * (BLOCK_KV / 2) + k_byte_offset);

            ta_ldmatrix_x4(V_rmem[mma_id_kv][mma_id_d], addr);
        }
    }

    constexpr int V_SF_STRIDE = BLOCK_KV / 16;
    const int sf_col_v = lane_id / 4;
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int hd_col = mma_id_d * MMA_N + sf_col_v;
            const uint32_t offset = (hd_col * V_SF_STRIDE + mma_id_kv * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
            asm volatile("ld.shared.u32 %0, [%1];"
                : "=r"(sfV_rmem[mma_id_kv][mma_id_d])
                : "r"(V_sf_smem + offset));
        }

    // MMA O += P @ V
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
                ta_mma_m16n8k64_nvfp4(
                    S_fp4_rmem[mma_id_q][mma_id_kv],
                    V_rmem[mma_id_kv][mma_id_d],
                    S_fp4_s_rmem[mma_id_q][mma_id_kv],
                    sfV_rmem[mma_id_kv][mma_id_d],
                    O_rmem[mma_id_q][mma_id_d]);
}

template<typename T, bool CAUSAL, int BLOCK_Q, int BLOCK_KV, int HEAD_DIM,
         int HEAD_DIM_2, int SCALE_DIM,
         int NUM_WARPS, int WARP_Q>
__launch_bounds__(NUM_WARPS * TA_WARP_SIZE)
__global__
void fp4_attention_kernel(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / TA_WARP_SIZE;
    const int lane_id = tid % TA_WARP_SIZE;

    const int num_q_blocks = ta_cdiv(q_len, BLOCK_Q);
    const int q_bid = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int batch_id = q_bid / num_q_heads;
    const int q_head = q_bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;
    const int v_kv = ta_cdiv(kv_capacity, 128) * 128;

    Q += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM_2;
    K += kv_bid * kv_capacity * HEAD_DIM_2;
    V += kv_bid * HEAD_DIM * (v_kv / 2);

    S_Q += (q_bid * q_len + q_block_id * BLOCK_Q) * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM;
    S_V += kv_bid * v_kv * SCALE_DIM;
    O += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM;

    extern __shared__ uint8_t smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);
    const uint32_t V_smem = Q_sf_smem + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    const uint32_t V_sf_smem = V_smem + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4];
    uint32_t sfQ_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K];

    float rowmax[WARP_Q/MMA_M][2];
    float rowsum[WARP_Q/MMA_M][2] = {};
    float O_rmem[WARP_Q/MMA_M][HEAD_DIM/MMA_N][4] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q/MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    // ---- load Q data global -> shared (swizzled) -> registers ----
    ta_gmem_to_smem<BLOCK_Q, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(Q_smem, Q, tid, HEAD_DIM_2);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // Pre-compute swizzled base address for Q ldmatrix.
    // Row offsets that are multiples of 8 rows can be added (swizzle repeats every 8 rows).
    // Column tile offsets use XOR: swizzle(base + col_delta) == swizzle(base) ^ col_delta.
    uint32_t Q_ld_base;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = ta_swizzle<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            uint32_t addr = Q_ld_base;
            addr += mma_id_q * MMA_M * HEAD_DIM_2;  // row: MMA_M=16, 16%8==0
            addr ^= mma_id_d * (MMA_K / 2);          // col via XOR
            ta_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }

    // ---- load Q scales global -> shared -> registers ----
    ta_load_scales<BLOCK_Q, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(Q_sf_smem, S_Q, SCALE_DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + sf_row_q;
            const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
            asm volatile("ld.shared.u32 %0, [%1];"
                : "=r"(sfQ_rmem[mma_id_q][mma_id_d])
                : "r"(Q_sf_smem + offset));
        }
    }

    // ---- KV loop ----
    __syncthreads();

    const int total_kv_iters = ta_cdiv(kv_len, BLOCK_KV);
    const int max_kv_pos = q_block_id * BLOCK_Q + BLOCK_Q - 1;
    const int num_kv_iters = CAUSAL
        ? min(max_kv_pos / BLOCK_KV + 1, total_kv_iters)
        : total_kv_iters;

    // Precompute per-thread query row offsets within the block.
    const int q_block_start = q_block_id * BLOCK_Q;
    const int q_row_upper = warp_id * WARP_Q + (lane_id / 4);       // rows 0-7 of MMA tile
    const int q_row_lower = q_row_upper + 8;                         // rows 8-15 of MMA tile
    const int q_row_upper_global = q_block_start + q_row_upper;
    const int q_row_lower_global = q_block_start + q_row_lower;
    const int k_col_base = (lane_id % 4) * 2;                        // base key col within MMA_N tile

    const uint32_t K_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_sf_smem = K_smem + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    // Pre-compute base address for K ldmatrix.
    uint32_t K_ld_base;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base = ta_swizzle<HEAD_DIM_2>(K_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        uint32_t S_fp4_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K];

        const int k_block_start = kv_iter * BLOCK_KV;
        const bool needs_causal_mask = CAUSAL && ((k_block_start + BLOCK_KV - 1) > q_block_start);

        // Load K first; issue V in a second async group so QK can overlap
        // with the V transfer.
        ta_gmem_to_smem<BLOCK_KV, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_smem, K, tid, HEAD_DIM_2);
        ta_load_scales<BLOCK_KV, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(K_sf_smem, S_K, SCALE_DIM, tid);
        asm volatile("cp.async.commit_group;");
        // every warp has finished reading the tile this copy overwrites
        __syncthreads();
        ta_gmem_to_smem<HEAD_DIM, BLOCK_KV / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_smem, V, tid, v_kv / 2);
        ta_load_scales<HEAD_DIM, BLOCK_KV / 16, TB_SIZE, __nv_fp8_e4m3>(V_sf_smem, S_V, v_kv / 16, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        fp4_qk_softmax_pack<CAUSAL, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, WARP_Q>(
            rowmax, rowsum, O_rmem, Q_rmem, sfQ_rmem, S_fp4_rmem, S_fp4_s_rmem,
            K_ld_base, K_sf_smem, k_block_start, needs_causal_mask,
            lane_id, k_col_base, q_row_upper_global, q_row_lower_global, softmax_scale);

        // V layout in smem: [HEAD_DIM, BLOCK_KV/2], stride = BLOCK_KV/2 bytes.
        // ldmatrix_x4: lanes 0-15 address tile mma_id_d, lanes 16-31 address tile mma_id_d+1.
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        fp4_pv<BLOCK_KV, HEAD_DIM, WARP_Q>(
            O_rmem, S_fp4_rmem, S_fp4_s_rmem, V_smem, V_sf_smem, lane_id);

        K += BLOCK_KV * HEAD_DIM_2;
        S_K += BLOCK_KV * SCALE_DIM;

        V  += BLOCK_KV / 2;
        S_V += BLOCK_KV / 16;
    }

    constexpr float FP4_RANGE_INV = 1.0f / (448.0f * 6.0f);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];

            float norm0 = FP4_RANGE_INV / rowsum[mma_id_q][0];
            float norm1 = FP4_RANGE_INV / rowsum[mma_id_q][1];

            regs[0] *= norm0;
            regs[1] *= norm0;
            regs[2] *= norm1;
            regs[3] *= norm1;

            reinterpret_cast<typename Traits::vec2*>(O + (row + 0) * HEAD_DIM + col)[0] =
                Traits::pack2(regs[0], regs[1]);
            reinterpret_cast<typename Traits::vec2*>(O + (row + 8) * HEAD_DIM + col)[0] =
                Traits::pack2(regs[2], regs[3]);
        }
}

// Writes 16 output columns (two m16n8 accumulator tiles) of two rows, token and token + 8, as the NVFP4 input of
// the next Linear, bit for bit as vLLM's scaled_fp4_quant would write the 16-bit output: each value is rounded to T,
// the 16 values share the scale sf = e4m3(global_scale * amax / 6), stored in the swizzled 128x4 layout of the
// block-scaled GEMMs, and each value becomes e2m1(v * rcp(sf * rcp(global_scale))), the lower column in the low
// nibble. The four lanes of a quad hold the 16 columns (columns 2p, 2p + 1 of each tile in lane p); they share the
// amax by shuffles, regroup the bytes by shuffles, and each lane stores one 4-byte word.
template<typename T>
__device__ __forceinline__ void store_nvfp4_output_group(
    const float (&left)[4], const float (&right)[4], float norm_upper, float norm_lower, float global_scale,
    uint8_t* out, uint8_t* out_sf, int token, bool upper_valid, bool lower_valid, int64_t row_bytes,
    int byte_col, int sf_col, int num_k_tiles, int lane_id)
{
    using Traits = PrecisionTraits<T>;
    const float upper[4] = {
        Traits::to_float(Traits::from_float(left[0] * norm_upper)),
        Traits::to_float(Traits::from_float(left[1] * norm_upper)),
        Traits::to_float(Traits::from_float(right[0] * norm_upper)),
        Traits::to_float(Traits::from_float(right[1] * norm_upper))};
    const float lower[4] = {
        Traits::to_float(Traits::from_float(left[2] * norm_lower)),
        Traits::to_float(Traits::from_float(left[3] * norm_lower)),
        Traits::to_float(Traits::from_float(right[2] * norm_lower)),
        Traits::to_float(Traits::from_float(right[3] * norm_lower))};

    float amax_upper = fmaxf(fmaxf(fabsf(upper[0]), fabsf(upper[1])), fmaxf(fabsf(upper[2]), fabsf(upper[3])));
    float amax_lower = fmaxf(fmaxf(fabsf(lower[0]), fabsf(lower[1])), fmaxf(fabsf(lower[2]), fabsf(lower[3])));
    amax_upper = fmaxf(amax_upper, __shfl_xor_sync(0xffffffffu, amax_upper, 1));
    amax_upper = fmaxf(amax_upper, __shfl_xor_sync(0xffffffffu, amax_upper, 2));
    amax_lower = fmaxf(amax_lower, __shfl_xor_sync(0xffffffffu, amax_lower, 1));
    amax_lower = fmaxf(amax_lower, __shfl_xor_sync(0xffffffffu, amax_lower, 2));

    const __nv_fp8_e4m3 sf_upper(global_scale * (amax_upper * ta_rcp_approx_ftz(6.0f)));
    const __nv_fp8_e4m3 sf_lower(global_scale * (amax_lower * ta_rcp_approx_ftz(6.0f)));
    const float sf_upper_value = static_cast<float>(sf_upper);
    const float sf_lower_value = static_cast<float>(sf_lower);
    const float scale_upper = sf_upper_value != 0.0f
        ? ta_rcp_approx_ftz(sf_upper_value * ta_rcp_approx_ftz(global_scale)) : 0.0f;
    const float scale_lower = sf_lower_value != 0.0f
        ? ta_rcp_approx_ftz(sf_lower_value * ta_rcp_approx_ftz(global_scale)) : 0.0f;

    // bytes, low to high: upper columns 2p/2p+1 of the left tile, of the right tile, then the same for lower
    const uint32_t packed = ta_cvt_8xf32_to_e2m1_packed(
        upper[1] * scale_upper, upper[0] * scale_upper, upper[3] * scale_upper, upper[2] * scale_upper,
        lower[1] * scale_lower, lower[0] * scale_lower, lower[3] * scale_lower, lower[2] * scale_lower);

    // word w gathers byte w of the four lanes in lane order: the row's bytes 0-3 (left tile) or 4-7 (right tile)
    const int quad_lane = lane_id % 4;
    uint32_t words[4];
    for (int w = 0; w < 4; w++) {
        uint32_t part = ((packed >> (8 * w)) & 0xFFu) << (8 * quad_lane);
        part |= __shfl_xor_sync(0xffffffffu, part, 1);
        part |= __shfl_xor_sync(0xffffffffu, part, 2);
        words[w] = part;
    }
    const uint32_t word = quad_lane == 0 ? words[0] : quad_lane == 1 ? words[1] : quad_lane == 2 ? words[2] : words[3];
    const bool is_lower = quad_lane >= 2;
    const int my_token = token + (is_lower ? 8 : 0);
    if (is_lower ? lower_valid : upper_valid) {
        *reinterpret_cast<uint32_t*>(out + my_token * row_bytes + byte_col + 4 * (quad_lane % 2)) = word;
        if (quad_lane % 2 == 0) {
            const int64_t sf_offset = ((static_cast<int64_t>(my_token >> 7) * num_k_tiles + (sf_col >> 2)) << 9)
                                    | ((my_token & 31) << 4) | (((my_token >> 5) & 3) << 2) | (sf_col & 3);
            out_sf[sf_offset] = is_lower ? sf_lower.__x : sf_upper.__x;
        }
    }
}

// One tile of PREFIX_KV 16-bit prefix keys, from key k0 on, copied from the paged cache into shared memory at buf (K
// rows, then V rows, HEAD_DIM bfloat16 each, swizzled) as one cp.async group; keys past prefix_len are zero-filled
// (P is zero there, and V must be finite).
template<int HEAD_DIM, int TB_SIZE, int PREFIX_KV>
__device__ __forceinline__ void paged_prefix_load_tile(
    const PagedBf16PrefixArgs& prefix_args, const int* table, int kv_head, int prefix_len, int k0, uint32_t buf,
    int tid)
{
    constexpr int ROW_BYTES = HEAD_DIM * 2;
    constexpr int CHUNKS = ROW_BYTES / 16;
    const auto* key_cache = reinterpret_cast<const __nv_bfloat16*>(prefix_args.key_cache);
    const auto* value_cache = reinterpret_cast<const __nv_bfloat16*>(prefix_args.value_cache);
    for (int i = tid; i < PREFIX_KV * CHUNKS; i += TB_SIZE) {
        const int j = i / CHUNKS;
        const int chunk = i % CHUNKS;
        const int key = k0 + j;
        const bool inside = key < prefix_len;
        const int block = inside ? __ldg(table + key / prefix_args.block_size) : 0;
        const int64_t offset = static_cast<int64_t>(block) * prefix_args.block_stride
                             + static_cast<int64_t>(inside ? key % prefix_args.block_size : 0)
                               * prefix_args.token_stride
                             + static_cast<int64_t>(kv_head) * prefix_args.head_stride + chunk * 8;
        const uint32_t k_dst = ta_swizzle<ROW_BYTES>(buf + j * ROW_BYTES + chunk * 16);
        const uint32_t v_dst = ta_swizzle<ROW_BYTES>(buf + (PREFIX_KV + j) * ROW_BYTES + chunk * 16);
        const int bytes = inside ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(k_dst), "l"(key_cache + offset), "r"(bytes));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(v_dst), "l"(value_cache + offset), "r"(bytes));
    }
    asm volatile("cp.async.commit_group;");
}

// The bfloat16 prefix keys of one sequence for the WARP_Q = 16 query rows of a warp, after the FP4 keys:
// S = Q16 K16^T in bfloat16 MMAs, brought to the FP4 logits' frame and scaled, folded into the running rowmax, rowsum
// and O_rmem as the FP4 loop folds its blocks, with P packed in bfloat16 times FP4_RANGE so that O_rmem keeps the FP4
// loop's units. Tiles of PREFIX_KV keys alternate between two shared-memory buffers: buf0, which the caller filled
// with tile 0 before the FP4 loop (paged_prefix_load_tile), and buf1, the FP4 loop's memory, free once every warp has
// left that loop; each tile's copy runs while the previous tile is computed. Key blocks past prefix_len are skipped.
template<int HEAD_DIM, int TB_SIZE, int PREFIX_KV>
__device__ __forceinline__ void paged_prefix_attend(
    float (&rowmax)[1][2], float (&rowsum)[1][2], float (&O_rmem)[1][HEAD_DIM / 8][4],
    const PagedBf16PrefixArgs& prefix_args, const int* table, int prefix_len, int q_head, int kv_head, int num_q_heads,
    int out_base, int row_upper, int row_lower, bool valid_upper, bool valid_lower, uint32_t buf0, uint32_t buf1,
    int tid, int lane_id, float softmax_scale)
{
    using Traits = PrecisionTraits<__nv_bfloat16>;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int ROW_BYTES = HEAD_DIM * 2;
    constexpr int N_TILES = PREFIX_KV / MMA_N;
    constexpr float FP4_RANGE = 448.0f * 6.0f;

    // the warp's 16-bit Q as m16n8k16 A fragments (row lane / 4 and + 8, columns 2 * (lane % 4) + {0, 1} and + 8 of
    // each 16 columns), zero for rows that are not written; and q . key_mean for the two rows (0 without key_mean)
    uint32_t Q16[HEAD_DIM / MMA_K][4];
    float shift[2] = {0.0f, 0.0f};
    const auto* q_bf16 = reinterpret_cast<const __nv_bfloat16*>(prefix_args.q_bf16);
    const __nv_bfloat16* q_upper =
        q_bf16 + (static_cast<int64_t>(out_base + row_upper) * num_q_heads + q_head) * HEAD_DIM;
    const __nv_bfloat16* q_lower =
        q_bf16 + (static_cast<int64_t>(out_base + row_lower) * num_q_heads + q_head) * HEAD_DIM;
    const float* key_mean = prefix_args.key_mean == nullptr ? nullptr : prefix_args.key_mean + kv_head * HEAD_DIM;
    const int col_base = (lane_id % 4) * 2;
    for (int d = 0; d < HEAD_DIM / MMA_K; d++) {
        const int col0 = d * MMA_K + col_base;
        const int col1 = col0 + 8;
        Q16[d][0] = valid_upper ? *reinterpret_cast<const uint32_t*>(q_upper + col0) : 0u;
        Q16[d][1] = valid_lower ? *reinterpret_cast<const uint32_t*>(q_lower + col0) : 0u;
        Q16[d][2] = valid_upper ? *reinterpret_cast<const uint32_t*>(q_upper + col1) : 0u;
        Q16[d][3] = valid_lower ? *reinterpret_cast<const uint32_t*>(q_lower + col1) : 0u;
        const float2 u0 = Traits::to_float2(*reinterpret_cast<const Traits::vec2*>(&Q16[d][0]));
        const float2 l0 = Traits::to_float2(*reinterpret_cast<const Traits::vec2*>(&Q16[d][1]));
        const float2 u1 = Traits::to_float2(*reinterpret_cast<const Traits::vec2*>(&Q16[d][2]));
        const float2 l1 = Traits::to_float2(*reinterpret_cast<const Traits::vec2*>(&Q16[d][3]));
        if (key_mean != nullptr) {
            const float k00 = __ldg(key_mean + col0), k01 = __ldg(key_mean + col0 + 1);
            const float k10 = __ldg(key_mean + col1), k11 = __ldg(key_mean + col1 + 1);
            shift[0] += u0.x * k00 + u0.y * k01 + u1.x * k10 + u1.y * k11;
            shift[1] += l0.x * k00 + l0.y * k01 + l1.x * k10 + l1.y * k11;
        }
    }
    for (int r = 0; r < 2; r++) {
        shift[r] += __shfl_xor_sync(0xFFFFFFFF, shift[r], 1);
        shift[r] += __shfl_xor_sync(0xFFFFFFFF, shift[r], 2);
    }

    const uint32_t K_lane = ((lane_id % 8) * HEAD_DIM + (lane_id / 8) * 8) * 2;
    const uint32_t V_lane = (PREFIX_KV * HEAD_DIM + (lane_id % 16) * HEAD_DIM + (lane_id / 16) * 8) * 2;
    const int tiles = ta_cdiv(prefix_len, PREFIX_KV);
    __syncthreads();  // every warp has left the FP4 loop, whose memory buf1 is
    for (int t = 0; t < tiles; t++) {
        const int k0 = t * PREFIX_KV;
        const uint32_t buf = t % 2 == 0 ? buf0 : buf1;
        if (t + 1 < tiles) {  // the next tile into the other buffer, which tile t - 1 was read from before the barrier
            paged_prefix_load_tile<HEAD_DIM, TB_SIZE, PREFIX_KV>(
                prefix_args, table, kv_head, prefix_len, k0 + PREFIX_KV, t % 2 == 0 ? buf1 : buf0, tid);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const uint32_t K_ld_base = ta_swizzle<ROW_BYTES>(buf + K_lane);
        const uint32_t V_ld_base = ta_swizzle<ROW_BYTES>(buf + V_lane);

        // S = Q16 K16^T over the 8-key blocks that hold prefix keys
        float S[N_TILES][4] = {};
        for (int n = 0; n < N_TILES; n++) {
            if (k0 + n * MMA_N >= prefix_len) break;
            for (int d = 0; d < HEAD_DIM / MMA_K; d++) {
                uint32_t K_frag[2];
                uint32_t addr = K_ld_base + n * MMA_N * ROW_BYTES;
                addr ^= d * MMA_K * 2;
                ta_ldmatrix_x2(K_frag, addr);
                Traits::mma(Q16[d], K_frag, S[n]);
            }
        }

        // the FP4 logits' frame, scale, keys past the prefix masked; then the running softmax as in the FP4 loop
        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        for (int n = 0; n < N_TILES; n++) {
            const int key0 = k0 + n * MMA_N + col_base;
            const bool in0 = key0 < prefix_len;
            const bool in1 = key0 + 1 < prefix_len;
            S[n][0] = in0 ? (S[n][0] - shift[0]) * softmax_scale : -INFINITY;
            S[n][1] = in1 ? (S[n][1] - shift[0]) * softmax_scale : -INFINITY;
            S[n][2] = in0 ? (S[n][2] - shift[1]) * softmax_scale : -INFINITY;
            S[n][3] = in1 ? (S[n][3] - shift[1]) * softmax_scale : -INFINITY;
            this_rowmax[0] = max(this_rowmax[0], max(S[n][0], S[n][1]));
            this_rowmax[1] = max(this_rowmax[1], max(S[n][2], S[n][3]));
        }
        for (int r = 0; r < 2; r++) {
            this_rowmax[r] = max(this_rowmax[r], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[r], 1));
            this_rowmax[r] = max(this_rowmax[r], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[r], 2));
            this_rowmax[r] = max(this_rowmax[r], rowmax[0][r]);
        }
        const float rescale[2] = {__expf(rowmax[0][0] - this_rowmax[0]), __expf(rowmax[0][1] - this_rowmax[1])};
        for (int d = 0; d < HEAD_DIM / MMA_N; d++) {
            O_rmem[0][d][0] *= rescale[0];
            O_rmem[0][d][1] *= rescale[0];
            O_rmem[0][d][2] *= rescale[1];
            O_rmem[0][d][3] *= rescale[1];
        }
        rowmax[0][0] = this_rowmax[0];
        rowmax[0][1] = this_rowmax[1];

        float this_rowsum[2] = {};
        uint32_t P[N_TILES / 2][4];
        for (int c = 0; c < N_TILES / 2; c++) {
            float* r0 = S[2 * c];
            float* r1 = S[2 * c + 1];
            for (int i = 0; i < 4; i++) {
                r0[i] = __expf(r0[i] - rowmax[0][i / 2]);
                r1[i] = __expf(r1[i] - rowmax[0][i / 2]);
            }
            this_rowsum[0] += r0[0] + r0[1] + r1[0] + r1[1];
            this_rowsum[1] += r0[2] + r0[3] + r1[2] + r1[3];
            Traits::vec2* p = reinterpret_cast<Traits::vec2*>(P[c]);
            p[0] = Traits::pack2(r0[0] * FP4_RANGE, r0[1] * FP4_RANGE);
            p[1] = Traits::pack2(r0[2] * FP4_RANGE, r0[3] * FP4_RANGE);
            p[2] = Traits::pack2(r1[0] * FP4_RANGE, r1[1] * FP4_RANGE);
            p[3] = Traits::pack2(r1[2] * FP4_RANGE, r1[3] * FP4_RANGE);
        }
        for (int r = 0; r < 2; r++) {
            this_rowsum[r] += __shfl_xor_sync(0xFFFFFFFF, this_rowsum[r], 1);
            this_rowsum[r] += __shfl_xor_sync(0xFFFFFFFF, this_rowsum[r], 2);
            rowsum[0][r] = rowsum[0][r] * rescale[r] + this_rowsum[r];
        }

        // O += P V16 over the 16-key blocks that hold prefix keys
        for (int c = 0; c < N_TILES / 2; c++) {
            if (k0 + c * MMA_K >= prefix_len) break;
            for (int d = 0; d < HEAD_DIM / MMA_N; d++) {
                uint32_t V_frag[2];
                uint32_t addr = V_ld_base + c * MMA_K * ROW_BYTES;
                addr ^= d * MMA_N * 2;
                ta_ldmatrix_x2_trans(V_frag, addr);
                Traits::mma(P[c], V_frag, O_rmem[0][d]);
            }
        }
        __syncthreads();  // every warp is done with this buffer before the next iteration's copy overwrites it
    }
}

// Variable-length self-attention over sequences packed along the token axis, one CTA per (sequence, query head,
// query block). Every sequence starts at a multiple of 64 tokens in the FP4 inputs (cu_seqlens_padded):
//   Q, S_Q: [num_q_heads, q_tokens, HEAD_DIM / 2], [num_q_heads, q_tokens, HEAD_DIM / 16]
//   K, S_K: [num_kv_heads, k_tokens, ...], tokens permuted within groups of 32 as for the other kernels
//   V, S_V: transposed, [num_kv_heads, HEAD_DIM, v_tokens / 2], [num_kv_heads, HEAD_DIM, v_tokens / 16]
//   key_bias: [num_q_heads, tokens] floats added to the logits of each key (or null)
// O is [total_tokens, num_q_heads, HEAD_DIM] at the unpadded offsets cu_seqlens; rows past a sequence are not written.
// With FP4_OUT, O is written as the NVFP4 input of the next Linear instead (store_nvfp4_output_group).
// q_start (or null): per sequence, the first query row; the rows before it (keys that a cache holds, whose queries
// were computed in an earlier call) are not written, and the grid starts at the query block that holds q_start.
// Every row's result depends only on its own query and the keys, so the rows that are written are those of a call
// without it.
// out_start (or null): per sequence, the output row of query row q_start (default: cu_seqlens + q_start).
// lse_out (or null): [num_q_heads, out_tokens] floats at the rows' output indices, the natural log-sum-exp of each
// written row's scaled logits as the kernel computes them, key_bias included (rowmax + log(rowsum)).
// PAGED_PREFIX (bfloat16 only): the 16-bit prefix of PagedBf16PrefixArgs after the FP4 keys (paged_prefix_attend).
template<typename T, bool CAUSAL, int BLOCK_Q, int BLOCK_KV, int HEAD_DIM,
         int HEAD_DIM_2, int SCALE_DIM,
         int NUM_WARPS, int WARP_Q, bool FP4_OUT = false, bool PAGED_PREFIX = false>
// Keeps the CTAs per SM that the 16-bit epilogue reaches (4 at head_dim 64, 3 at 128), which the FP4 epilogue
// would otherwise lower.
__launch_bounds__(NUM_WARPS * TA_WARP_SIZE, HEAD_DIM == 64 ? 4 : 3)
__global__
void fp4_attention_varlen_kernel(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    const float* key_bias,
    T* O,
    const int* cu_seqlens,
    const int* cu_seqlens_padded,
    int q_tokens,
    int k_tokens,
    int v_tokens,
    int max_q_blocks,
    int num_q_heads,
    int num_kv_heads,
    uint8_t* O_fp4 = nullptr,
    uint8_t* O_sf = nullptr,
    const float* out_global_scale = nullptr,
    const int* q_start = nullptr,
    const int* out_start = nullptr,
    float* lse_out = nullptr,
    int out_tokens = 0,
    PagedBf16PrefixArgs prefix_args = {})
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / TA_WARP_SIZE;
    const int lane_id = tid % TA_WARP_SIZE;

    const int q_head = (bid / max_q_blocks) % num_q_heads;
    const int seq = bid / (max_q_blocks * num_q_heads);
    const int seq_start = cu_seqlens[seq];
    const int seq_len = cu_seqlens[seq + 1] - seq_start;
    const int first_row = q_start == nullptr ? 0 : q_start[seq];
    // the grid covers the query blocks from the one that holds row q_start on
    const int q_block_id = bid % max_q_blocks + first_row / BLOCK_Q;
    if (q_block_id * BLOCK_Q >= seq_len) return;
    // query row r of this sequence goes to output row out_base + r
    const int out_base = (out_start == nullptr ? seq_start + first_row : out_start[seq]) - first_row;
    const int padded_start = cu_seqlens_padded[seq];
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    const int64_t q_row = static_cast<int64_t>(q_head) * q_tokens + padded_start + q_block_id * BLOCK_Q;
    const int64_t k_row = static_cast<int64_t>(kv_head) * k_tokens + padded_start;
    Q += q_row * HEAD_DIM_2;
    S_Q += q_row * SCALE_DIM;
    K += k_row * HEAD_DIM_2;
    S_K += k_row * SCALE_DIM;
    V += static_cast<int64_t>(kv_head) * HEAD_DIM * (v_tokens / 2) + padded_start / 2;
    S_V += static_cast<int64_t>(kv_head) * HEAD_DIM * (v_tokens / 16) + padded_start / 16;
    const float* bias = key_bias == nullptr
        ? nullptr : key_bias + static_cast<int64_t>(q_head) * q_tokens + padded_start;

    extern __shared__ uint8_t smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);
    const uint32_t V_smem = Q_sf_smem + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    const uint32_t V_sf_smem = V_smem + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4];
    uint32_t sfQ_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K];

    float rowmax[WARP_Q/MMA_M][2];
    float rowsum[WARP_Q/MMA_M][2] = {};
    float O_rmem[WARP_Q/MMA_M][HEAD_DIM/MMA_N][4] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q/MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    // ---- load Q data global -> shared (swizzled) -> registers ----
    ta_gmem_to_smem<BLOCK_Q, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(Q_smem, Q, tid, HEAD_DIM_2);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    uint32_t Q_ld_base;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = ta_swizzle<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            uint32_t addr = Q_ld_base;
            addr += mma_id_q * MMA_M * HEAD_DIM_2;
            addr ^= mma_id_d * (MMA_K / 2);
            ta_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }

    // ---- load Q scales global -> shared -> registers ----
    ta_load_scales<BLOCK_Q, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(Q_sf_smem, S_Q, SCALE_DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + sf_row_q;
            const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
            asm volatile("ld.shared.u32 %0, [%1];"
                : "=r"(sfQ_rmem[mma_id_q][mma_id_d])
                : "r"(Q_sf_smem + offset));
        }
    }

    // ---- the bfloat16 prefix's first tile (paged_prefix_attend), issued before the FP4 loop's first K/V copy ----
    // buf0 lies past the FP4 loop's shared memory, which paged_prefix_attend reuses as buf1 afterwards
    constexpr int PREFIX_KV = 32;
    int prefix_len = 0;
    const int* prefix_table = nullptr;
    uint32_t prefix_buf0 = 0;
    if constexpr (PAGED_PREFIX) {
        static_assert(WARP_Q == 16, "paged_prefix_attend holds one m16 tile per warp");
        prefix_len = prefix_args.prefix_len[seq];
        prefix_table = prefix_args.block_table + static_cast<int64_t>(seq) * prefix_args.block_table_stride;
        prefix_buf0 = __cvta_generic_to_shared(smem) + 2 * PREFIX_KV * HEAD_DIM * 2;
        if (prefix_len > 0)
            paged_prefix_load_tile<HEAD_DIM, TB_SIZE, PREFIX_KV>(
                prefix_args, prefix_table, kv_head, prefix_len, 0, prefix_buf0, tid);
    }

    // ---- KV loop over the keys of this sequence ----
    __syncthreads();

    const int total_kv_iters = ta_cdiv(seq_len, BLOCK_KV);
    const int max_kv_pos = q_block_id * BLOCK_Q + BLOCK_Q - 1;
    const int num_kv_iters = CAUSAL
        ? min(max_kv_pos / BLOCK_KV + 1, total_kv_iters)
        : total_kv_iters;

    const int q_block_start = q_block_id * BLOCK_Q;
    const int q_row_upper = warp_id * WARP_Q + (lane_id / 4);
    const int q_row_lower = q_row_upper + 8;
    const int q_row_upper_global = q_block_start + q_row_upper;
    const int q_row_lower_global = q_block_start + q_row_lower;
    const int k_col_base = (lane_id % 4) * 2;

    const uint32_t K_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_sf_smem = K_smem + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t K_ld_base;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base = ta_swizzle<HEAD_DIM_2>(K_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        uint32_t S_fp4_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K];

        const int k_block_start = kv_iter * BLOCK_KV;
        const bool needs_causal_mask = CAUSAL && ((k_block_start + BLOCK_KV - 1) > q_block_start);

        ta_gmem_to_smem<BLOCK_KV, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_smem, K, tid, HEAD_DIM_2);
        ta_load_scales<BLOCK_KV, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(K_sf_smem, S_K, SCALE_DIM, tid);
        asm volatile("cp.async.commit_group;");
        // every warp has finished reading the tile this copy overwrites
        __syncthreads();
        ta_gmem_to_smem<HEAD_DIM, BLOCK_KV / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_smem, V, tid, v_tokens / 2);
        ta_load_scales<HEAD_DIM, BLOCK_KV / 16, TB_SIZE, __nv_fp8_e4m3>(V_sf_smem, S_V, v_tokens / 16, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        fp4_qk_softmax_pack<CAUSAL, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, WARP_Q, true>(
            rowmax, rowsum, O_rmem, Q_rmem, sfQ_rmem, S_fp4_rmem, S_fp4_s_rmem,
            K_ld_base, K_sf_smem, k_block_start, needs_causal_mask,
            lane_id, k_col_base, q_row_upper_global, q_row_lower_global, softmax_scale,
            seq_len, bias);

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        fp4_pv<BLOCK_KV, HEAD_DIM, WARP_Q>(
            O_rmem, S_fp4_rmem, S_fp4_s_rmem, V_smem, V_sf_smem, lane_id);

        K += BLOCK_KV * HEAD_DIM_2;
        S_K += BLOCK_KV * SCALE_DIM;

        V  += BLOCK_KV / 2;
        S_V += BLOCK_KV / 16;
    }

    if constexpr (PAGED_PREFIX) {
        if (prefix_len > 0)  // the same for every thread of the CTA, so its barriers are uniform
            paged_prefix_attend<HEAD_DIM, TB_SIZE, PREFIX_KV>(
                rowmax, rowsum, O_rmem, prefix_args, prefix_table, prefix_len, q_head, kv_head, num_q_heads, out_base,
                q_row_upper_global, q_row_lower_global,
                q_row_upper_global >= first_row && q_row_upper_global < seq_len,
                q_row_lower_global >= first_row && q_row_lower_global < seq_len,
                prefix_buf0, __cvta_generic_to_shared(smem), tid, lane_id, softmax_scale);
    }

    constexpr float FP4_RANGE_INV = 1.0f / (448.0f * 6.0f);

    if (lse_out != nullptr && lane_id % 4 == 0) {  // the four lanes of a quad hold the same rowmax and rowsum
        float* lse_head = lse_out + static_cast<int64_t>(q_head) * out_tokens + out_base;
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            const int row = q_block_start + warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            if (row >= first_row && row < seq_len)
                lse_head[row] = rowmax[mma_id_q][0] + logf(rowsum[mma_id_q][0]);
            if (row + 8 >= first_row && row + 8 < seq_len)
                lse_head[row + 8] = rowmax[mma_id_q][1] + logf(rowsum[mma_id_q][1]);
        }
    }

    if constexpr (FP4_OUT) {
        const float global_scale = __ldg(out_global_scale);
        const int64_t row_bytes = static_cast<int64_t>(num_q_heads) * HEAD_DIM_2;
        const int num_k_tiles = ta_cdiv(num_q_heads * HEAD_DIM, 64);
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            const int row = q_block_start + warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const float norm0 = FP4_RANGE_INV / rowsum[mma_id_q][0];
            const float norm1 = FP4_RANGE_INV / rowsum[mma_id_q][1];
            for (int group = 0; group < HEAD_DIM / 16; group++)
                store_nvfp4_output_group<T>(
                    O_rmem[mma_id_q][2 * group], O_rmem[mma_id_q][2 * group + 1], norm0, norm1, global_scale,
                    O_fp4, O_sf, out_base + row, row >= first_row && row < seq_len,
                    row + 8 >= first_row && row + 8 < seq_len, row_bytes,
                    q_head * HEAD_DIM_2 + 8 * group, q_head * (HEAD_DIM / 16) + group, num_k_tiles, lane_id);
        }
        return;
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int row = q_block_start + warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];
            const float norm0 = FP4_RANGE_INV / rowsum[mma_id_q][0];
            const float norm1 = FP4_RANGE_INV / rowsum[mma_id_q][1];

            if (row >= first_row && row < seq_len) {
                T* out = O + (static_cast<int64_t>(out_base + row) * num_q_heads + q_head) * HEAD_DIM + col;
                reinterpret_cast<typename Traits::vec2*>(out)[0] = Traits::pack2(regs[0] * norm0, regs[1] * norm0);
            }
            if (row + 8 >= first_row && row + 8 < seq_len) {
                T* out = O + (static_cast<int64_t>(out_base + row + 8) * num_q_heads + q_head) * HEAD_DIM + col;
                reinterpret_cast<typename Traits::vec2*>(out)[0] = Traits::pack2(regs[2] * norm1, regs[3] * norm1);
            }
        }
}

// HEAD_DIM==256: double-buffered KV loop
template<typename T, bool CAUSAL, int BLOCK_Q, int BLOCK_KV, int HEAD_DIM,
         int HEAD_DIM_2, int SCALE_DIM,
         int NUM_WARPS, int WARP_Q>
__launch_bounds__(NUM_WARPS * TA_WARP_SIZE)
__global__
void fp4_attention_hd256_kernel(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    using Traits = PrecisionTraits<T>;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;
    constexpr int MMA_M = 16;
    constexpr int MMA_K = 64;
    constexpr int MMA_N = 8;

    const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / TA_WARP_SIZE;
    const int lane_id = tid % TA_WARP_SIZE;

    const int num_q_blocks = ta_cdiv(q_len, BLOCK_Q);
    const int q_bid = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int batch_id = q_bid / num_q_heads;
    const int q_head = q_bid - batch_id * num_q_heads;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_bid = batch_id * num_kv_heads + kv_head;
    const int v_kv = ta_cdiv(kv_capacity, 128) * 128;

    Q += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM_2;
    K += kv_bid * kv_capacity * HEAD_DIM_2;
    V += kv_bid * HEAD_DIM * (v_kv / 2);

    S_Q += (q_bid * q_len + q_block_id * BLOCK_Q) * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM;
    S_V += kv_bid * v_kv * SCALE_DIM;
    O += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM;

    extern __shared__ uint8_t smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    uint32_t Q_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4];
    uint32_t sfQ_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K];

    float rowmax[WARP_Q/MMA_M][2];
    float rowsum[WARP_Q/MMA_M][2] = {};
    float O_rmem[WARP_Q/MMA_M][HEAD_DIM/MMA_N][4] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q/MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    // ---- load Q data global -> shared (swizzled) -> registers ----
    ta_gmem_to_smem<BLOCK_Q, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(Q_smem, Q, tid, HEAD_DIM_2);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // Pre-compute swizzled base address for Q ldmatrix.
    // Row offsets that are multiples of 8 rows can be added (swizzle repeats every 8 rows).
    // Column tile offsets use XOR: swizzle(base + col_delta) == swizzle(base) ^ col_delta.
    uint32_t Q_ld_base;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 16;
        Q_ld_base = ta_swizzle<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);
    }

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            uint32_t addr = Q_ld_base;
            addr += mma_id_q * MMA_M * HEAD_DIM_2;  // row: MMA_M=16, 16%8==0
            addr ^= mma_id_d * (MMA_K / 2);          // col via XOR
            ta_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }

    // ---- load Q scales global -> shared -> registers ----
    ta_load_scales<BLOCK_Q, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(Q_sf_smem, S_Q, SCALE_DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    int sf_row_q = 0;
    if (lane_id % 4 == 0) sf_row_q = (lane_id / 4);
    else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + sf_row_q;
            const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
            asm volatile("ld.shared.u32 %0, [%1];"
                : "=r"(sfQ_rmem[mma_id_q][mma_id_d])
                : "r"(Q_sf_smem + offset));
        }
    }

    // ---- KV loop ----
    __syncthreads();

    const int total_kv_iters = ta_cdiv(kv_len, BLOCK_KV);
    const int max_kv_pos = q_block_id * BLOCK_Q + BLOCK_Q - 1;
    const int num_kv_iters = CAUSAL
        ? min(max_kv_pos / BLOCK_KV + 1, total_kv_iters)
        : total_kv_iters;

    // Precompute per-thread query row offsets within the block.
    const int q_block_start = q_block_id * BLOCK_Q;
    const int q_row_upper = warp_id * WARP_Q + (lane_id / 4);       // rows 0-7 of MMA tile
    const int q_row_lower = q_row_upper + 8;                         // rows 8-15 of MMA tile
    const int q_row_upper_global = q_block_start + q_row_upper;
    const int q_row_lower_global = q_block_start + q_row_lower;
    const int k_col_base = (lane_id % 4) * 2;                        // base key col within MMA_N tile

    // ---- Double-buffered KV loop: prefetch next tile's K+V during compute ----
    constexpr int K_BUF_BYTES = BLOCK_KV * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_KV * SCALE_DIM  * (int)sizeof(__nv_fp8_e4m3);
    constexpr int V_BUF_BYTES = HEAD_DIM * (BLOCK_KV / 2)  * (int)sizeof(__nv_fp4x2_e2m1)
                               + HEAD_DIM * (BLOCK_KV / 16) * (int)sizeof(__nv_fp8_e4m3);
    constexpr int KV_BUF_BYTES = K_BUF_BYTES + V_BUF_BYTES;

    const uint32_t smem_base = __cvta_generic_to_shared(smem);

    // Pre-compute base address for K ldmatrix (buffer 0).
    const int row_off = lane_id % 8;
    const int col_off = (lane_id / 8) * 16;
    const uint32_t K_ld_base_0 = ta_swizzle<HEAD_DIM_2>(
        smem_base + row_off * HEAD_DIM_2 + col_off);

    uint32_t cur_buf_off = 0;  // toggles between 0 and KV_BUF_BYTES

    auto start_kv_load = [&](uint32_t buf_off, int iter) {
        const uint32_t K_buf    = smem_base + buf_off;
        const uint32_t K_sf_buf = K_buf    + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);
        const uint32_t V_buf    = K_sf_buf + BLOCK_KV * SCALE_DIM   * sizeof(__nv_fp8_e4m3);
        const uint32_t V_sf_buf = V_buf    + HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1);
        const auto* Kp  = K   + static_cast<int64_t>(iter) * BLOCK_KV * HEAD_DIM_2;
        const auto* SKp = S_K + static_cast<int64_t>(iter) * BLOCK_KV * SCALE_DIM;
        const auto* Vp  = V   + static_cast<int64_t>(iter) * BLOCK_KV / 2;
        const auto* SVp = S_V + static_cast<int64_t>(iter) * BLOCK_KV / 16;
        ta_gmem_to_smem<BLOCK_KV, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_buf,    Kp,  tid, HEAD_DIM_2);
        ta_load_scales<BLOCK_KV, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>    (K_sf_buf, SKp, SCALE_DIM, tid);
        ta_gmem_to_smem<HEAD_DIM, BLOCK_KV / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_buf,  Vp,  tid, v_kv / 2);
        ta_load_scales<HEAD_DIM, BLOCK_KV / 16, TB_SIZE, __nv_fp8_e4m3>  (V_sf_buf, SVp, v_kv / 16, tid);
        asm volatile("cp.async.commit_group;");
    };

    if (num_kv_iters > 0)
        start_kv_load(0, 0);

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        uint32_t S_fp4_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K];

        const int k_block_start = kv_iter * BLOCK_KV;
        const bool needs_causal_mask = CAUSAL && ((k_block_start + BLOCK_KV - 1) > q_block_start);

        // Prefetch next tile, then wait for the tile issued last iteration.
        const uint32_t nxt_buf_off = KV_BUF_BYTES - cur_buf_off;
        // every warp has finished reading the tile this copy overwrites
        __syncthreads();
        if (kv_iter + 1 < num_kv_iters) {
            start_kv_load(nxt_buf_off, kv_iter + 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_all;");
        }
        __syncthreads();

        const uint32_t cur_K_ld = K_ld_base_0 + cur_buf_off;
        const uint32_t cur_K_sf = smem_base + cur_buf_off
                                + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);
        const uint32_t cur_V    = cur_K_sf + BLOCK_KV * SCALE_DIM * sizeof(__nv_fp8_e4m3);
        const uint32_t cur_V_sf = cur_V    + HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1);

        fp4_qk_softmax_pack<CAUSAL, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM, WARP_Q>(
            rowmax, rowsum, O_rmem, Q_rmem, sfQ_rmem, S_fp4_rmem, S_fp4_s_rmem,
            cur_K_ld, cur_K_sf, k_block_start, needs_causal_mask,
            lane_id, k_col_base, q_row_upper_global, q_row_lower_global, softmax_scale);

        fp4_pv<BLOCK_KV, HEAD_DIM, WARP_Q>(
            O_rmem, S_fp4_rmem, S_fp4_s_rmem, cur_V, cur_V_sf, lane_id);

        cur_buf_off = KV_BUF_BYTES - cur_buf_off;
    }

    constexpr float FP4_RANGE_INV = 1.0f / (448.0f * 6.0f);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];

            float norm0 = FP4_RANGE_INV / rowsum[mma_id_q][0];
            float norm1 = FP4_RANGE_INV / rowsum[mma_id_q][1];

            regs[0] *= norm0;
            regs[1] *= norm0;
            regs[2] *= norm1;
            regs[3] *= norm1;

            reinterpret_cast<typename Traits::vec2*>(O + (row + 0) * HEAD_DIM + col)[0] =
                Traits::pack2(regs[0], regs[1]);
            reinterpret_cast<typename Traits::vec2*>(O + (row + 8) * HEAD_DIM + col)[0] =
                Traits::pack2(regs[2], regs[3]);
        }
}

template<typename T, bool CAUSAL, int HEAD_DIM>
static void launch_fp4_attention(
    const __nv_fp4x2_e2m1* Q, const __nv_fp4x2_e2m1* K, const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q, const __nv_fp8_e4m3* S_K, const __nv_fp8_e4m3* S_V,
    T* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads) {

    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int WARP_Q = 16;
    constexpr int NUM_WARPS = BLOCK_Q / WARP_Q;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;

    const int num_blocks = bs * ta_cdiv(q_len, BLOCK_Q);

    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int v_phase_smem = HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1)
                                + HEAD_DIM * (BLOCK_KV / 16) * sizeof(__nv_fp8_e4m3);
    constexpr int k_phase_smem = BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                                + BLOCK_KV * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int smem_size = q_phase_smem + v_phase_smem > k_phase_smem
                            ? q_phase_smem + v_phase_smem : k_phase_smem;

    auto kernel = fp4_attention_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM,
                                       HEAD_DIM_2, SCALE_DIM,
                                       NUM_WARPS, WARP_Q>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<num_blocks, TB_SIZE, smem_size>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

// HEAD_DIM==256 launcher: BLOCK_Q/KV=128 and double-buffered KV smem sizing.
template<typename T, bool CAUSAL>
static void launch_fp4_attention_hd256(
    const __nv_fp4x2_e2m1* Q, const __nv_fp4x2_e2m1* K, const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q, const __nv_fp8_e4m3* S_K, const __nv_fp8_e4m3* S_V,
    T* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads) {

    constexpr int HEAD_DIM = 256;
    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int BLOCK_Q = 128;
    constexpr int BLOCK_KV = 128;
    constexpr int WARP_Q = 16;
    constexpr int NUM_WARPS = BLOCK_Q / WARP_Q;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;

    const int num_blocks = bs * ta_cdiv(q_len, BLOCK_Q);

    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int v_phase_smem = HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1)
                                + HEAD_DIM * (BLOCK_KV / 16) * sizeof(__nv_fp8_e4m3);
    constexpr int k_phase_smem = BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                                + BLOCK_KV * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int kv_double_buf_smem = 2 * (k_phase_smem + v_phase_smem);
    constexpr int smem_size = q_phase_smem > kv_double_buf_smem
                            ? q_phase_smem : kv_double_buf_smem;

    auto kernel = fp4_attention_hd256_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM,
                                             HEAD_DIM_2, SCALE_DIM,
                                             NUM_WARPS, WARP_Q>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<num_blocks, TB_SIZE, smem_size>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

template<typename T, bool CAUSAL, int HEAD_DIM>
static void launch_fp4_attention_varlen(
    const __nv_fp4x2_e2m1* Q, const __nv_fp4x2_e2m1* K, const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q, const __nv_fp8_e4m3* S_K, const __nv_fp8_e4m3* S_V,
    const float* key_bias, T* O, const int* cu_seqlens, const int* cu_seqlens_padded,
    int num_seqs, int max_seqlen, int q_tokens, int k_tokens, int v_tokens,
    int num_q_heads, int num_kv_heads, uint8_t* O_fp4, uint8_t* O_sf, const float* out_global_scale,
    const int* q_start, const int* out_start, float* lse_out, int out_tokens, cudaStream_t stream,
    const PagedBf16PrefixArgs& prefix_args = {}) {

    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int WARP_Q = 16;
    constexpr int NUM_WARPS = BLOCK_Q / WARP_Q;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;

    // with q_start, max_seqlen is the most rows a sequence computes, which can start inside a block
    const int max_q_blocks = ta_cdiv(max_seqlen, BLOCK_Q) + (q_start != nullptr ? 1 : 0);
    const int num_blocks = num_seqs * num_q_heads * max_q_blocks;

    constexpr int q_phase_smem = BLOCK_Q * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                               + BLOCK_Q * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int v_phase_smem = HEAD_DIM * (BLOCK_KV / 2) * sizeof(__nv_fp4x2_e2m1)
                                + HEAD_DIM * (BLOCK_KV / 16) * sizeof(__nv_fp8_e4m3);
    constexpr int k_phase_smem = BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1)
                                + BLOCK_KV * SCALE_DIM * sizeof(__nv_fp8_e4m3);
    constexpr int smem_size = q_phase_smem + v_phase_smem > k_phase_smem
                            ? q_phase_smem + v_phase_smem : k_phase_smem;

    if (prefix_args.prefix_len != nullptr) {  // the same kernel, with the prefix's key tiles in shared memory
        // the bindings allow the prefix for causal bfloat16 attention only
        if constexpr (std::is_same_v<T, __nv_bfloat16> && CAUSAL) {
            // two buffers of K and V tiles of 32 bfloat16 keys: buf1 over the FP4 loop's memory, buf0 past it
            constexpr int prefix_buf = 2 * 32 * HEAD_DIM * 2;
            static_assert(smem_size <= prefix_buf, "the FP4 loop's memory must fit in buf1");
            constexpr int prefix_smem_size = 2 * prefix_buf;
            auto kernel = O_fp4 != nullptr
                ? fp4_attention_varlen_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM,
                                              NUM_WARPS, WARP_Q, true, true>
                : fp4_attention_varlen_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM,
                                              NUM_WARPS, WARP_Q, false, true>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, prefix_smem_size);
            kernel<<<num_blocks, TB_SIZE, prefix_smem_size, stream>>>(
                Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
                q_tokens, k_tokens, v_tokens, max_q_blocks, num_q_heads, num_kv_heads, O_fp4, O_sf,
                out_global_scale, q_start, out_start, lse_out, out_tokens, prefix_args);
        }
        return;
    }

    auto kernel = O_fp4 != nullptr
        ? fp4_attention_varlen_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM,
                                      NUM_WARPS, WARP_Q, true>
        : fp4_attention_varlen_kernel<T, CAUSAL, BLOCK_Q, BLOCK_KV, HEAD_DIM, HEAD_DIM_2, SCALE_DIM,
                                      NUM_WARPS, WARP_Q, false>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<num_blocks, TB_SIZE, smem_size, stream>>>(
        Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
        q_tokens, k_tokens, v_tokens, max_q_blocks, num_q_heads, num_kv_heads, O_fp4, O_sf, out_global_scale,
        q_start, out_start, lse_out, out_tokens, PagedBf16PrefixArgs{});
}

template<typename T, bool CAUSAL>
static void dispatch_fp4_attention(
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    T* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
    if (head_dim == 64) {
        launch_fp4_attention<T, CAUSAL, 64>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    } else if (head_dim == 128) {
        launch_fp4_attention<T, CAUSAL, 128>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    } else {
        launch_fp4_attention_hd256<T, CAUSAL>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    }
}

template<typename T, bool CAUSAL>
static void fp4_attention_nvfp4_typed(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
    auto Q = reinterpret_cast<const __nv_fp4x2_e2m1*>(Q_raw);
    auto K = reinterpret_cast<const __nv_fp4x2_e2m1*>(K_raw);
    auto V = reinterpret_cast<const __nv_fp4x2_e2m1*>(V_raw);
    auto S_Q = reinterpret_cast<const __nv_fp8_e4m3*>(S_Q_raw);
    auto S_K = reinterpret_cast<const __nv_fp8_e4m3*>(S_K_raw);
    auto S_V = reinterpret_cast<const __nv_fp8_e4m3*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    dispatch_fp4_attention<T, CAUSAL>(
        Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
        kv_capacity, num_q_heads, num_kv_heads, head_dim);
}

void fp4_attention_causal_nvfp4(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        fp4_attention_nvfp4_typed<__nv_bfloat16, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    } else {
        fp4_attention_nvfp4_typed<half, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}

void fp4_attention_noncausal_nvfp4(
    const void* Q_raw,
    const void* K_raw,
    const void* V_raw,
    const void* S_Q_raw,
    const void* S_K_raw,
    const void* S_V_raw,
    void* O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16) {
    if (is_bf16) {
        fp4_attention_nvfp4_typed<__nv_bfloat16, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    } else {
        fp4_attention_nvfp4_typed<half, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
}

template<typename T, bool CAUSAL>
static void fp4_attention_varlen_nvfp4_typed(
    const void* Q_raw, const void* K_raw, const void* V_raw,
    const void* S_Q_raw, const void* S_K_raw, const void* S_V_raw,
    const float* key_bias, void* O_raw, const int* cu_seqlens, const int* cu_seqlens_padded,
    int num_seqs, int max_seqlen, int q_tokens, int k_tokens, int v_tokens,
    int num_q_heads, int num_kv_heads, int head_dim, uint8_t* O_fp4, uint8_t* O_sf,
    const float* out_global_scale, const int* q_start, const int* out_start, float* lse_out, int out_tokens,
    cudaStream_t stream, const PagedBf16PrefixArgs& prefix_args) {
    auto Q = reinterpret_cast<const __nv_fp4x2_e2m1*>(Q_raw);
    auto K = reinterpret_cast<const __nv_fp4x2_e2m1*>(K_raw);
    auto V = reinterpret_cast<const __nv_fp4x2_e2m1*>(V_raw);
    auto S_Q = reinterpret_cast<const __nv_fp8_e4m3*>(S_Q_raw);
    auto S_K = reinterpret_cast<const __nv_fp8_e4m3*>(S_K_raw);
    auto S_V = reinterpret_cast<const __nv_fp8_e4m3*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);
    if (head_dim == 64) {
        launch_fp4_attention_varlen<T, CAUSAL, 64>(
            Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
            num_seqs, max_seqlen, q_tokens, k_tokens, v_tokens, num_q_heads, num_kv_heads,
            O_fp4, O_sf, out_global_scale, q_start, out_start, lse_out, out_tokens, stream, prefix_args);
    } else {
        launch_fp4_attention_varlen<T, CAUSAL, 128>(
            Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
            num_seqs, max_seqlen, q_tokens, k_tokens, v_tokens, num_q_heads, num_kv_heads,
            O_fp4, O_sf, out_global_scale, q_start, out_start, lse_out, out_tokens, stream, prefix_args);
    }
}

void fp4_attention_varlen_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    const float* key_bias, void* O, const int* cu_seqlens, const int* cu_seqlens_padded,
    int num_seqs, int max_seqlen, int q_tokens, int k_tokens, int v_tokens,
    int num_q_heads, int num_kv_heads, int head_dim, bool causal, bool is_bf16,
    uint8_t* O_fp4, uint8_t* O_sf, const float* out_global_scale,
    const int* q_start, const int* out_start, float* lse_out, int out_tokens, cudaStream_t stream,
    const PagedBf16PrefixArgs* paged_prefix) {
    const PagedBf16PrefixArgs prefix_args = paged_prefix != nullptr ? *paged_prefix : PagedBf16PrefixArgs{};
    if (is_bf16 && causal) {
        fp4_attention_varlen_nvfp4_typed<__nv_bfloat16, true>(
            Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
            num_seqs, max_seqlen, q_tokens, k_tokens, v_tokens, num_q_heads, num_kv_heads, head_dim,
            O_fp4, O_sf, out_global_scale, q_start, out_start, lse_out, out_tokens, stream, prefix_args);
    } else if (is_bf16) {
        fp4_attention_varlen_nvfp4_typed<__nv_bfloat16, false>(
            Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
            num_seqs, max_seqlen, q_tokens, k_tokens, v_tokens, num_q_heads, num_kv_heads, head_dim,
            O_fp4, O_sf, out_global_scale, q_start, out_start, lse_out, out_tokens, stream, prefix_args);
    } else if (causal) {
        fp4_attention_varlen_nvfp4_typed<half, true>(
            Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
            num_seqs, max_seqlen, q_tokens, k_tokens, v_tokens, num_q_heads, num_kv_heads, head_dim,
            O_fp4, O_sf, out_global_scale, q_start, out_start, lse_out, out_tokens, stream, prefix_args);
    } else {
        fp4_attention_varlen_nvfp4_typed<half, false>(
            Q, K, V, S_Q, S_K, S_V, key_bias, O, cu_seqlens, cu_seqlens_padded,
            num_seqs, max_seqlen, q_tokens, k_tokens, v_tokens, num_q_heads, num_kv_heads, head_dim,
            O_fp4, O_sf, out_global_scale, q_start, out_start, lse_out, out_tokens, stream, prefix_args);
    }
}
