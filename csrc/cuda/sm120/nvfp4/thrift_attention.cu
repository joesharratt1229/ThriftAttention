#include <cstdint>
#include <float.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include "thriftattention/sm120/cuda_common.cuh"

// ===========================================================================
// Mixed-precision causal attention kernel.
// FP4 is used for non-selected KV blocks; FP16 is used for selected KV blocks.
// Grid: (batch * heads * query_blocks), one CTA per query tile.
// ===========================================================================

template<int BLOCK_Q, int BLOCK_KV_FP4, int HEAD_DIM,
         int HEAD_DIM_2, int SCALE_DIM,
         int NUM_WARPS, int WARP_Q>
__launch_bounds__(NUM_WARPS * TA_WARP_SIZE)
__global__
void thrift_attention_causal_kernel(
    const half* Q_fp16_in,
    const half* K_fp16_in,
    const half* V_fp16_in,
    const unsigned long long* topk_mask_in,
    int topk_word_count,
    const __nv_fp4x2_e2m1* Q,
    const __nv_fp4x2_e2m1* K,
    const __nv_fp4x2_e2m1* V,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    __half* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;
    constexpr int MMA_M = 16;
    constexpr int MMA_K_FP4 = 64;
    constexpr int MMA_K_FP16 = 16;
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

    Q   += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM_2;
    K   += kv_bid * kv_capacity * HEAD_DIM_2;
    V   += kv_bid * HEAD_DIM * (v_kv / 2);
    S_Q += (q_bid * q_len + q_block_id * BLOCK_Q) * SCALE_DIM;
    S_K += kv_bid * kv_capacity * SCALE_DIM;
    S_V += kv_bid * v_kv * SCALE_DIM;
    O   += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM;

    Q_fp16_in += (q_bid * q_len + q_block_id * BLOCK_Q) * HEAD_DIM;
    K_fp16_in += kv_bid * kv_len * HEAD_DIM;
    V_fp16_in += kv_bid * kv_len * HEAD_DIM;

    extern __shared__ uint8_t smem[];

    // ---- Persistent register state ----
    uint32_t Q_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K_FP4][4];
    uint32_t sfQ_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K_FP4];

    float rowmax[WARP_Q / MMA_M][2];
    float rowsum[WARP_Q / MMA_M][2] = {};
    float O_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    // ==================== PHASE 1: Load Q FP4 + scales -> registers ====================
    const uint32_t Q_smem    = __cvta_generic_to_shared(smem);
    const uint32_t Q_sf_smem = Q_smem + BLOCK_Q * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);

    ta_gmem_to_smem<BLOCK_Q, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(Q_smem, Q, tid, HEAD_DIM_2);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = (lane_id / 16) * 16;
        uint32_t Q_ld_base = ta_swizzle<HEAD_DIM_2>(Q_smem + row_off * HEAD_DIM_2 + col_off);

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
                uint32_t addr = Q_ld_base;
                addr += mma_id_q * MMA_M * HEAD_DIM_2;
                addr ^= mma_id_d * (MMA_K_FP4 / 2);
                ta_ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
            }
    }

    ta_load_scales<BLOCK_Q, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(Q_sf_smem, S_Q, SCALE_DIM, tid);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    {
        int sf_row_q = 0;
        if (lane_id % 4 == 0)      sf_row_q = (lane_id / 4);
        else if (lane_id % 4 == 1) sf_row_q = (lane_id / 4) + 8;

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
                const int row = warp_id * WARP_Q + mma_id_q * MMA_M + sf_row_q;
                const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(sfQ_rmem[mma_id_q][mma_id_d])
                    : "r"(Q_sf_smem + offset));
            }
    }

    // ==================== PHASE 1b: Load Q FP16 -> registers ====================
    const int q_block_start = q_block_id * BLOCK_Q;
    const int max_kv_pos = q_block_start + BLOCK_Q - 1;
    const int num_kv_iters = min(max_kv_pos / BLOCK_KV_FP4 + 1, ta_cdiv(kv_len, BLOCK_KV_FP4));


    constexpr int TOPK_UNIT_TOKENS = 64;
    constexpr int TOPK_UNITS_PER_ITER = BLOCK_KV_FP4 / TOPK_UNIT_TOKENS;
    static_assert(BLOCK_KV_FP4 == 64 || BLOCK_KV_FP4 == 128,
                  "attention top-k bitset assumes 64-token top-k units");
    const int total_topk_units = ta_cdiv(kv_len, TOPK_UNIT_TOKENS);

    constexpr int MAX_KV_BLOCK_WORDS = 32;  // 2048 * 64 tokens
    const int num_kv_words = min(min(ta_cdiv(total_topk_units, 64), topk_word_count), MAX_KV_BLOCK_WORDS);
    __shared__ unsigned long long topk_mask[MAX_KV_BLOCK_WORDS];
    if (tid < MAX_KV_BLOCK_WORDS) {
        topk_mask[tid] = (tid < num_kv_words)
            ? topk_mask_in[static_cast<int64_t>(q_bid) * topk_word_count + tid]
            : 0ULL;
    }
    __syncthreads();

    const int causal_topk_units = min(num_kv_iters * TOPK_UNITS_PER_ITER, total_topk_units);
    const int causal_words = min(ta_cdiv(causal_topk_units, 64), num_kv_words);
    bool has_fp16_thread = false;
    for (int w = tid; w < causal_words; w += TB_SIZE) {
        uint64_t word_mask = topk_mask[w];
        if (w == causal_words - 1 && (causal_topk_units & 63))
            word_mask &= ((1ULL << (causal_topk_units & 63)) - 1ULL);
        has_fp16_thread |= (word_mask != 0ULL);
    }
    const bool has_fp16 = __syncthreads_or(has_fp16_thread);

    uint32_t Q_fp16_rmem[HEAD_DIM / MMA_K_FP16][4];
    if (has_fp16) {
        const uint32_t Q_fp16_smem = __cvta_generic_to_shared(smem);
        constexpr int Q_FP16_ROW_BYTES = HEAD_DIM * (int)sizeof(half);
        constexpr int Q_FP16_TOTAL_BYTES = BLOCK_Q * Q_FP16_ROW_BYTES;
        constexpr int Q_FP16_TOTAL_CHUNKS = Q_FP16_TOTAL_BYTES / 16;

        for (int chunk = tid; chunk < Q_FP16_TOTAL_CHUNKS; chunk += TB_SIZE) {
            const int byte_off = chunk * 16;
            const int row = byte_off / Q_FP16_ROW_BYTES;
            const int col = byte_off % Q_FP16_ROW_BYTES;
            uint32_t dst = ta_swizzle<Q_FP16_ROW_BYTES>(Q_fp16_smem + row * Q_FP16_ROW_BYTES + col);
            const char* src = reinterpret_cast<const char*>(Q_fp16_in + row * HEAD_DIM) + col;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        const int q_row = warp_id * WARP_Q + (lane_id % 16);
        const int q_col_bytes = (lane_id / 16) * 8 * (int)sizeof(half);
        uint32_t Q_fp16_ld_base = ta_swizzle<Q_FP16_ROW_BYTES>(
            Q_fp16_smem + q_row * Q_FP16_ROW_BYTES + q_col_bytes);

        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP16; mma_id_d++) {
            uint32_t addr = Q_fp16_ld_base;
            addr ^= mma_id_d * MMA_K_FP16 * (int)sizeof(half);
            ta_ldmatrix_x4(Q_fp16_rmem[mma_id_d], addr);
        }
    }

    // ==================== PHASE 2: Single-pass KV loop ====================
    __syncthreads();

    const uint32_t smem_base = __cvta_generic_to_shared(smem);

    // FP4 smem addresses
    const uint32_t K_smem_fp4    = smem_base;
    const uint32_t K_sf_smem     = K_smem_fp4 + BLOCK_KV_FP4 * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1);
    const uint32_t V_smem_fp4    = K_sf_smem + BLOCK_KV_FP4 * SCALE_DIM * (int)sizeof(__nv_fp8_e4m3);
    const uint32_t V_sf_smem_fp4 = V_smem_fp4 + HEAD_DIM * (BLOCK_KV_FP4 / 2) * (int)sizeof(__nv_fp4x2_e2m1);

    // FP16 smem address (alias with FP4 — never used simultaneously)
    // K and V loaded sequentially into the same region (not simultaneously)
    const uint32_t KV_fp16_smem = smem_base;

    uint32_t K_ld_base_fp4;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base_fp4 = ta_swizzle<HEAD_DIM_2>(K_smem_fp4 + row_off * HEAD_DIM_2 + col_off);
    }

    const int q_row_upper = warp_id * WARP_Q + (lane_id / 4);
    const int q_row_lower = q_row_upper + 8;
    const int q_row_upper_global = q_block_start + q_row_upper;
    const int q_row_lower_global = q_block_start + q_row_lower;
    const int k_col_base  = (lane_id % 4) * 2;

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        bool is_topk = false;
        #pragma unroll
        for (int u = 0; u < TOPK_UNITS_PER_ITER; u++) {
            const int topk_unit = kv_iter * TOPK_UNITS_PER_ITER + u;
            const int topk_word = topk_unit >> 6;
            is_topk |= (topk_word < MAX_KV_BLOCK_WORDS) &&
                       ((topk_mask[topk_word] >> (topk_unit & 63)) & 1ULL);
        }
        const int k_block_start = kv_iter * BLOCK_KV_FP4;
        const bool needs_causal_mask = (k_block_start + BLOCK_KV_FP4 - 1) > q_block_start;

        if (is_topk) {
            // ---- FP16 path: process all BLOCK_KV_FP4 rows at once ----
            // Split K/V loading to reuse smem. Stream tiles to save registers.
            constexpr int FP16_ROWS = BLOCK_KV_FP4;
            constexpr int FP16_ROW_BYTES = HEAD_DIM * (int)sizeof(half);
            constexpr int FP16_N_TILES = FP16_ROWS / MMA_N;
            constexpr int FP16_PV_CHUNKS = FP16_ROWS / MMA_K_FP16;

            const half* K_fp16_ptr = K_fp16_in + kv_iter * FP16_ROWS * HEAD_DIM;
            ta_gmem_to_smem<FP16_ROWS, HEAD_DIM, TB_SIZE, half>(
                KV_fp16_smem, K_fp16_ptr, tid, HEAD_DIM);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_all;");
            __syncthreads();

            uint32_t K_fp16_ld_base = ta_swizzle<FP16_ROW_BYTES>(
                KV_fp16_smem + ((lane_id % 8) * HEAD_DIM + (lane_id / 8) * 8) * (int)sizeof(half));

            float S_fp16[FP16_N_TILES][4] = {};
            for (int mma_id_kv = 0; mma_id_kv < FP16_N_TILES; mma_id_kv++) {
                uint32_t K_tile[HEAD_DIM / MMA_K_FP16][2];
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP16; mma_id_d++) {
                    uint32_t addr = K_fp16_ld_base
                        + mma_id_kv * MMA_N * FP16_ROW_BYTES;
                    addr ^= mma_id_d * MMA_K_FP16 * (int)sizeof(half);
                    ta_ldmatrix_x2(K_tile[mma_id_d], addr);
                }
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP16; mma_id_d++)
                    ta_mma_m16n8k16_f16(
                        Q_fp16_rmem[mma_id_d], K_tile[mma_id_d], S_fp16[mma_id_kv]);
            }

            __syncthreads();
            const half* V_fp16_ptr = V_fp16_in + kv_iter * FP16_ROWS * HEAD_DIM;
            ta_gmem_to_smem<FP16_ROWS, HEAD_DIM, TB_SIZE, half>(
                KV_fp16_smem, V_fp16_ptr, tid, HEAD_DIM);
            asm volatile("cp.async.commit_group;");

            if (needs_causal_mask) {
                for (int mma_id_kv = 0; mma_id_kv < FP16_N_TILES; mma_id_kv++) {
                    const int k_col_0 = k_block_start + mma_id_kv * MMA_N + k_col_base;
                    const int k_col_1 = k_col_0 + 1;
                    if (k_col_0 > q_row_upper_global) S_fp16[mma_id_kv][0] = -INFINITY;
                    if (k_col_1 > q_row_upper_global) S_fp16[mma_id_kv][1] = -INFINITY;
                    if (k_col_0 > q_row_lower_global) S_fp16[mma_id_kv][2] = -INFINITY;
                    if (k_col_1 > q_row_lower_global) S_fp16[mma_id_kv][3] = -INFINITY;
                }
            }

            for (int mma_id_kv = 0; mma_id_kv < FP16_N_TILES; mma_id_kv++)
                for (int r = 0; r < 4; r++)
                    S_fp16[mma_id_kv][r] *= softmax_scale;

            float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
            for (int mma_id_kv = 0; mma_id_kv < FP16_N_TILES; mma_id_kv++) {
                this_rowmax[0] = max(this_rowmax[0], max(S_fp16[mma_id_kv][0], S_fp16[mma_id_kv][1]));
                this_rowmax[1] = max(this_rowmax[1], max(S_fp16[mma_id_kv][2], S_fp16[mma_id_kv][3]));
            }
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
            this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
            this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

            this_rowmax[0] = max(this_rowmax[0], rowmax[0][0]);
            this_rowmax[1] = max(this_rowmax[1], rowmax[0][1]);

            float rescale[2] = {
                __expf(rowmax[0][0] - this_rowmax[0]),
                __expf(rowmax[0][1] - this_rowmax[1])
            };
            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                O_rmem[0][mma_id_d][0] *= rescale[0];
                O_rmem[0][mma_id_d][1] *= rescale[0];
                O_rmem[0][mma_id_d][2] *= rescale[1];
                O_rmem[0][mma_id_d][3] *= rescale[1];
            }
            rowmax[0][0] = this_rowmax[0];
            rowmax[0][1] = this_rowmax[1];

            float this_rowsumexp[2] = {};
            uint32_t P_rmem[FP16_PV_CHUNKS][4];

            for (int mma_blk = 0; mma_blk < FP16_PV_CHUNKS; mma_blk++) {
                const int t0 = 2 * mma_blk;
                const int t1 = t0 + 1;
                float *r0 = S_fp16[t0];
                float *r1 = S_fp16[t1];

                r0[0] = __expf(r0[0] - rowmax[0][0]);
                r0[1] = __expf(r0[1] - rowmax[0][0]);
                r0[2] = __expf(r0[2] - rowmax[0][1]);
                r0[3] = __expf(r0[3] - rowmax[0][1]);

                r1[0] = __expf(r1[0] - rowmax[0][0]);
                r1[1] = __expf(r1[1] - rowmax[0][0]);
                r1[2] = __expf(r1[2] - rowmax[0][1]);
                r1[3] = __expf(r1[3] - rowmax[0][1]);

                this_rowsumexp[0] += r0[0] + r0[1] + r1[0] + r1[1];
                this_rowsumexp[1] += r0[2] + r0[3] + r1[2] + r1[3];

                constexpr float FP4_RANGE = 448.0f * 6.0f;
                __half2* p = reinterpret_cast<__half2*>(P_rmem[mma_blk]);
                p[0] = __floats2half2_rn(r0[0] * FP4_RANGE, r0[1] * FP4_RANGE);
                p[1] = __floats2half2_rn(r0[2] * FP4_RANGE, r0[3] * FP4_RANGE);
                p[2] = __floats2half2_rn(r1[0] * FP4_RANGE, r1[1] * FP4_RANGE);
                p[3] = __floats2half2_rn(r1[2] * FP4_RANGE, r1[3] * FP4_RANGE);
            }

            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
            this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
            this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);
            rowsum[0][0] = rowsum[0][0] * rescale[0] + this_rowsumexp[0];
            rowsum[0][1] = rowsum[0][1] * rescale[1] + this_rowsumexp[1];

            asm volatile("cp.async.wait_all;");
            __syncthreads();

            uint32_t V_fp16_ld_base = ta_swizzle<FP16_ROW_BYTES>(
                KV_fp16_smem + ((lane_id % 16) * HEAD_DIM + (lane_id / 16) * 8) * (int)sizeof(half));

            for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
                for (int mma_id_kv = 0; mma_id_kv < FP16_PV_CHUNKS; mma_id_kv++) {
                    uint32_t V_tile[2];
                    uint32_t addr = V_fp16_ld_base
                        + mma_id_kv * MMA_K_FP16 * FP16_ROW_BYTES;
                    addr ^= mma_id_d * MMA_N * (int)sizeof(half);
                    ta_ldmatrix_x2_trans(V_tile, addr);
                    ta_mma_m16n8k16_f16(P_rmem[mma_id_kv], V_tile, O_rmem[0][mma_id_d]);
                }

            __syncthreads();
        } else {
            // ---- FP4 path: process 64 KV rows ----
            const __nv_fp4x2_e2m1* K_ptr = K + kv_iter * BLOCK_KV_FP4 * HEAD_DIM_2;
            const __nv_fp8_e4m3*   SK_ptr = S_K + kv_iter * BLOCK_KV_FP4 * SCALE_DIM;
            const __nv_fp4x2_e2m1* V_ptr = V + kv_iter * BLOCK_KV_FP4 / 2;
            const __nv_fp8_e4m3*   SV_ptr = S_V + kv_iter * BLOCK_KV_FP4 / 16;

            // K load (group 1)
            ta_gmem_to_smem<BLOCK_KV_FP4, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_smem_fp4, K_ptr, tid, HEAD_DIM_2);
            ta_load_scales<BLOCK_KV_FP4, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(K_sf_smem, SK_ptr, SCALE_DIM, tid);
            asm volatile("cp.async.commit_group;");
            // V load (group 2) — overlaps with QK^T compute
            ta_gmem_to_smem<HEAD_DIM, BLOCK_KV_FP4 / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_smem_fp4, V_ptr, tid, v_kv / 2);
            ta_load_scales<HEAD_DIM, BLOCK_KV_FP4 / 16, TB_SIZE, __nv_fp8_e4m3>(V_sf_smem_fp4, SV_ptr, v_kv / 16, tid);
            asm volatile("cp.async.commit_group;");
            // Wait for K only (V still in flight)
            asm volatile("cp.async.wait_group 1;");
            __syncthreads();

            uint32_t K_rmem[BLOCK_KV_FP4 / MMA_N][HEAD_DIM / MMA_K_FP4][2];
            uint32_t sfK_rmem[BLOCK_KV_FP4 / MMA_N][HEAD_DIM / MMA_K_FP4];

            if constexpr (HEAD_DIM / MMA_K_FP4 >= 2) {
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_N; mma_id_kv++) {
                    uint32_t addr = K_ld_base_fp4 + mma_id_kv * MMA_N * HEAD_DIM_2;
                    ta_ldmatrix_x4(K_rmem[mma_id_kv][0], addr);
                }
            } else {
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_N; mma_id_kv++)
                    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
                        uint32_t addr = K_ld_base_fp4;
                        addr += mma_id_kv * MMA_N * HEAD_DIM_2;
                        addr ^= mma_id_d * (MMA_K_FP4 / 2);
                        ta_ldmatrix_x2(K_rmem[mma_id_kv][mma_id_d], addr);
                    }
            }

            const int sf_row_k = lane_id / 4;
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_N; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++) {
                    const int row = mma_id_kv * MMA_N + sf_row_k;
                    const uint32_t offset = (row * SCALE_DIM + mma_id_d * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                    asm volatile("ld.shared.u32 %0, [%1];"
                        : "=r"(sfK_rmem[mma_id_kv][mma_id_d])
                        : "r"(K_sf_smem + offset));
                }

            float S_rmem[WARP_Q / MMA_M][BLOCK_KV_FP4 / MMA_N][4] = {};

            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_N; mma_id_kv++)
                    for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_K_FP4; mma_id_d++)
                        ta_mma_m16n8k64_nvfp4(
                            Q_rmem[mma_id_q][mma_id_d],
                            K_rmem[mma_id_kv][mma_id_d],
                            sfQ_rmem[mma_id_q][mma_id_d],
                            sfK_rmem[mma_id_kv][mma_id_d],
                            S_rmem[mma_id_q][mma_id_kv]);

            if (needs_causal_mask) {
                for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_N; mma_id_kv++) {
                        const int k_phys_0 = mma_id_kv * MMA_N + k_col_base;
                        const int k_phys_1 = k_phys_0 + 1;
                        const int k_col_0 = k_block_start + k_phys_0;
                        const int k_col_1 = k_block_start + k_phys_1;
                        if (k_col_0 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][0] = -INFINITY;
                        if (k_col_1 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][1] = -INFINITY;
                        if (k_col_0 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][2] = -INFINITY;
                        if (k_col_1 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][3] = -INFINITY;
                    }
            }

            uint32_t S_fp4_rmem[WARP_Q / MMA_M][BLOCK_KV_FP4 / MMA_K_FP4][4];
            uint32_t S_fp4_s_rmem[WARP_Q / MMA_M][BLOCK_KV_FP4 / MMA_K_FP4];

            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_N; mma_id_kv++)
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                        S_rmem[mma_id_q][mma_id_kv][reg_id] *= softmax_scale;

                float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
                float p_group_max_upper[BLOCK_KV_FP4 / MMA_N / 2];
                float p_group_max_lower[BLOCK_KV_FP4 / MMA_N / 2];
                for (int blk = 0; blk < BLOCK_KV_FP4 / MMA_N / 2; blk++) {
                    const int t0 = 2 * blk, t1 = t0 + 1;
                    float gmax_upper = max(max(S_rmem[mma_id_q][t0][0], S_rmem[mma_id_q][t0][1]),
                                           max(S_rmem[mma_id_q][t1][0], S_rmem[mma_id_q][t1][1]));
                    float gmax_lower = max(max(S_rmem[mma_id_q][t0][2], S_rmem[mma_id_q][t0][3]),
                                           max(S_rmem[mma_id_q][t1][2], S_rmem[mma_id_q][t1][3]));
                    gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 1));
                    gmax_upper = max(gmax_upper, __shfl_xor_sync(0xFFFFFFFF, gmax_upper, 2));
                    gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 1));
                    gmax_lower = max(gmax_lower, __shfl_xor_sync(0xFFFFFFFF, gmax_lower, 2));
                    p_group_max_upper[blk] = gmax_upper;
                    p_group_max_lower[blk] = gmax_lower;
                    this_rowmax[0] = max(this_rowmax[0], gmax_upper);
                    this_rowmax[1] = max(this_rowmax[1], gmax_lower);
                }

                this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
                this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

                float rescale[2] = {
                    __expf(rowmax[mma_id_q][0] - this_rowmax[0]),
                    __expf(rowmax[mma_id_q][1] - this_rowmax[1])
                };
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                    O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                    O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
                }
                rowmax[mma_id_q][0] = this_rowmax[0];
                rowmax[mma_id_q][1] = this_rowmax[1];

                float this_rowsumexp[2] = {};
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_N; mma_id_kv++) {
                    float *r = S_rmem[mma_id_q][mma_id_kv];
                    r[0] = __expf(r[0] - rowmax[mma_id_q][0]);
                    r[1] = __expf(r[1] - rowmax[mma_id_q][0]);
                    r[2] = __expf(r[2] - rowmax[mma_id_q][1]);
                    r[3] = __expf(r[3] - rowmax[mma_id_q][1]);
                    this_rowsumexp[0] += r[0] + r[1];
                    this_rowsumexp[1] += r[2] + r[3];
                }
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);
                rowsum[mma_id_q][0] = rowsum[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
                rowsum[mma_id_q][1] = rowsum[mma_id_q][1] * rescale[1] + this_rowsumexp[1];

                constexpr float FP4_RANGE = 448.0f * 6.0f;
                constexpr float FP4_MAX   = 6.0f;

                float sf_P_upper[BLOCK_KV_FP4 / MMA_N / 2];
                float sf_P_lower[BLOCK_KV_FP4 / MMA_N / 2];
                for (int blk = 0; blk < BLOCK_KV_FP4 / MMA_N / 2; blk++) {
                    const int t0 = 2 * blk, t1 = t0 + 1;
                    const float gmax_upper = p_group_max_upper[blk];
                    const float gmax_lower = p_group_max_lower[blk];
                    const bool valid_upper = gmax_upper > -FLT_MAX * 0.5f;
                    const bool valid_lower = gmax_lower > -FLT_MAX * 0.5f;
                    const float inv_upper = valid_upper ? (FP4_MAX * __expf(rowmax[mma_id_q][0] - gmax_upper)) : 0.0f;
                    const float inv_lower = valid_lower ? (FP4_MAX * __expf(rowmax[mma_id_q][1] - gmax_lower)) : 0.0f;
                    sf_P_upper[blk] = valid_upper ? (FP4_RANGE / FP4_MAX * __expf(gmax_upper - rowmax[mma_id_q][0])) : 1.0f;
                    sf_P_lower[blk] = valid_lower ? (FP4_RANGE / FP4_MAX * __expf(gmax_lower - rowmax[mma_id_q][1])) : 1.0f;
                    S_rmem[mma_id_q][t0][0] *= inv_upper;  S_rmem[mma_id_q][t0][1] *= inv_upper;
                    S_rmem[mma_id_q][t1][0] *= inv_upper;  S_rmem[mma_id_q][t1][1] *= inv_upper;
                    S_rmem[mma_id_q][t0][2] *= inv_lower;  S_rmem[mma_id_q][t0][3] *= inv_lower;
                    S_rmem[mma_id_q][t1][2] *= inv_lower;  S_rmem[mma_id_q][t1][3] *= inv_lower;
                }

                const int qid = lane_id & 3;
                for (int g = 0; g < BLOCK_KV_FP4 / MMA_N / 4; g++) {
                    for (int r = 0; r < 4; r++) {
                        float send = (qid & 1) ? S_rmem[mma_id_q][g*4+0][r] : S_rmem[mma_id_q][g*4+1][r];
                        float recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                        if (qid & 1) S_rmem[mma_id_q][g*4+0][r] = recv;
                        else         S_rmem[mma_id_q][g*4+1][r] = recv;

                        send = (qid & 1) ? S_rmem[mma_id_q][g*4+2][r] : S_rmem[mma_id_q][g*4+3][r];
                        recv = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                        if (qid & 1) S_rmem[mma_id_q][g*4+2][r] = recv;
                        else         S_rmem[mma_id_q][g*4+3][r] = recv;

                        send = (qid & 2) ? S_rmem[mma_id_q][g*4+0][r] : S_rmem[mma_id_q][g*4+2][r];
                        recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                        if (qid & 2) S_rmem[mma_id_q][g*4+0][r] = recv;
                        else         S_rmem[mma_id_q][g*4+2][r] = recv;

                        send = (qid & 2) ? S_rmem[mma_id_q][g*4+1][r] : S_rmem[mma_id_q][g*4+3][r];
                        recv = __shfl_xor_sync(0xFFFFFFFF, send, 2);
                        if (qid & 2) S_rmem[mma_id_q][g*4+1][r] = recv;
                        else         S_rmem[mma_id_q][g*4+3][r] = recv;
                    }

                    float *r0 = S_rmem[mma_id_q][g*4], *r1 = S_rmem[mma_id_q][g*4+1],
                          *r2 = S_rmem[mma_id_q][g*4+2], *r3 = S_rmem[mma_id_q][g*4+3];
                    S_fp4_rmem[mma_id_q][0][2*g]   = ta_cvt_8xf32_to_e2m1_packed(
                        r0[1],r0[0],r1[1],r1[0], r2[1],r2[0],r3[1],r3[0]);
                    S_fp4_rmem[mma_id_q][0][2*g+1] = ta_cvt_8xf32_to_e2m1_packed(
                        r0[3],r0[2],r1[3],r1[2], r2[3],r2[2],r3[3],r3[2]);
                }

                for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV_FP4 / MMA_K_FP4; mma_sc_id++) {
                    int base = mma_sc_id * 4;
                    uint32_t sfP_upper_packed = ta_cvt_4xf32_to_e4m3_packed(
                        sf_P_upper[base+1], sf_P_upper[base+0],
                        sf_P_upper[base+3], sf_P_upper[base+2]);
                    uint32_t sfP_lower_packed = ta_cvt_4xf32_to_e4m3_packed(
                        sf_P_lower[base+1], sf_P_lower[base+0],
                        sf_P_lower[base+3], sf_P_lower[base+2]);
                    S_fp4_s_rmem[mma_id_q][mma_sc_id] =
                        (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
                }
            }

            // Wait for V load to complete
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();

            uint32_t V_rmem[BLOCK_KV_FP4 / MMA_K_FP4][HEAD_DIM / MMA_N][2];
            uint32_t sfV_rmem[BLOCK_KV_FP4 / MMA_K_FP4][HEAD_DIM / MMA_N];

            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_K_FP4; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d += 2) {
                    const int n_idx = mma_id_d * MMA_N + (lane_id / 16) * MMA_N + (lane_id % 8);
                    const int k_byte_offset = mma_id_kv * 32 + ((lane_id % 16) / 8) * 16;
                    uint32_t addr = ta_swizzle<BLOCK_KV_FP4 / 2>(
                        V_smem_fp4 + n_idx * (BLOCK_KV_FP4 / 2) + k_byte_offset);
                    ta_ldmatrix_x4(V_rmem[mma_id_kv][mma_id_d], addr);
                }

            constexpr int V_SF_STRIDE = BLOCK_KV_FP4 / 16;
            const int sf_col_v = lane_id / 4;
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_K_FP4; mma_id_kv++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                    const int hd_col = mma_id_d * MMA_N + sf_col_v;
                    const uint32_t offset = (hd_col * V_SF_STRIDE + mma_id_kv * 4) * (uint32_t)sizeof(__nv_fp8_e4m3);
                    asm volatile("ld.shared.u32 %0, [%1];"
                        : "=r"(sfV_rmem[mma_id_kv][mma_id_d])
                        : "r"(V_sf_smem_fp4 + offset));
                }

            for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++)
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV_FP4 / MMA_K_FP4; mma_id_kv++)
                        ta_mma_m16n8k64_nvfp4(
                            S_fp4_rmem[mma_id_q][mma_id_kv],
                            V_rmem[mma_id_kv][mma_id_d],
                            S_fp4_s_rmem[mma_id_q][mma_id_kv],
                            sfV_rmem[mma_id_kv][mma_id_d],
                            O_rmem[mma_id_q][mma_id_d]);

            __syncthreads();
        }
    }

    // ==================== PHASE 3: Output normalization + write ====================
    constexpr float FP4_RANGE_INV = 1.0f / (448.0f * 6.0f);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
            float *regs = O_rmem[mma_id_q][mma_id_d];

            float norm0 = FP4_RANGE_INV / rowsum[mma_id_q][0];
            float norm1 = FP4_RANGE_INV / rowsum[mma_id_q][1];

            regs[0] *= norm0;  regs[1] *= norm0;
            regs[2] *= norm1;  regs[3] *= norm1;

            reinterpret_cast<__half2*>(O + (row + 0) * HEAD_DIM + col)[0] =
                __floats2half2_rn(regs[0], regs[1]);
            reinterpret_cast<__half2*>(O + (row + 8) * HEAD_DIM + col)[0] =
                __floats2half2_rn(regs[2], regs[3]);
        }
}

template<int HEAD_DIM, int BLOCK_Q, int BLOCK_KV_FP4>
static void launch_thrift_attention_causal(
    const half* Q_fp16,
    const half* K_fp16,
    const half* V_fp16,
    const unsigned long long* topk_mask,
    int topk_word_count,
    const __nv_fp4x2_e2m1* Q_fp4,
    const __nv_fp4x2_e2m1* K_fp4,
    const __nv_fp4x2_e2m1* V_fp4,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    __half* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    constexpr int HEAD_DIM_2 = HEAD_DIM / 2;
    constexpr int SCALE_DIM = HEAD_DIM / 16;
    constexpr int WARP_Q = 16;
    constexpr int NUM_WARPS = BLOCK_Q / WARP_Q;
    constexpr int TB_SIZE = NUM_WARPS * TA_WARP_SIZE;

    const int num_blocks = bs * ta_cdiv(q_len, BLOCK_Q);

    constexpr int q_phase_smem =
        BLOCK_Q * HEAD_DIM_2 * (int)sizeof(__nv_fp4x2_e2m1) +
        BLOCK_Q * SCALE_DIM * (int)sizeof(__nv_fp8_e4m3);

    constexpr int v_phase_smem =
        HEAD_DIM * (BLOCK_KV_FP4 / 2) * (int)sizeof(__nv_fp4x2_e2m1) +
        HEAD_DIM * (BLOCK_KV_FP4 / 16) * (int)sizeof(__nv_fp8_e4m3);

    constexpr int fp4_kv_smem = q_phase_smem + v_phase_smem;

    constexpr int q_fp16_smem = BLOCK_Q * HEAD_DIM * (int)sizeof(half);

    // FP16 KV path: loads K or V (not both) at BLOCK_KV_FP4 rows
    constexpr int fp16_kv_smem = BLOCK_KV_FP4 * HEAD_DIM * (int)sizeof(half);

    constexpr int smem_12 = (fp4_kv_smem > q_fp16_smem) ? fp4_kv_smem : q_fp16_smem;
    constexpr int smem_size = (smem_12 > fp16_kv_smem) ? smem_12 : fp16_kv_smem;

    auto kernel = thrift_attention_causal_kernel<
        BLOCK_Q, BLOCK_KV_FP4, HEAD_DIM,
        HEAD_DIM_2, SCALE_DIM, NUM_WARPS, WARP_Q>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel<<<num_blocks, TB_SIZE, smem_size>>>(
        Q_fp16, K_fp16, V_fp16,
        topk_mask, topk_word_count,
        Q_fp4, K_fp4, V_fp4,
        S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);

}

template<int HEAD_DIM>
static void dispatch_thrift_attention_causal(
    const half* Q_fp16,
    const half* K_fp16,
    const half* V_fp16,
    const unsigned long long* topk_mask,
    int topk_word_count,
    const __nv_fp4x2_e2m1* Q_fp4,
    const __nv_fp4x2_e2m1* K_fp4,
    const __nv_fp4x2_e2m1* V_fp4,
    const __nv_fp8_e4m3* S_Q,
    const __nv_fp8_e4m3* S_K,
    const __nv_fp8_e4m3* S_V,
    __half* O,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads)
{
    return launch_thrift_attention_causal<HEAD_DIM, 64, 64>(
        Q_fp16, K_fp16, V_fp16, topk_mask, topk_word_count, Q_fp4, K_fp4, V_fp4,
        S_Q, S_K, S_V, O, bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

void thrift_attention_causal_nvfp4(
    const void* Q_fp16_raw,
    const void* K_fp16_raw,
    const void* V_fp16_raw,
    const void* topk_mask_raw,
    int topk_word_count,
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
    int head_dim)
{
    auto Q_fp16 = reinterpret_cast<const half*>(Q_fp16_raw);
    auto K_fp16 = reinterpret_cast<const half*>(K_fp16_raw);
    auto V_fp16 = reinterpret_cast<const half*>(V_fp16_raw);
    auto topk_mask = reinterpret_cast<const unsigned long long*>(topk_mask_raw);

    auto Q = reinterpret_cast<const __nv_fp4x2_e2m1*>(Q_raw);
    auto K = reinterpret_cast<const __nv_fp4x2_e2m1*>(K_raw);
    auto V = reinterpret_cast<const __nv_fp4x2_e2m1*>(V_raw);

    auto S_Q = reinterpret_cast<const __nv_fp8_e4m3*>(S_Q_raw);
    auto S_K = reinterpret_cast<const __nv_fp8_e4m3*>(S_K_raw);
    auto S_V = reinterpret_cast<const __nv_fp8_e4m3*>(S_V_raw);
    auto O = reinterpret_cast<__half*>(O_raw);

    if (head_dim == 64)
        dispatch_thrift_attention_causal<64>(
            Q_fp16, K_fp16, V_fp16,
            topk_mask, topk_word_count,
            Q, K, V, S_Q, S_K, S_V, O,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
    else
        dispatch_thrift_attention_causal<128>(
            Q_fp16, K_fp16, V_fp16,
            topk_mask, topk_word_count,
            Q, K, V, S_Q, S_K, S_V, O,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}
