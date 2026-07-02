// SM120 NVFP4 attention — staged power-of-two softmax rewrite
//
// Three stages applied to the EXP_APPROX=true path (baseline __expf path kept
// intact for A/B, and it also benefits from Stage 2):
//
//   Stage 1 — FUSE, don't append.
//     * The per-element `*= softmax_scale` loop is deleted. rowmax is tracked
//       on RAW scores (max commutes with a positive scale).
//     * scale + base-change(log2e) + max-subtract + magic-round collapse into
//       ONE FFMA per element:  r = fmaf(s_raw, k, MAGIC - m_raw*k).
//     * One FMNMX lower guard (handles -INF mask values and NaN); the upper
//       clamp is unnecessary because s - m <= 0 structurally.
//
//   Stage 2 — DELETE dead work (applies to BOTH paths).
//     * The old sf_P block computed attention_exp(rowmax - rowmax) == exp(0)
//       sixteen times per row-group per iteration. inv == 6.0, sf == 448.0
//       are constants; the e4m3-packed scale word is the compile-time
//       constant 0x7E7E7E7E (448 in e4m3, replicated).
//
//   Stage 3 — bit-built power-of-two weight, SHARED cvt quantization
//     (approx path only).
//     * The weight 4 * 2^n is constructed from the magic-added bits in two
//       integer ops: (bits(r) << 23) + bits(4.0f). No MUFU.EX2 is issued.
//     * Quantization reuses the SAME ta_cvt_8xf32_to_e2m1_packed loop as the
//       baseline: cvt.rn on {4, 2, 1, 0.5, <=0.25} reproduces the
//       power-of-two ladder codes {6, 4, 2, 1, 0} exactly (0.25 is a tie and
//       rounds to even = 0). The x4 bias keeps the row max on the top
//       representable power-of-two rung; since the rowsum accumulates the
//       SAME x4-biased weight, the epilogue only divides out the 448 scale.
//     * Rationale (ncu): the kernel runs at 25% occupancy (164 regs, 3
//       blocks/SM; the cliff to 2 blocks is at 170 regs) with ~2.7 active
//       warps/scheduler — latency-bound, no pipe saturated. So the approx
//       path must SHORTEN the serial QK -> softmax -> PV chain without
//       adding live registers. A previous attempt that hand-packed nibbles
//       (5-op clamp ladder + shift/or chains, ~10 extra ALU ops and ~10
//       extra live values per row) regressed badly for exactly that reason.
//       This version keeps the register profile of the baseline and removes
//       FMUL(log2e) + MUFU.EX2 + FMUL(x6) per element in exchange for
//       FMNMX + SHL + IADD on full-rate pipes.
//
//   Stage 4 — INTEGER-SNAPPED rowmax (approx path only).
//     * rowmax[] stores R = TA_MAGIC + ceil-ish(max * k) (magic-biased rung)
//       instead of the raw max. The per-row FFMA addend is the exact FADD
//       c = 2*MAGIC - R, and the O/rowsum rescale is exactly 2^(R_old - R) —
//       no drift between rescaled O entries and their re-quantization.
//     * A uniform rung shift cancels between numerator and denominator, so
//       snapping costs no accuracy class.
//     * R changes only when a row's max crosses a rung boundary (1/k raw
//       units). The warp votes via __any_sync and SKIPS the O-rescale loop
//       when no row moved — the common case once the running max stabilizes.
//
//   Stage 5 — MMA-accumulated rowsum (approx path only).
//     * The PV mma gets a 17th output tile: P @ ones (B = e2m1 code 2 in
//       every nibble, scale = e4m3 1.0; both compile-time constants, no
//       smem traffic). B being all-ones makes every column of the tile equal
//       to the row's denominator, so each thread reads its own copy — the
//       per-iteration FADD rowsum, the 4-shuffle butterfly, and the running
//       rowsum FMA are all deleted, and the O-rescale/skip logic covers the
//       denominator automatically. Epilogue norm = 1/sum_tile: every
//       packaging constant cancels between numerator and denominator.
//
// Numerics notes for the approx path:
//   * each weight is 2^round((s-m)*k): within x/÷sqrt(2) of exact exp.
//   * Stage 5 changes the denominator to the sum of QUANTIZED weights:
//     elements with n <= -4 (code 0) no longer contribute. Numerator and
//     denominator are now exactly consistent (true weighted average of V
//     rows), instead of a denominator that includes sub-rung tail mass the
//     numerator drops.
//   * fully-padded rows produce NaN garbage (0 * 1/0) instead of ~0/eps
//     garbage; both are garbage for padding, unchanged contract.
//
// Suggested measurement after each stage (ncu):
//   sm__inst_executed_pipe_xu      -> approx drops the EX2 share (~2/3); the
//                                     two cvt.e2m1x2 per 8 values remain
//   launch__registers_per_thread   -> MUST stay <= 170 (3-blocks/SM cliff)
//   smsp__issue_active / stalls    -> latency-bound; watch warp eligibility

#include <cstdint>
#include <float.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "thriftattention/sm120/cuda_common.cuh"

// ---------------------------------------------------------------------------
// Power-of-two softmax helpers (Stage 1 / Stage 3 building blocks)
// ---------------------------------------------------------------------------

// 1.5 * 2^23: floats in [2^23, 2^24) are spaced exactly 1.0 apart, so an FADD
// against this constant performs round-to-nearest-integer on the wide pipe.
constexpr float    TA_MAGIC        = 12582912.0f;
constexpr uint32_t TA_MAGIC_BITS   = 0x4B400000u;   // __float_as_uint(TA_MAGIC)
// Lower guard: keeps round(x) >= -126 so the exponent-field construction
// cannot wrap. fmaxf against this also launders -INF (masked) and NaN inputs.
constexpr float    TA_MAGIC_FLOOR  = TA_MAGIC - 126.0f;
// 2 * TA_MAGIC (= 1.5 * 2^24, exactly representable). For a snapped rung
// R = TA_MAGIC + u stored magic-biased, the per-row FFMA addend
// c = TA_MAGIC - u is recovered exactly as TA_MAGIC_X2 - R (both operands
// sit on the unit grid of [2^23, 2^24), so the FADD is exact).
constexpr float    TA_MAGIC_X2     = 25165824.0f;

// 2^n from the magic-added bits: (bits << 23) drops the constant exactly
// ((TA_MAGIC_BITS << 23) mod 2^32 == 0) and +0x3F800000 supplies the bias.
// Output always has a zero mantissa: only {2^j} are ever produced.
__device__ __forceinline__
float ta_pow2_from_bits(uint32_t br)
{
    return __uint_as_float((br << 23) + 0x3F800000u);
}

// 4 * 2^n from the magic-added bits (bias constant = bits of 4.0f). The x4
// puts the row max on e2m1's top power-of-two rung {4, 2, 1, 0.5}; the same
// biased value feeds the rowsum, so the pair cancels in the epilogue.
// Requires n >= -126 (guaranteed by the TA_MAGIC_FLOOR guard) and n <= 0
// (structural: s - m <= 0).
__device__ __forceinline__
float ta_pow2x4_from_bits(uint32_t br)
{
    return __uint_as_float((br << 23) + 0x40800000u);
}

template<typename T, bool CAUSAL, bool EXP_APPROX, bool MICROBLOCK_P,
         int BLOCK_Q, int BLOCK_KV, int HEAD_DIM,
         int HEAD_DIM_2, int SCALE_DIM,
         int NUM_WARPS, int WARP_Q>
// The 3 pins occupancy at 3 blocks/SM (<= 170 regs with 128 threads on a
// 64K-reg SM): crossing that register cliff costs ~33% of resident warps on
// an already latency-bound kernel, which is never worth a few extra regs.
__launch_bounds__(NUM_WARPS * TA_WARP_SIZE, 3)
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
    constexpr float LOG2_E = 1.4426950408889634f;

    // k folds 1/sqrt(d) and (approx path only) the e->2 base change.
    const float softmax_scale =
        rsqrtf(static_cast<float>(HEAD_DIM)) * (EXP_APPROX ? LOG2_E : 1.0f);

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
    uint32_t K_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
    uint32_t V_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
    uint32_t sfQ_rmem[WARP_Q / MMA_M][HEAD_DIM / MMA_K];
    uint32_t sfK_rmem[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K];
    uint32_t sfV_rmem[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N];

    // Approx (Stage 5): the rowsum rides the PV mma as a 17th output tile
    // against an all-ones e2m1 column block — B all-ones makes every column
    // of that tile equal to the row's sum of QUANTIZED weights, so each
    // thread reads its own copy (no shuffle reduction) and the O-rescale /
    // skip logic covers the denominator for free.
    constexpr int O_TILES = HEAD_DIM / MMA_N + (EXP_APPROX ? 1 : 0);

    // Baseline: RAW-domain running max. Approx: the SNAPPED rung
    // R = TA_MAGIC + ceil-ish(max * k), kept magic-biased so rung deltas and
    // the per-row FFMA addend come out of exact unit-grid FADDs.
    float rowmax[WARP_Q/MMA_M][2];
    float rowsum[WARP_Q/MMA_M][2] = {};   // baseline only
    float O_rmem[WARP_Q/MMA_M][O_TILES][4] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q/MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = EXP_APPROX ? TA_MAGIC_FLOOR : -FLT_MAX;
        rowmax[mma_id_q][1] = EXP_APPROX ? TA_MAGIC_FLOOR : -FLT_MAX;
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
    const uint32_t K_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_sf_smem = K_smem + BLOCK_KV * HEAD_DIM_2 * sizeof(__nv_fp4x2_e2m1);

    const int total_kv_iters = ta_cdiv(kv_len, BLOCK_KV);
    const int max_kv_pos = q_block_id * BLOCK_Q + BLOCK_Q - 1;
    const int num_kv_iters = CAUSAL
        ? min(max_kv_pos / BLOCK_KV + 1, total_kv_iters)
        : total_kv_iters;

    // Pre-compute base address for K ldmatrix.
    uint32_t K_ld_base;
    {
        const int row_off = lane_id % 8;
        const int col_off = (lane_id / 8) * 16;
        K_ld_base = ta_swizzle<HEAD_DIM_2>(K_smem + row_off * HEAD_DIM_2 + col_off);
    }

    // Precompute per-thread query row offsets within the block.
    const int q_block_start = q_block_id * BLOCK_Q;
    const int q_row_upper = warp_id * WARP_Q + (lane_id / 4);       // rows 0-7 of MMA tile
    const int q_row_lower = q_row_upper + 8;                         // rows 8-15 of MMA tile
    const int q_row_upper_global = q_block_start + q_row_upper;
    const int q_row_lower_global = q_block_start + q_row_lower;
    const int k_col_base = (lane_id % 4) * 2;                        // base key col within MMA_N tile

    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
        uint32_t S_fp4_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
        uint32_t S_fp4_s_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K];

        const int k_block_start = kv_iter * BLOCK_KV;
        const bool needs_causal_mask = CAUSAL && ((k_block_start + BLOCK_KV - 1) > q_block_start);

        // Load K first; issue V in a second async group so QK can overlap
        // with the V transfer.
        ta_gmem_to_smem<BLOCK_KV, HEAD_DIM_2, TB_SIZE, __nv_fp4x2_e2m1>(K_smem, K, tid, HEAD_DIM_2);
        ta_load_scales<BLOCK_KV, SCALE_DIM, TB_SIZE, __nv_fp8_e4m3>(K_sf_smem, S_K, SCALE_DIM, tid);
        asm volatile("cp.async.commit_group;");
        ta_gmem_to_smem<HEAD_DIM, BLOCK_KV / 2, TB_SIZE, __nv_fp4x2_e2m1>(V_smem, V, tid, v_kv / 2);
        ta_load_scales<HEAD_DIM, BLOCK_KV / 16, TB_SIZE, __nv_fp8_e4m3>(V_sf_smem, S_V, v_kv / 16, tid);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        if constexpr (HEAD_DIM / MMA_K >= 2) {
            // ldmatrix_x4: lanes 0-15 cover mma_id_d=0, lanes 16-31 cover mma_id_d=1
            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                uint32_t addr = K_ld_base + mma_id_kv * MMA_N * HEAD_DIM_2;
                ta_ldmatrix_x4(K_rmem[mma_id_kv][0], addr);
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

        // ---- Apply causal mask on blocks overlapping the query tile ----
        if constexpr (CAUSAL) {
            if (needs_causal_mask) {
                for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
                    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                        const int k_phys_0 = mma_id_kv * MMA_N + k_col_base;
                        const int k_phys_1 = k_phys_0 + 1;
                        const int k_col_0 = k_block_start + ta_kv_logical_from_physical(k_phys_0);
                        const int k_col_1 = k_block_start + ta_kv_logical_from_physical(k_phys_1);

                        if (k_col_0 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][0] = -INFINITY;
                        if (k_col_1 > q_row_upper_global) S_rmem[mma_id_q][mma_id_kv][1] = -INFINITY;
                        if (k_col_0 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][2] = -INFINITY;
                        if (k_col_1 > q_row_lower_global) S_rmem[mma_id_q][mma_id_kv][3] = -INFINITY;
                    }
                }
            }
        }

        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            // ---------------------------------------------------------------
            // Stage 1: rowmax over RAW scores. In microblock-P mode each
            // 16-entry P scale block is owned by a lane pair: lanes 0/1 hold
            // one block and lanes 2/3 hold the other. The block max only
            // reduces over XOR-1. XOR-2 is used only to exchange the other
            // pair's already-reduced max so rowmax and the four-scale word
            // can be formed without merging the microblocks.
            // ---------------------------------------------------------------
            constexpr int P_SCALE_MMA_N_TILES = 4;
            // checks whether bit 1 of lane id is set
            const bool quad_pair_hi = (lane_id & 2) != 0;
            float own0_upper = -FLT_MAX, own0_lower = -FLT_MAX;
            float own1_upper = -FLT_MAX, own1_lower = -FLT_MAX;
            float other0_upper = -FLT_MAX, other0_lower = -FLT_MAX;
            float other1_upper = -FLT_MAX, other1_lower = -FLT_MAX;
            float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};

            if constexpr (EXP_APPROX && MICROBLOCK_P) {
                //seperates out into 32 physical columsn
                for (int mma_id_kv = 0; mma_id_kv < P_SCALE_MMA_N_TILES; mma_id_kv++) {
                    //computes block max for lower and upper tile
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    own0_upper = max(own0_upper, max(regs[0], regs[1]));
                    own0_lower = max(own0_lower, max(regs[2], regs[3]));
                }
                for (int mma_id_kv = P_SCALE_MMA_N_TILES; mma_id_kv < 2 * P_SCALE_MMA_N_TILES; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    own1_upper = max(own1_upper, max(regs[0], regs[1]));
                    own1_lower = max(own1_lower, max(regs[2], regs[3]));
                }
                
                // pair reduction in quad (a0,...,a7)(a8,....,a15) gets merged 
                // applies this to all parts of tile owned
                own0_upper = max(own0_upper, __shfl_xor_sync(0xFFFFFFFF, own0_upper, 1));
                own0_lower = max(own0_lower, __shfl_xor_sync(0xFFFFFFFF, own0_lower, 1));
                own1_upper = max(own1_upper, __shfl_xor_sync(0xFFFFFFFF, own1_upper, 1));
                own1_lower = max(own1_lower, __shfl_xor_sync(0xFFFFFFFF, own1_lower, 1));
                

                //reuses block max computation to obtain rowmax
                other0_upper = __shfl_xor_sync(0xFFFFFFFF, own0_upper, 2);
                other0_lower = __shfl_xor_sync(0xFFFFFFFF, own0_lower, 2);
                other1_upper = __shfl_xor_sync(0xFFFFFFFF, own1_upper, 2);
                other1_lower = __shfl_xor_sync(0xFFFFFFFF, own1_lower, 2);

                this_rowmax[0] = max(max(own0_upper, own1_upper), max(other0_upper, other1_upper));
                this_rowmax[1] = max(max(own0_lower, own1_lower), max(other0_lower, other1_lower));
            } else {
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                    this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
                }
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
                this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
                this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));
            }

            // rescale previous O for new rowmax
            float rescale[2];
            if constexpr (EXP_APPROX) {
                // -----------------------------------------------------------
                // Stage 4: INTEGER-SNAPPED rowmax. The block max is snapped
                // UP onto the rung grid (the +0.5f guarantees R >= max * k,
                // so n <= 0 still holds element-wise; a uniform rung shift
                // cancels between numerator and denominator). Two payoffs:
                //   * the rescale is exactly 2^(R_old - R_new) from bits —
                //     old O entries match their re-quantization exactly;
                //   * R changes only when the max crosses a rung (~1/k raw
                //     units), so most iterations the whole warp votes "no
                //     change" and SKIPS the 64-FMUL O-rescale loop.
                // -----------------------------------------------------------
                const float R_old0 = rowmax[mma_id_q][0];
                const float R_old1 = rowmax[mma_id_q][1];
                const float R0 = fmaxf(fmaf(this_rowmax[0], softmax_scale, 0.5f) + TA_MAGIC, R_old0);
                const float R1 = fmaxf(fmaf(this_rowmax[1], softmax_scale, 0.5f) + TA_MAGIC, R_old1);
                rowmax[mma_id_q][0] = R0;
                rowmax[mma_id_q][1] = R1;

                const bool rung_change = (R0 != R_old0) || (R1 != R_old1);
                if (__any_sync(0xFFFFFFFFu, rung_change)) {
                    // R_old - R is an exact non-positive integer (unit grid);
                    // re-biasing by TA_MAGIC and flooring guards the
                    // first-block jump from TA_MAGIC_FLOOR (2^-126 ~ 0).
                    const float d0 = fmaxf((R_old0 - R0) + TA_MAGIC, TA_MAGIC_FLOOR);
                    const float d1 = fmaxf((R_old1 - R1) + TA_MAGIC, TA_MAGIC_FLOOR);
                    rescale[0] = ta_pow2_from_bits(__float_as_uint(d0));
                    rescale[1] = ta_pow2_from_bits(__float_as_uint(d1));
                    for (int mma_id_d = 0; mma_id_d < O_TILES; mma_id_d++) {
                        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
                    }
                } else {
                    rescale[0] = 1.0f;
                    rescale[1] = 1.0f;
                }
            } else {
                // new rowmax (raw domain)
                this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
                this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

                rescale[0] = __expf((rowmax[mma_id_q][0] - this_rowmax[0]) * softmax_scale);
                rescale[1] = __expf((rowmax[mma_id_q][1] - this_rowmax[1]) * softmax_scale);
                for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
                    O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
                    O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
                    O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
                }

                // save new rowmax
                rowmax[mma_id_q][0] = this_rowmax[0];
                rowmax[mma_id_q][1] = this_rowmax[1];
            }

            float this_rowsumexp[2] = {};

            if constexpr (EXP_APPROX && MICROBLOCK_P) {
                // -----------------------------------------------------------
                // Microblock-P approx path. Each e4m3 P scale covers 16
                // logical P entries: four MMA_N fragments across one lane
                // pair, separately for the upper/lower row groups.
                //
                // B is snapped onto the same integer log2 rung grid as the
                // running row max R. P codes store 4 * 2^(x - B), while the
                // e4m3 scale stores 448 * 2^(B - R). The 448 range multiplier
                // preserves e4m3 dynamic range and cancels against the
                // denominator tile, exactly like the uniform-scale approx
                // path's packaging constant.
                // -----------------------------------------------------------

                //same scale is used for P@V and P@ones so 1792 factor cancels out
                constexpr float P_SCALE_RANGE = 448.0f;
                static_assert(BLOCK_KV == MMA_K);

                // compute clamped maximum for [a0,...,a15][a32, ... ,a47] for lower and upper rows respectively
                own0_upper = fminf(fmaxf(fmaf(own0_upper, softmax_scale, 0.5f) + TA_MAGIC, TA_MAGIC_FLOOR), rowmax[mma_id_q][0]);
                own0_lower = fminf(fmaxf(fmaf(own0_lower, softmax_scale, 0.5f) + TA_MAGIC, TA_MAGIC_FLOOR), rowmax[mma_id_q][1]);
                own1_upper = fminf(fmaxf(fmaf(own1_upper, softmax_scale, 0.5f) + TA_MAGIC, TA_MAGIC_FLOOR), rowmax[mma_id_q][0]);
                own1_lower = fminf(fmaxf(fmaf(own1_lower, softmax_scale, 0.5f) + TA_MAGIC, TA_MAGIC_FLOOR), rowmax[mma_id_q][1]);
                

                //we get TA_MAGIC - B
                const float c0_upper = TA_MAGIC_X2 - own0_upper;
                const float c0_lower = TA_MAGIC_X2 - own0_lower;
                const float c1_upper = TA_MAGIC_X2 - own1_upper;
                const float c1_lower = TA_MAGIC_X2 - own1_lower;

                for (int mma_id_kv = 0; mma_id_kv < P_SCALE_MMA_N_TILES; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    // TA_MAGIC + (regs[0] * softmax_scale - B sofmax_scale)
                    const float e0 = fmaxf(fmaf(regs[0], softmax_scale, c0_upper), TA_MAGIC_FLOOR);
                    const float e1 = fmaxf(fmaf(regs[1], softmax_scale, c0_upper), TA_MAGIC_FLOOR);
                    const float e2 = fmaxf(fmaf(regs[2], softmax_scale, c0_lower), TA_MAGIC_FLOOR);
                    const float e3 = fmaxf(fmaf(regs[3], softmax_scale, c0_lower), TA_MAGIC_FLOOR);
                    //scale so max is at top of e2m1 exponent range
                    regs[0] = ta_pow2x4_from_bits(__float_as_uint(e0));
                    regs[1] = ta_pow2x4_from_bits(__float_as_uint(e1));
                    regs[2] = ta_pow2x4_from_bits(__float_as_uint(e2));
                    regs[3] = ta_pow2x4_from_bits(__float_as_uint(e3));
                }
                // implements this in two tiles
                for (int mma_id_kv = P_SCALE_MMA_N_TILES; mma_id_kv < 2 * P_SCALE_MMA_N_TILES; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    const float e0 = fmaxf(fmaf(regs[0], softmax_scale, c1_upper), TA_MAGIC_FLOOR);
                    const float e1 = fmaxf(fmaf(regs[1], softmax_scale, c1_upper), TA_MAGIC_FLOOR);
                    const float e2 = fmaxf(fmaf(regs[2], softmax_scale, c1_lower), TA_MAGIC_FLOOR);
                    const float e3 = fmaxf(fmaf(regs[3], softmax_scale, c1_lower), TA_MAGIC_FLOOR);
                    regs[0] = ta_pow2x4_from_bits(__float_as_uint(e0));
                    regs[1] = ta_pow2x4_from_bits(__float_as_uint(e1));
                    regs[2] = ta_pow2x4_from_bits(__float_as_uint(e2));
                    regs[3] = ta_pow2x4_from_bits(__float_as_uint(e3));
                }

                float sf_own0_upper = P_SCALE_RANGE * ta_pow2_from_bits(__float_as_uint(
                    fmaxf((own0_upper - rowmax[mma_id_q][0]) + TA_MAGIC, TA_MAGIC_FLOOR)));
                float sf_own0_lower = P_SCALE_RANGE * ta_pow2_from_bits(__float_as_uint(
                    fmaxf((own0_lower - rowmax[mma_id_q][1]) + TA_MAGIC, TA_MAGIC_FLOOR)));
                float sf_own1_upper = P_SCALE_RANGE * ta_pow2_from_bits(__float_as_uint(
                    fmaxf((own1_upper - rowmax[mma_id_q][0]) + TA_MAGIC, TA_MAGIC_FLOOR)));
                float sf_own1_lower = P_SCALE_RANGE * ta_pow2_from_bits(__float_as_uint(
                    fmaxf((own1_lower - rowmax[mma_id_q][1]) + TA_MAGIC, TA_MAGIC_FLOOR)));
                
                //shuffle 
                // at this point lane 0 and 1 own [s0, s2] in 64 row dimension
                // warp shuffle so that lane 0 [s0, s1, s2, s3]
                const float sf_other0_upper = __shfl_xor_sync(0xFFFFFFFF, sf_own0_upper, 2);
                const float sf_other0_lower = __shfl_xor_sync(0xFFFFFFFF, sf_own0_lower, 2);
                const float sf_other1_upper = __shfl_xor_sync(0xFFFFFFFF, sf_own1_upper, 2);
                const float sf_other1_lower = __shfl_xor_sync(0xFFFFFFFF, sf_own1_lower, 2);

                //mapping back to stable names
                const float sf0_upper = quad_pair_hi ? sf_other0_upper : sf_own0_upper;
                const float sf0_lower = quad_pair_hi ? sf_other0_lower : sf_own0_lower;
                const float sf1_upper = quad_pair_hi ? sf_own0_upper : sf_other0_upper;
                const float sf1_lower = quad_pair_hi ? sf_own0_lower : sf_other0_lower;
                const float sf2_upper = quad_pair_hi ? sf_other1_upper : sf_own1_upper;
                const float sf2_lower = quad_pair_hi ? sf_other1_lower : sf_own1_lower;
                const float sf3_upper = quad_pair_hi ? sf_own1_upper : sf_other1_upper;
                const float sf3_lower = quad_pair_hi ? sf_own1_lower : sf_other1_lower;

                const uint32_t sfP_upper_packed = ta_cvt_4xf32_to_e4m3_packed(
                    sf1_upper, sf0_upper, sf3_upper, sf2_upper);
                const uint32_t sfP_lower_packed = ta_cvt_4xf32_to_e4m3_packed(
                    sf1_lower, sf0_lower, sf3_lower, sf2_lower);
                S_fp4_s_rmem[mma_id_q][0] =
                    (lane_id % 4 == 0) ? sfP_upper_packed : sfP_lower_packed;
            } else if constexpr (EXP_APPROX) {
                // -----------------------------------------------------------
                // Stage 1 + 3: one FFMA does scale * s in log2 domain AND
                // the round-to-integer (the snapped rung + magic constant
                // ride in the per-row addend c = 2*MAGIC - R, an exact
                // unit-grid FADD); one FMNMX guards the bottom (-INF mask
                // values, NaN, exponent-wrap safety n >= -126). The biased
                // weight 4 * 2^n is then one LEA on the bits — no MUFU.EX2
                // and no x6 range loop. Weights overwrite S_rmem in place so
                // the cvt pack loop below is shared with the baseline
                // (identical register profile and nibble order).
                // -----------------------------------------------------------
                const float c0 = TA_MAGIC_X2 - rowmax[mma_id_q][0];
                const float c1 = TA_MAGIC_X2 - rowmax[mma_id_q][1];

                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    const float r0 = fmaxf(fmaf(regs[0], softmax_scale, c0), TA_MAGIC_FLOOR);
                    const float r1 = fmaxf(fmaf(regs[1], softmax_scale, c0), TA_MAGIC_FLOOR);
                    const float r2 = fmaxf(fmaf(regs[2], softmax_scale, c1), TA_MAGIC_FLOOR);
                    const float r3 = fmaxf(fmaf(regs[3], softmax_scale, c1), TA_MAGIC_FLOOR);

                    // No FP rowsum here: the denominator is accumulated by
                    // the PV mma against the all-ones tile (Stage 5).
                    regs[0] = ta_pow2x4_from_bits(__float_as_uint(r0));
                    regs[1] = ta_pow2x4_from_bits(__float_as_uint(r1));
                    regs[2] = ta_pow2x4_from_bits(__float_as_uint(r2));
                    regs[3] = ta_pow2x4_from_bits(__float_as_uint(r3));
                }
            } else {
                // -----------------------------------------------------------
                // Baseline path: __expf, with the scale fused into the exp
                // argument (the raw-domain rowmax makes both paths share
                // bookkeeping) and the constant x6 range multiply (Stage 2).
                // -----------------------------------------------------------
                const float c0 = -rowmax[mma_id_q][0] * softmax_scale;
                const float c1 = -rowmax[mma_id_q][1] * softmax_scale;

                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    regs[0] = __expf(fmaf(regs[0], softmax_scale, c0));
                    regs[1] = __expf(fmaf(regs[1], softmax_scale, c0));
                    regs[2] = __expf(fmaf(regs[2], softmax_scale, c1));
                    regs[3] = __expf(fmaf(regs[3], softmax_scale, c1));

                    this_rowsumexp[0] += regs[0] + regs[1];
                    this_rowsumexp[1] += regs[2] + regs[3];
                }

                // Stage 2: inv == FP4_MAX * exp(0) == 6.0 exactly; the old
                // per-block loop recomputed this constant with 16 MUFU ops.
                constexpr float FP4_MAX = 6.0f;
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
                    float *regs = S_rmem[mma_id_q][mma_id_kv];
                    regs[0] *= FP4_MAX;
                    regs[1] *= FP4_MAX;
                    regs[2] *= FP4_MAX;
                    regs[3] *= FP4_MAX;
                }
            }

            // cvt-based e2m1 quantization, shared by both paths. Approx
            // inputs are exact powers of two in [2^-124, 4]: cvt.rn lands
            // them on codes {6, 4, 2, 1}, and everything at or below 0.25
            // rounds to 0 (0.25 is a tie, to even).
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

            // Baseline only: butterfly-reduce the FP rowsum and fold it into
            // the running total. The approx denominator instead rides the PV
            // mma (Stage 5): no shuffles, no per-iteration rowsum update.
            if constexpr (!EXP_APPROX) {
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

                rowsum[mma_id_q][0] = rowsum[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
                rowsum[mma_id_q][1] = rowsum[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
            }

            if constexpr (!EXP_APPROX || !MICROBLOCK_P) {
                // -----------------------------------------------------------
                // Stage 2: P's e4m3 block scale is the CONSTANT 448 (== the
                // old FP4_RANGE/FP4_MAX * exp(0)). 448 in e4m3 is the byte
                // 0x7E; replicated four times. The old code burned 16 exp(0)
                // MUFU ops plus an e4m3 cvt per row-group per iteration to
                // compute this. A fully-masked row contributes zero through
                // its codes (all 0), so the constant scale is safe there too.
                // -----------------------------------------------------------
                for (int mma_sc_id = 0; mma_sc_id < BLOCK_KV / MMA_K; mma_sc_id++)
                    S_fp4_s_rmem[mma_id_q][mma_sc_id] = 0x7E7E7E7Eu;
            }
        }

        // V layout in smem: [HEAD_DIM, BLOCK_KV/2], stride = BLOCK_KV/2 bytes.
        // ldmatrix_x4: lanes 0-15 address tile mma_id_d, lanes 16-31 address tile mma_id_d+1.
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

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

        // MMA O += P @ V. Approx (Stage 5): tile HEAD_DIM/MMA_N is P @ ones
        // (e2m1 code 2 == 1.0 in every nibble, e4m3 scale 1.0 == 0x38) — the
        // mma reduces the quantized weights along kv, so every column of
        // that tile holds the row's denominator. No smem, ldmatrix, or
        // shuffles back this tile; the operands are compile-time constants.
        uint32_t ones_b[2] = {0x22222222u, 0x22222222u};
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
            for (int mma_id_d = 0; mma_id_d < O_TILES; mma_id_d++)
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                    const bool sum_tile = EXP_APPROX && mma_id_d == HEAD_DIM / MMA_N;
                    ta_mma_m16n8k64_nvfp4(
                        S_fp4_rmem[mma_id_q][mma_id_kv],
                        sum_tile ? ones_b : V_rmem[mma_id_kv][mma_id_d],
                        S_fp4_s_rmem[mma_id_q][mma_id_kv],
                        sum_tile ? 0x38383838u : sfV_rmem[mma_id_kv][mma_id_d],
                        O_rmem[mma_id_q][mma_id_d]);
                }

        K += BLOCK_KV * HEAD_DIM_2;
        S_K += BLOCK_KV * SCALE_DIM;

        V  += BLOCK_KV / 2;
        S_V += BLOCK_KV / 16;
    }

    // Epilogue. Approx: numerator and denominator both came through the PV
    // mma with identical P codes and the 448 scale, so EVERY packaging
    // constant cancels — norm is simply 1/sum_tile (cols of the ones tile
    // all hold the row's denominator; [0]/[2] are this thread's rows).
    // Baseline: stored exp * 6 with sf = 448 while rowsum accumulated plain
    // exp -> divide by 448*6.
    constexpr float P_DEQUANT_INV = 1.0f / (448.0f * 6.0f);

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < HEAD_DIM / MMA_N; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];

            float norm0, norm1;
            if constexpr (EXP_APPROX) {
                norm0 = 1.0f / O_rmem[mma_id_q][HEAD_DIM / MMA_N][0];
                norm1 = 1.0f / O_rmem[mma_id_q][HEAD_DIM / MMA_N][2];
            } else {
                norm0 = P_DEQUANT_INV / rowsum[mma_id_q][0];
                norm1 = P_DEQUANT_INV / rowsum[mma_id_q][1];
            }

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

template<typename T, bool CAUSAL, bool EXP_APPROX, bool MICROBLOCK_P, int HEAD_DIM>
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

    auto kernel = fp4_attention_kernel<T, CAUSAL, EXP_APPROX, MICROBLOCK_P,
                                       BLOCK_Q, BLOCK_KV, HEAD_DIM,
                                       HEAD_DIM_2, SCALE_DIM,
                                       NUM_WARPS, WARP_Q>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<num_blocks, TB_SIZE, smem_size>>>(
        Q, K, V, S_Q, S_K, S_V, O,
        bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads);
}

template<typename T, bool CAUSAL, bool EXP_APPROX, bool MICROBLOCK_P>
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
        launch_fp4_attention<T, CAUSAL, EXP_APPROX, MICROBLOCK_P, 64>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads);
    } else {
        launch_fp4_attention<T, CAUSAL, EXP_APPROX, MICROBLOCK_P, 128>(
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
    int head_dim,
    bool exp_approx,
    bool microblock_p) {
    auto Q = reinterpret_cast<const __nv_fp4x2_e2m1*>(Q_raw);
    auto K = reinterpret_cast<const __nv_fp4x2_e2m1*>(K_raw);
    auto V = reinterpret_cast<const __nv_fp4x2_e2m1*>(V_raw);
    auto S_Q = reinterpret_cast<const __nv_fp8_e4m3*>(S_Q_raw);
    auto S_K = reinterpret_cast<const __nv_fp8_e4m3*>(S_K_raw);
    auto S_V = reinterpret_cast<const __nv_fp8_e4m3*>(S_V_raw);
    auto O = reinterpret_cast<T*>(O_raw);

    if (exp_approx) {
        if (microblock_p) {
            dispatch_fp4_attention<T, CAUSAL, true, true>(
                Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
                kv_capacity, num_q_heads, num_kv_heads, head_dim);
        } else {
            dispatch_fp4_attention<T, CAUSAL, true, false>(
                Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
                kv_capacity, num_q_heads, num_kv_heads, head_dim);
        }
    } else {
        dispatch_fp4_attention<T, CAUSAL, false, false>(
            Q, K, V, S_Q, S_K, S_V, O, bs, q_len, kv_len,
            kv_capacity, num_q_heads, num_kv_heads, head_dim);
    }
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
    bool is_bf16,
    bool exp_approx,
    bool microblock_p) {
    if (is_bf16) {
        fp4_attention_nvfp4_typed<__nv_bfloat16, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads,
            head_dim, exp_approx, microblock_p);
    } else {
        fp4_attention_nvfp4_typed<half, true>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads,
            head_dim, exp_approx, microblock_p);
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
    bool is_bf16,
    bool exp_approx,
    bool microblock_p) {
    if (is_bf16) {
        fp4_attention_nvfp4_typed<__nv_bfloat16, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads,
            head_dim, exp_approx, microblock_p);
    } else {
        fp4_attention_nvfp4_typed<half, false>(
            Q_raw, K_raw, V_raw, S_Q_raw, S_K_raw, S_V_raw, O_raw,
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads,
            head_dim, exp_approx, microblock_p);
    }
}
