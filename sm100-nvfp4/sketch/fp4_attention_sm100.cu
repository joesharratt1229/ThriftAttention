#include <cstdint>
#include <cstdio>
#include <float.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int CTA_GROUP = 1;

constexpr uint64_t EVICT_NORMAL = 0x1000000000000000;
constexpr float TA_MAGIC = 12582912.0f;
constexpr float TA_MAGIC_FLOOR = TA_MAGIC - 126.0f;
constexpr float TA_MAGIC_X2 = 25165824.0f;
// Lazy rescale (FA4-style rescale_threshold): the row max used for P/sf
// normalisation only advances when a tile max exceeds it by more than
// LAZY_RESCALE_T log2 units, so acc_scale is exactly 1.0 (no O rescale) on
// almost every iteration.  sf_p headroom for the lag is reserved by shrinking
// the sf range: (448 / 2^T) * 2^lag <= 448 stays exactly representable in
// e4m3, and the constant factor cancels in the row-sum normalisation.
constexpr float LAZY_RESCALE_T = 4.0f;
constexpr float P_SCALE_RANGE = 448.0f / 16.0f;

constexpr int FA4_NUM_WARPS = 16;
constexpr int FA4_THREADS = FA4_NUM_WARPS * WARP_SIZE;
constexpr int FA4_HEAD_DIM = 128;
constexpr int FA4_Q_STAGES = 1;
constexpr int FA4_Q_STAGE_ROWS = 128;
constexpr int FA4_Q_TILE_ROWS = FA4_Q_STAGES * FA4_Q_STAGE_ROWS;
constexpr int FA4_KV_TILE = 128;
constexpr int FA4_KV_CHUNKS = FA4_KV_TILE / 64;
constexpr int FA4_MMA_M = 128;
constexpr int FA4_MMA_K = 64;
constexpr int FA4_QK_N = 128;
constexpr int FA4_PV_N = 128;
constexpr int FA4_QK_K_ITERS = FA4_HEAD_DIM / FA4_MMA_K;
constexpr int FA4_PV_K_ITERS = FA4_KV_CHUNKS;
constexpr int FA4_S_BUFS = 2;
constexpr int FA4_NUM_SLOTS = 16;
constexpr int FA4_NUM_KV_PAIRS = FA4_NUM_SLOTS / 2;

constexpr int SF_ATOM_BYTES = 512;
constexpr int Q_DATA_BYTES = FA4_Q_STAGE_ROWS * (FA4_HEAD_DIM / 2);
constexpr int Q_SF_BYTES = FA4_QK_K_ITERS * SF_ATOM_BYTES;
constexpr int KV_DATA_BYTES = FA4_KV_TILE * (FA4_HEAD_DIM / 2);
constexpr int KV_CHUNK_BYTES = KV_DATA_BYTES / FA4_KV_CHUNKS;
constexpr int K_SF_BYTES = FA4_QK_K_ITERS * SF_ATOM_BYTES;
constexpr int V_SF_BYTES = FA4_PV_K_ITERS * SF_ATOM_BYTES;
constexpr int KV_SF_SLOT_BYTES = 1024;
constexpr int KV_SLOT_BYTES = KV_DATA_BYTES + KV_SF_SLOT_BYTES;
constexpr int O_STAGE_BYTES = FA4_Q_STAGE_ROWS * FA4_HEAD_DIM * sizeof(__nv_bfloat16);
// Pairwise softmax handoff: partner half-row maxima + lazily updated row max.
constexpr int STATS_BYTES = 2 * FA4_Q_STAGE_ROWS * sizeof(float);

constexpr int SMEM_Q = 0;
constexpr int SMEM_SF_Q = SMEM_Q + Q_DATA_BYTES;
constexpr int SMEM_KV = SMEM_SF_Q + Q_SF_BYTES;
constexpr int SMEM_O = SMEM_KV + FA4_NUM_SLOTS * KV_SLOT_BYTES;
constexpr int SMEM_STATS = SMEM_O + O_STAGE_BYTES;
constexpr int SMEM_ONES = SMEM_STATS + STATS_BYTES;
constexpr int SMEM_ONES_SF = SMEM_ONES + 512;
constexpr int SMEM_BYTES = SMEM_ONES_SF + 512;

// tmem: O accumulator, two full-width S buffers (ping-pong across KV
// iterations), row-sum accumulator, and the scale-factor planes.  P and
// sf_p for chunk c live INSIDE the consumed S buffer at cols c*64 + 0..11.
constexpr int TMEM_O = 0;
constexpr int TMEM_S_A = 128;
constexpr int TMEM_S_B = 256;
constexpr int TMEM_OSUM = 384;
constexpr int TMEM_SFQ = 392;
constexpr int TMEM_SFK0 = 400;
constexpr int TMEM_SFK1 = 408;
constexpr int TMEM_SFV0 = 416;
constexpr int TMEM_SFV1 = 424;
constexpr int TMEM_SFONES = 432;
constexpr int TMEM_COLS = 512;
constexpr int FA4_SUM_N = 8;

enum class MbarId : int {
    QFull = 0,
    QEmpty = QFull + 1,
    KVFull = QEmpty + 1,
    KVEmpty = KVFull + FA4_NUM_SLOTS,
    SFull = KVEmpty + FA4_NUM_SLOTS,
    SEmpty = SFull + FA4_S_BUFS,
    P0Ready = SEmpty + FA4_S_BUFS,
    P1Ready = P0Ready + FA4_S_BUFS,
    OFull = P1Ready + FA4_S_BUFS,
    OEpi = OFull + 1,
    OEmpty = OEpi + 1,
    OStore = OEmpty + 1,
    TmemDealloc = OStore + 1,
    Count = TmemDealloc + 1
};

__device__ __forceinline__
int mbar_offset(MbarId id, int index = 0)
{
    return (static_cast<int>(id) + index) * 8;
}

__device__ __forceinline__
uint32_t elect_sync()
{
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
}

template <int N>
__device__ __forceinline__
void setmaxnreg_inc()
{
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N));
}

template <int N>
__device__ __forceinline__
void setmaxnreg_dec()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N));
}

__device__ __forceinline__
uint64_t desc_encode(uint64_t x)
{
    return (x & 0x3ffffULL) >> 4ULL;
}

__device__ __forceinline__
uint64_t make_desc_data_64b(uint32_t addr)
{
    constexpr int SBO = 8 * 64;
    return desc_encode(addr)
        | (desc_encode(SBO) << 32ULL)
        | (1ULL << 46ULL)
        | (4ULL << 61ULL);
}

__device__ __forceinline__
uint64_t make_desc_data_32b(uint32_t addr)
{
    constexpr int SBO = 8 * 32;
    return desc_encode(addr)
        | (desc_encode(SBO) << 32ULL)
        | (1ULL << 46ULL)
        | (6ULL << 61ULL);
}

__device__ __forceinline__
uint64_t make_desc_sf(uint32_t addr)
{
    constexpr int SBO = 8 * 16;
    return desc_encode(addr)
        | (desc_encode(SBO) << 32ULL)
        | (1ULL << 46ULL);
}

__device__ __forceinline__
void mbarrier_init(uint32_t mbar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar), "r"(count));
}

__device__ __forceinline__
void mbarrier_wait(uint32_t mbar, int phase)
{
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar), "r"(phase), "r"(ticks));
}

__device__ __forceinline__
void mbarrier_arrive(uint32_t mbar)
{
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar)
        : "memory");
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint32_t mbar, uint32_t bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar), "r"(bytes)
        : "memory");
}

__device__ __forceinline__
void tcgen05_alloc(uint32_t smem_addr, uint32_t cols)
{
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(cols));
}

__device__ __forceinline__
void tcgen05_dealloc(uint32_t tmem_addr, uint32_t cols)
{
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_addr), "r"(cols));
}

__device__ __forceinline__
void tcgen05_relinquish_alloc_permit()
{
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_commit(uint32_t mbar)
{
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mbar)
        : "memory");
}

__device__ __forceinline__
void tcgen05_cp(uint32_t taddr, uint64_t smem_desc)
{
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(smem_desc));
}

__device__ __forceinline__
void tcgen05_wait_st()
{
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__
void tcgen05_fence_after()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__
void tcgen05_mma_nvfp4(uint32_t tmem,
                       uint64_t a_desc,
                       uint64_t b_desc,
                       uint32_t i_desc,
                       uint32_t sf_a_tmem,
                       uint32_t sf_b_tmem,
                       int enable_input_d)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(sf_a_tmem), "r"(sf_b_tmem), "r"(enable_input_d));
}

__device__ __forceinline__
void tcgen05_mma_nvfp4_pv(uint32_t tmem,
                          uint32_t a_tmem,
                          uint64_t b_desc,
                          uint32_t i_desc,
                          uint32_t sf_a_tmem,
                          uint32_t sf_b_tmem,
                          int enable_input_d)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], [%1], %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(tmem), "r"(a_tmem), "l"(b_desc), "r"(i_desc),
           "r"(sf_a_tmem), "r"(sf_b_tmem), "r"(enable_input_d));
}

__device__ __forceinline__
void tma_gmem2smem(uint32_t dst, const void* src, uint32_t size, uint32_t mbar, uint64_t cache_policy = EVICT_NORMAL)
{
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;"
        :: "r"(dst), "l"(src), "r"(size), "r"(mbar), "l"(cache_policy)
        : "memory");
}

__device__ __forceinline__
void tma_2d_gmem2smem(uint32_t dst, const void* tmap, int x, int y, uint32_t mbar, uint64_t cache_policy = EVICT_NORMAL)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar), "l"(cache_policy)
        : "memory");
}

__device__ __forceinline__
void tma_2d_smem2gmem(const void* tmap, int x, int y, uint32_t src, uint64_t cache_policy = EVICT_NORMAL)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint "
        "[%0, {%1, %2}], [%3], %4;"
        :: "l"(tmap), "r"(x), "r"(y), "r"(src), "l"(cache_policy)
        : "memory");
}

__device__ __forceinline__
void cp_async_bulk_commit_group()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__
void cp_async_bulk_wait_group_read_0()
{
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__
void tcgen05_ld_32x32bx32(float* dst, uint32_t taddr)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_ld_32x32bx64(float* dst, uint32_t taddr)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31]),
          "=f"(dst[32]), "=f"(dst[33]), "=f"(dst[34]), "=f"(dst[35]),
          "=f"(dst[36]), "=f"(dst[37]), "=f"(dst[38]), "=f"(dst[39]),
          "=f"(dst[40]), "=f"(dst[41]), "=f"(dst[42]), "=f"(dst[43]),
          "=f"(dst[44]), "=f"(dst[45]), "=f"(dst[46]), "=f"(dst[47]),
          "=f"(dst[48]), "=f"(dst[49]), "=f"(dst[50]), "=f"(dst[51]),
          "=f"(dst[52]), "=f"(dst[53]), "=f"(dst[54]), "=f"(dst[55]),
          "=f"(dst[56]), "=f"(dst[57]), "=f"(dst[58]), "=f"(dst[59]),
          "=f"(dst[60]), "=f"(dst[61]), "=f"(dst[62]), "=f"(dst[63])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void named_barrier_sync(int id, int count)
{
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count));
}

__device__ __forceinline__
void tcgen05_ld_32x32bx1(float* dst, uint32_t taddr)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
        : "=f"(dst[0])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_ld_32x32bx8(float* dst, uint32_t taddr)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_st_32x32bx1(uint32_t taddr, uint32_t src)
{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%1], {%0};"
        :: "r"(src), "r"(taddr)
        : "memory");
}

__device__ __forceinline__
void tcgen05_st_32x32bx8(uint32_t taddr, const uint32_t* src)
{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], "
        "{ %0, %1, %2, %3, %4, %5, %6, %7};"
        :: "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]),
           "r"(taddr)
        : "memory");
}

__device__ __forceinline__
void tcgen05_st_32x32bx32(uint32_t taddr, const float* src)
{
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31};"
        :: "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]),
           "r"(taddr)
        : "memory");
}

__device__ __forceinline__
float ta_pow2_from_bits(uint32_t br)
{
    return __uint_as_float((br << 23) + 0x3f800000u);
}

__device__ __forceinline__
float ta_pow2x4_from_bits(uint32_t br)
{
    return __uint_as_float((br << 23) + 0x40800000u);
}

__device__ __forceinline__
float ta_fmax3(float a, float b, float c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    float d;
    asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
#else
    return fmaxf(fmaxf(a, b), c);
#endif
}

__device__ __forceinline__
float ta_reduce_max_16(const float* x)
{
    const float m0 = ta_fmax3(x[0], x[1], x[2]);
    const float m1 = ta_fmax3(x[3], x[4], x[5]);
    const float m2 = ta_fmax3(x[6], x[7], x[8]);
    const float m3 = ta_fmax3(x[9], x[10], x[11]);
    const float m4 = ta_fmax3(x[12], x[13], x[14]);
    return ta_fmax3(ta_fmax3(m0, m1, m2), ta_fmax3(m3, m4, x[15]), -FLT_MAX);
}

__device__ __forceinline__
float ta_score_to_p(float score, float softmax_scale_log2, float block_addend)
{
    const float rounded = fmaxf(fmaf(score, softmax_scale_log2, block_addend), TA_MAGIC_FLOOR);
    return ta_pow2x4_from_bits(__float_as_uint(rounded));
}

// Blackwell paired-lane fp32 ops: one instruction per two elements on the
// FMA pipe.  Per-lane rounding is identical to the scalar ops.
__device__ __forceinline__
void ta_fma2(float& d0, float& d1, float a0, float a1, float b, float c)
{
    asm("{\n\t"
        ".reg .b64 ra, rb, rc, rd;\n\t"
        "mov.b64 ra, {%2, %3};\n\t"
        "mov.b64 rb, {%4, %4};\n\t"
        "mov.b64 rc, {%5, %5};\n\t"
        "fma.rn.f32x2 rd, ra, rb, rc;\n\t"
        "mov.b64 {%0, %1}, rd;\n\t"
        "}"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b), "f"(c));
}

__device__ __forceinline__
void ta_mul2(float& d0, float& d1, float a0, float a1, float b)
{
    asm("{\n\t"
        ".reg .b64 ra, rb, rd;\n\t"
        "mov.b64 ra, {%2, %3};\n\t"
        "mov.b64 rb, {%4, %4};\n\t"
        "mul.f32x2 rd, ra, rb;\n\t"
        "mov.b64 {%0, %1}, rd;\n\t"
        "}"
        : "=f"(d0), "=f"(d1)
        : "f"(a0), "f"(a1), "f"(b));
}

__device__ __forceinline__
void ta_score_to_p2(float& p0, float& p1, float s0, float s1,
                    float softmax_scale_log2, float block_addend)
{
    float r0, r1;
    ta_fma2(r0, r1, s0, s1, softmax_scale_log2, block_addend);
    r0 = fmaxf(r0, TA_MAGIC_FLOOR);
    r1 = fmaxf(r1, TA_MAGIC_FLOOR);
    p0 = ta_pow2x4_from_bits(__float_as_uint(r0));
    p1 = ta_pow2x4_from_bits(__float_as_uint(r1));
}

__device__ __forceinline__
uint32_t ta_cvt_8xf32_to_e2m1_packed(float f0, float f1, float f2, float f3,
                                     float f4, float f5, float f6, float f7)
{
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

__device__ __forceinline__
uint32_t ta_cvt_4xf32_to_e4m3_packed(float f0, float f1, float f2, float f3)
{
    uint32_t packed;
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, %1, %2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, %3, %4;\n\t"
        "mov.b32 %0, {lo, hi};\n\t"
        "}"
        : "=r"(packed)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3));
    return packed;
}

__device__ __forceinline__
uint32_t fa4_qk_idesc()
{
    return (1U << 7U)
        |  (1U << 10U)
        |  ((uint32_t)FA4_QK_N >> 3U << 17U)
        |  ((uint32_t)FA4_MMA_M >> 7U << 27U);
}

__device__ __forceinline__
uint32_t fa4_pv_idesc()
{
    return (1U << 7U)
        |  (1U << 10U)
        |  ((uint32_t)FA4_PV_N >> 3U << 17U)
        |  ((uint32_t)FA4_MMA_M >> 7U << 27U);
}

__device__ __forceinline__
uint32_t fa4_sum_idesc()
{
    return (1U << 7U)
        |  (1U << 10U)
        |  ((uint32_t)FA4_SUM_N >> 3U << 17U)
        |  ((uint32_t)FA4_MMA_M >> 7U << 27U);
}

extern "C" __global__
__launch_bounds__(FA4_THREADS, 1)
void nvfp4_sm100_attention_kernel(const __grid_constant__ CUtensorMap q_tmap,
                                  const __grid_constant__ CUtensorMap k_tmap,
                                  const __grid_constant__ CUtensorMap v_tmap,
                                  const __grid_constant__ CUtensorMap o_tmap,
                                  const uint8_t* __restrict__ sf_q_atoms,
                                  const uint8_t* __restrict__ sf_k_atoms,
                                  const uint8_t* __restrict__ sf_v_atoms,
                                  int q_len,
                                  int kv_len,
                                  int num_q_heads,
                                  int num_kv_heads,
                                  float softmax_scale_log2,
                                  float v_descale,
                                  int num_tiles)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int q_per_kv = num_q_heads / num_kv_heads;
    const int kv_iters = kv_len / FA4_KV_TILE;

    // wg0: softmax owners (chunk 0, cols 0-63, stats + rescale + epilogue);
    // wg1: softmax partners (chunk 1, cols 64-127) -- warps w and w+4 share
    // tmem sub-partition w%4, so both can serve the same 32 rows;
    // wg2: unused (exits); wg3: mma / epilogue-store / load.
    if (warp_id < 8) {
        setmaxnreg_inc<176>();
    } else if (warp_id < 12) {
        setmaxnreg_dec<80>();
    } else {
        setmaxnreg_dec<80>();
    }

    extern __shared__ __align__(1024) char smem_storage[];
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_storage));
    // Pairwise handoff: partner's half-row max in, owner's lazily updated
    // row max out.  Indexed by row (0..127).
    float* hmax1 = reinterpret_cast<float*>(smem_storage + SMEM_STATS);
    float* hm_used = hmax1 + FA4_Q_STAGE_ROWS;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbars[static_cast<int>(MbarId::Count)];
    __shared__ uint32_t tmem_addr_storage;

    const uint32_t mbar_base = static_cast<uint32_t>(__cvta_generic_to_shared(mbars));
    const uint32_t tmem_addr_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_storage));

    if (warp_id == 0 && elect_sync()) {
        mbarrier_init(mbar_base + mbar_offset(MbarId::QFull), 1);
        mbarrier_init(mbar_base + mbar_offset(MbarId::QEmpty), 1);
        for (int i = 0; i < FA4_NUM_SLOTS; i++) {
            mbarrier_init(mbar_base + mbar_offset(MbarId::KVFull, i), 1);
            mbarrier_init(mbar_base + mbar_offset(MbarId::KVEmpty, i), 1);
        }
        for (int i = 0; i < FA4_S_BUFS; i++) {
            mbarrier_init(mbar_base + mbar_offset(MbarId::SFull, i), 1);
            mbarrier_init(mbar_base + mbar_offset(MbarId::SEmpty, i), 1);
            // Owners: P chunk 0 stored and any O rescale finished.
            mbarrier_init(mbar_base + mbar_offset(MbarId::P0Ready, i), FA4_Q_STAGE_ROWS);
            // Partners' P chunk 1 plus the owners' rescale-done arrivals.
            mbarrier_init(mbar_base + mbar_offset(MbarId::P1Ready, i), 2 * FA4_Q_STAGE_ROWS);
        }
        mbarrier_init(mbar_base + mbar_offset(MbarId::OFull), 1);
        mbarrier_init(mbar_base + mbar_offset(MbarId::OEpi), FA4_Q_STAGE_ROWS);
        mbarrier_init(mbar_base + mbar_offset(MbarId::OEmpty), FA4_Q_STAGE_ROWS);
        mbarrier_init(mbar_base + mbar_offset(MbarId::OStore), 1);
        mbarrier_init(mbar_base + mbar_offset(MbarId::TmemDealloc), FA4_Q_STAGE_ROWS);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    if (warp_id == 12) {
        tcgen05_alloc(tmem_addr_smem, TMEM_COLS);
        tcgen05_relinquish_alloc_permit();
    }

    if (warp_id == 13) {
        uint32_t* ones_data = reinterpret_cast<uint32_t*>(smem_storage + SMEM_ONES);
        uint32_t* ones_sf = reinterpret_cast<uint32_t*>(smem_storage + SMEM_ONES_SF);
        for (int i = lane_id; i < 128; i += WARP_SIZE) {
            ones_data[i] = 0x22222222u;  // fp4 e2m1 1.0 pairs
            ones_sf[i] = 0x38383838u;    // e4m3 1.0
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }

    __syncthreads();

    if (warp_id == 15 || (warp_id >= 8 && warp_id < 12)) {
        return;
    }

    const uint32_t tmem_base = tmem_addr_storage;
    const int q_blocks_128 = q_len / FA4_Q_STAGE_ROWS;
    const int kv_tiles = kv_len / FA4_KV_TILE;
    const int kv_blocks_64 = kv_len / 64;

    // Persistent CTAs: tile coordinates are per-role state (each warp holds
    // its own copy in registers and re-derives them per tile as needed).
    int q_block, q_head, batch, kv_head;
    int q_row_base, q_global_row_base, kv_global_row_base, v_global_row_base;
    int q_sf_head_base, k_sf_head_base, v_sf_head_base;
    auto set_tile = [&](int gtile) {
        q_block = gtile % q_blocks_128;
        q_head = (gtile / q_blocks_128) % num_q_heads;
        batch = gtile / (q_blocks_128 * num_q_heads);
        kv_head = q_head / q_per_kv;
        q_row_base = q_block * FA4_Q_TILE_ROWS;
        q_global_row_base = (batch * num_q_heads + q_head) * q_len + q_row_base;
        kv_global_row_base = (batch * num_kv_heads + kv_head) * kv_len;
        v_global_row_base = (batch * num_kv_heads + kv_head) * FA4_HEAD_DIM;
        q_sf_head_base = (batch * num_q_heads + q_head) * q_blocks_128;
        k_sf_head_base = (batch * num_kv_heads + kv_head) * kv_tiles;
        v_sf_head_base = (batch * num_kv_heads + kv_head) * kv_blocks_64;
    };
    set_tile(blockIdx.x);

    auto q_smem = [&]() -> uint32_t { return smem + SMEM_Q; };
    auto sf_q_smem = [&]() -> uint32_t { return smem + SMEM_SF_Q; };
    auto kv_slot_smem = [&](int slot) -> uint32_t {
        return smem + SMEM_KV + slot * KV_SLOT_BYTES;
    };
    auto kv_slot_sf_smem = [&](int slot) -> uint32_t {
        return kv_slot_smem(slot) + KV_DATA_BYTES;
    };
    auto o_stage_smem = [&]() -> uint32_t { return smem + SMEM_O; };
    auto o_stage_ptr = [&]() -> __nv_bfloat16* {
        return reinterpret_cast<__nv_bfloat16*>(smem_storage + SMEM_O);
    };
    auto s_tmem = [&](int buf) -> uint32_t {
        return tmem_base + (buf == 0 ? TMEM_S_A : TMEM_S_B);
    };
    auto o_tmem = [&]() -> uint32_t { return tmem_base + TMEM_O; };
    auto o_sum_tmem = [&]() -> uint32_t { return tmem_base + TMEM_OSUM; };
    // P chunk c and its sf overwrite consumed columns of the live S buffer.
    auto p_tmem = [&](int buf, int chunk) -> uint32_t {
        return s_tmem(buf) + chunk * 64;
    };
    auto sf_p_tmem = [&](int buf, int chunk) -> uint32_t {
        return s_tmem(buf) + chunk * 64 + 8;
    };
    auto sf_k_tmem = [&](int buffer) -> uint32_t {
        return tmem_base + (buffer == 0 ? TMEM_SFK0 : TMEM_SFK1);
    };
    auto sf_v_tmem = [&](int buffer) -> uint32_t {
        return tmem_base + (buffer == 0 ? TMEM_SFV0 : TMEM_SFV1);
    };
    auto q_sf_src = [&]() -> const uint8_t* {
        return sf_q_atoms + ((q_sf_head_base + q_block) * FA4_QK_K_ITERS) * SF_ATOM_BYTES;
    };
    auto k_sf_src = [&](int kv_iter) -> const uint8_t* {
        return sf_k_atoms + ((k_sf_head_base + kv_iter) * FA4_QK_K_ITERS) * SF_ATOM_BYTES;
    };
    auto v_sf_src = [&](int kv_iter) -> const uint8_t* {
        return sf_v_atoms + ((v_sf_head_base + kv_iter * FA4_PV_K_ITERS)) * SF_ATOM_BYTES;
    };

    auto issue_q = [&]() {
        const uint32_t mbar = mbar_base + mbar_offset(MbarId::QFull);
        tma_2d_gmem2smem(q_smem(), &q_tmap, 0, q_global_row_base, mbar);
        tma_gmem2smem(sf_q_smem(), q_sf_src(), Q_SF_BYTES, mbar);
        mbarrier_arrive_expect_tx(mbar, Q_DATA_BYTES + Q_SF_BYTES);
    };

    auto issue_k = [&](int slot, int g, int kv_iter) {
        const int full_phase = (g / FA4_NUM_KV_PAIRS) & 1;
        mbarrier_wait(mbar_base + mbar_offset(MbarId::KVEmpty, slot), full_phase ^ 1);

        const uint32_t mbar = mbar_base + mbar_offset(MbarId::KVFull, slot);
        const int kv_row = kv_global_row_base + kv_iter * FA4_KV_TILE;
        tma_2d_gmem2smem(kv_slot_smem(slot), &k_tmap, 0, kv_row, mbar);
        tma_gmem2smem(kv_slot_sf_smem(slot), k_sf_src(kv_iter), K_SF_BYTES, mbar);
        mbarrier_arrive_expect_tx(mbar, KV_DATA_BYTES + K_SF_BYTES);
    };

    auto issue_v = [&](int slot, int g, int kv_iter) {
        const int full_phase = (g / FA4_NUM_KV_PAIRS) & 1;
        mbarrier_wait(mbar_base + mbar_offset(MbarId::KVEmpty, slot), full_phase ^ 1);

        const uint32_t mbar = mbar_base + mbar_offset(MbarId::KVFull, slot);
        // The 32B-swizzled V tmap box is one 64-kv-wide chunk; issue both.
        #pragma unroll
        for (int c = 0; c < FA4_KV_CHUNKS; c++) {
            const int kv_col = kv_iter * FA4_KV_TILE + c * 64;
            tma_2d_gmem2smem(kv_slot_smem(slot) + c * KV_CHUNK_BYTES, &v_tmap,
                             kv_col, v_global_row_base, mbar);
        }
        tma_gmem2smem(kv_slot_sf_smem(slot), v_sf_src(kv_iter), V_SF_BYTES, mbar);
        mbarrier_arrive_expect_tx(mbar, KV_DATA_BYTES + V_SF_BYTES);
    };

    auto copy_sf_q_to_tmem = [&]() {
        const uint64_t sf_desc = make_desc_sf(sf_q_smem());
        #pragma unroll
        for (int d = 0; d < FA4_QK_K_ITERS; d++) {
            tcgen05_cp(tmem_base + TMEM_SFQ + d * 4, sf_desc + (uint64_t)d * (SF_ATOM_BYTES >> 4));
        }
    };

    auto copy_sf_k_to_tmem = [&](int slot, int buffer) {
        const uint64_t sf_desc = make_desc_sf(kv_slot_sf_smem(slot));
        #pragma unroll
        for (int d = 0; d < FA4_QK_K_ITERS; d++) {
            tcgen05_cp(sf_k_tmem(buffer) + d * 4, sf_desc + (uint64_t)d * (SF_ATOM_BYTES >> 4));
        }
    };

    auto copy_sf_v_to_tmem = [&](int slot, int buffer) {
        const uint64_t sf_desc = make_desc_sf(kv_slot_sf_smem(slot));
        #pragma unroll
        for (int c = 0; c < FA4_PV_K_ITERS; c++) {
            tcgen05_cp(sf_v_tmem(buffer) + c * 4, sf_desc + (uint64_t)c * (SF_ATOM_BYTES >> 4));
        }
    };

    auto qk_gemm = [&](int buf, int k_slot, int sf_k_buffer) {
        #pragma unroll
        for (int d = 0; d < FA4_QK_K_ITERS; d++) {
            const uint64_t q_desc = make_desc_data_64b(q_smem() + d * 32);
            const uint64_t k_desc = make_desc_data_64b(kv_slot_smem(k_slot) + d * 32);
            tcgen05_mma_nvfp4(
                s_tmem(buf),
                q_desc,
                k_desc,
                fa4_qk_idesc(),
                tmem_base + TMEM_SFQ + d * 4,
                sf_k_tmem(sf_k_buffer) + d * 4,
                d != 0);
        }
    };

    auto pv_gemm = [&](int buf, int chunk, int v_slot, int sf_v_buffer, bool accumulate) {
        const uint64_t v_desc = make_desc_data_32b(kv_slot_smem(v_slot) + chunk * KV_CHUNK_BYTES);
        tcgen05_mma_nvfp4_pv(
            o_tmem(),
            p_tmem(buf, chunk),
            v_desc,
            fa4_pv_idesc(),
            sf_p_tmem(buf, chunk),
            sf_v_tmem(sf_v_buffer) + chunk * 4,
            accumulate ? 1 : 0);
        // Row-sum ride-along.
        tcgen05_mma_nvfp4_pv(
            o_sum_tmem(),
            p_tmem(buf, chunk),
            make_desc_data_32b(smem + SMEM_ONES),
            fa4_sum_idesc(),
            sf_p_tmem(buf, chunk),
            tmem_base + TMEM_SFONES,
            accumulate ? 1 : 0);
    };

    auto rescale_o = [&](float scale) {
        const uint32_t row_tmem = static_cast<uint32_t>(warp_id * WARP_SIZE) << 16;
        float vals[32];
        #pragma unroll
        for (int col = 0; col < FA4_HEAD_DIM; col += 32) {
            tcgen05_ld_32x32bx32(vals, o_tmem() + row_tmem + col);
            #pragma unroll
            for (int i = 0; i < 32; i += 2) {
                ta_mul2(vals[i], vals[i + 1], vals[i], vals[i + 1], scale);
            }
            tcgen05_st_32x32bx32(o_tmem() + row_tmem + col, vals);
        }
        float sums[FA4_SUM_N];
        tcgen05_ld_32x32bx8(sums, o_sum_tmem() + row_tmem);
        #pragma unroll
        for (int i = 0; i < FA4_SUM_N; i += 2) {
            ta_mul2(sums[i], sums[i + 1], sums[i], sums[i + 1], scale);
        }
        tcgen05_st_32x32bx8(o_sum_tmem() + row_tmem,
                            reinterpret_cast<const uint32_t*>(sums));
        tcgen05_wait_st();
    };

    auto reduce_blocks = [&](const float* scores, float* block_row_max) {
        #pragma unroll
        for (int block = 0; block < 4; block++) {
            const float local_max = ta_reduce_max_16(scores + block * 16);
            block_row_max[block] = fmaxf(fmaf(local_max, softmax_scale_log2, 0.5f) + TA_MAGIC, TA_MAGIC_FLOOR);
        }
    };

    auto convert_p = [&](int buf, int chunk, const float* scores,
                         const float* block_row_max, uint32_t row_tmem) {
        uint32_t p_words[8];
        #pragma unroll
        for (int block = 0; block < 4; block++) {
            const float block_addend = TA_MAGIC_X2 - block_row_max[block];
            const int col = block * 16;
            {
                float p0, p1, p2, p3, p4, p5, p6, p7;
                ta_score_to_p2(p0, p1, scores[col + 0], scores[col + 1], softmax_scale_log2, block_addend);
                ta_score_to_p2(p2, p3, scores[col + 2], scores[col + 3], softmax_scale_log2, block_addend);
                ta_score_to_p2(p4, p5, scores[col + 4], scores[col + 5], softmax_scale_log2, block_addend);
                ta_score_to_p2(p6, p7, scores[col + 6], scores[col + 7], softmax_scale_log2, block_addend);
                p_words[block * 2] = ta_cvt_8xf32_to_e2m1_packed(
                    p1, p0, p3, p2,
                    p5, p4, p7, p6);
            }
            {
                float p8, p9, p10, p11, p12, p13, p14, p15;
                ta_score_to_p2(p8, p9, scores[col + 8], scores[col + 9], softmax_scale_log2, block_addend);
                ta_score_to_p2(p10, p11, scores[col + 10], scores[col + 11], softmax_scale_log2, block_addend);
                ta_score_to_p2(p12, p13, scores[col + 12], scores[col + 13], softmax_scale_log2, block_addend);
                ta_score_to_p2(p14, p15, scores[col + 14], scores[col + 15], softmax_scale_log2, block_addend);
                p_words[block * 2 + 1] = ta_cvt_8xf32_to_e2m1_packed(
                    p9, p8, p11, p10,
                    p13, p12, p15, p14);
            }
        }
        tcgen05_st_32x32bx8(p_tmem(buf, chunk) + row_tmem, p_words);
    };

    auto sf_word_of = [&](const float* bm, float m_used) -> uint32_t {
        float sf_p[4];
        #pragma unroll
        for (int block = 0; block < 4; block++) {
            const float sf_delta = fmaxf((bm[block] - m_used) + TA_MAGIC, TA_MAGIC_FLOOR);
            sf_p[block] = P_SCALE_RANGE * ta_pow2_from_bits(__float_as_uint(sf_delta));
        }
        return ta_cvt_4xf32_to_e4m3_packed(sf_p[1], sf_p[0], sf_p[3], sf_p[2]);
    };

    // Owner warp w (0-3): chunk 0 of every tile for rows 32w..32w+31, plus
    // the lazy row-max bookkeeping, O rescale, and the per-tile epilogue.
    // All pipeline phases continue across tiles (drain-free persistence).
    auto softmax_owner = [&]() {
        const int row = warp_id * WARP_SIZE + lane_id;
        const uint32_t row_tmem = static_cast<uint32_t>(warp_id * WARP_SIZE) << 16;
        const int bar_id = 2 + warp_id;
        int gi = 0;

        for (int lt = 0, gtile = blockIdx.x; gtile < num_tiles;
             lt++, gtile += static_cast<int>(gridDim.x)) {
            float m_used = TA_MAGIC_FLOOR;

            for (int iter = 0; iter < kv_iters; iter++) {
                const int g = gi + iter;
                const int buf = g & 1;
                const int ph = (g >> 1) & 1;
                float scores[64];
                float block_row_max[4];

                mbarrier_wait(mbar_base + mbar_offset(MbarId::SFull, buf), ph);
                tcgen05_fence_after();
                tcgen05_ld_32x32bx64(scores, s_tmem(buf) + row_tmem);
                reduce_blocks(scores, block_row_max);
                float half_max = ta_fmax3(block_row_max[0], block_row_max[1], block_row_max[2]);
                half_max = fmaxf(half_max, block_row_max[3]);

                named_barrier_sync(bar_id, 2 * WARP_SIZE);  // partner posted its max
                const float tile_max = fmaxf(half_max, hmax1[row]);
                float acc_scale = 1.0f;
                if (tile_max > m_used + LAZY_RESCALE_T) {
                    const float acc_delta = fmaxf((m_used - tile_max) + TA_MAGIC, TA_MAGIC_FLOOR);
                    acc_scale = ta_pow2_from_bits(__float_as_uint(acc_delta));
                    m_used = tile_max;
                }
                hm_used[row] = m_used;
                named_barrier_sync(bar_id, 2 * WARP_SIZE);  // m_used visible

                // SFull implies the previous PV completed (commit chain), so
                // the rare rescale is ordered; iter 0 rescale can't fire (the
                // lazy threshold always trips into the acc path fresh, but O
                // is overwritten by the first PV of the tile anyway).
                if (iter > 0 && __any_sync(0xffffffffu, acc_scale != 1.0f)) {
                    tcgen05_fence_after();
                    rescale_o(acc_scale);
                }

                convert_p(buf, 0, scores, block_row_max, row_tmem);
                tcgen05_st_32x32bx1(sf_p_tmem(buf, 0) + row_tmem + warp_id, sf_word_of(block_row_max, m_used));
                tcgen05_wait_st();
                mbarrier_arrive(mbar_base + mbar_offset(MbarId::P0Ready, buf));
                mbarrier_arrive(mbar_base + mbar_offset(MbarId::P1Ready, buf));
            }
            gi += kv_iters;

            // Per-tile epilogue: normalise by the tmem row sum, stage O in
            // smem, then release the O accumulator for the next tile.
            __nv_bfloat16* out = o_stage_ptr() + row * FA4_HEAD_DIM;
            float vals[32];
            mbarrier_wait(mbar_base + mbar_offset(MbarId::OFull), lt & 1);
            tcgen05_fence_after();
            float rs;
            tcgen05_ld_32x32bx1(&rs, o_sum_tmem() + row_tmem);
            const float norm = ((rs == 0.0f || rs != rs) ? 1.0f : (1.0f / rs)) * v_descale;
            // The store warp must have finished reading the previous tile's
            // staged O out of smem.
            mbarrier_wait(mbar_base + mbar_offset(MbarId::OStore), (lt & 1) ^ 1);
            #pragma unroll
            for (int col = 0; col < FA4_HEAD_DIM; col += 32) {
                tcgen05_ld_32x32bx32(vals, o_tmem() + row_tmem + col);
                nv_bfloat162* out2 = reinterpret_cast<nv_bfloat162*>(out + col);
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    ta_mul2(vals[2 * i], vals[2 * i + 1], vals[2 * i], vals[2 * i + 1], norm);
                    out2[i] = __float22bfloat162_rn({vals[2 * i], vals[2 * i + 1]});
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(mbar_base + mbar_offset(MbarId::OEpi));
            mbarrier_arrive(mbar_base + mbar_offset(MbarId::OEmpty));
        }

        mbarrier_arrive(mbar_base + mbar_offset(MbarId::TmemDealloc));
    };

    // Partner warp w+4: chunk 1 (cols 64-127) for the same rows.
    auto softmax_partner = [&]() {
        const int w = warp_id - 4;
        const int row = w * WARP_SIZE + lane_id;
        const uint32_t row_tmem = static_cast<uint32_t>(w * WARP_SIZE) << 16;
        const int bar_id = 2 + w;
        int gi = 0;

        for (int gtile = blockIdx.x; gtile < num_tiles; gtile += static_cast<int>(gridDim.x)) {
            for (int iter = 0; iter < kv_iters; iter++) {
                const int g = gi + iter;
                const int buf = g & 1;
                const int ph = (g >> 1) & 1;
                float scores[64];
                float block_row_max[4];

                mbarrier_wait(mbar_base + mbar_offset(MbarId::SFull, buf), ph);
                tcgen05_fence_after();
                tcgen05_ld_32x32bx64(scores, s_tmem(buf) + row_tmem + 64);
                reduce_blocks(scores, block_row_max);
                float half_max = ta_fmax3(block_row_max[0], block_row_max[1], block_row_max[2]);
                hmax1[row] = fmaxf(half_max, block_row_max[3]);

                named_barrier_sync(bar_id, 2 * WARP_SIZE);  // max posted
                named_barrier_sync(bar_id, 2 * WARP_SIZE);  // m_used ready
                const float m_used = hm_used[row];

                convert_p(buf, 1, scores, block_row_max, row_tmem);
                tcgen05_st_32x32bx1(sf_p_tmem(buf, 1) + row_tmem + w, sf_word_of(block_row_max, m_used));
                tcgen05_wait_st();
                mbarrier_arrive(mbar_base + mbar_offset(MbarId::P1Ready, buf));
            }
            gi += kv_iters;
        }
    };

    auto epilogue_store = [&]() {
        for (int lt = 0, gtile = blockIdx.x; gtile < num_tiles;
             lt++, gtile += static_cast<int>(gridDim.x)) {
            set_tile(gtile);
            mbarrier_wait(mbar_base + mbar_offset(MbarId::OEpi), lt & 1);
            tma_2d_smem2gmem(&o_tmap, 0, q_global_row_base, o_stage_smem());
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read_0();
            mbarrier_arrive(mbar_base + mbar_offset(MbarId::OStore));
        }
    };

    auto load_loop = [&]() {
        if (!elect_sync()) {
            return;
        }
        int gi = 0;
        for (int lt = 0, gtile = blockIdx.x; gtile < num_tiles;
             lt++, gtile += static_cast<int>(gridDim.x)) {
            set_tile(gtile);
            mbarrier_wait(mbar_base + mbar_offset(MbarId::QEmpty), (lt & 1) ^ 1);
            issue_q();
            for (int iter = 0; iter < kv_iters; iter++) {
                const int g = gi + iter;
                const int k_slot = 2 * (g % FA4_NUM_KV_PAIRS);
                issue_k(k_slot, g, iter);
                issue_v(k_slot + 1, g, iter);
            }
            gi += kv_iters;
        }
    };

    auto mma_loop = [&]() {
        const bool elected = elect_sync() != 0;

        if (elected) {
            tcgen05_cp(tmem_base + TMEM_SFONES, make_desc_sf(smem + SMEM_ONES_SF));

            int gi = 0;
            for (int lt = 0, gtile = blockIdx.x; gtile < num_tiles;
                 lt++, gtile += static_cast<int>(gridDim.x)) {
                mbarrier_wait(mbar_base + mbar_offset(MbarId::QFull), lt & 1);
                tcgen05_fence_after();
                copy_sf_q_to_tmem();

                // First QK of the tile.
                {
                    const int g = gi;
                    const int buf = g & 1;
                    const int k_slot = 2 * (g % FA4_NUM_KV_PAIRS);
                    const int k_phase = (g / FA4_NUM_KV_PAIRS) & 1;
                    mbarrier_wait(mbar_base + mbar_offset(MbarId::KVFull, k_slot), k_phase);
                    tcgen05_fence_after();
                    copy_sf_k_to_tmem(k_slot, buf);
                    mbarrier_wait(mbar_base + mbar_offset(MbarId::SEmpty, buf), (((g >> 1)) & 1) ^ 1);
                    tcgen05_fence_after();
                    qk_gemm(buf, k_slot, buf);
                    tcgen05_commit(mbar_base + mbar_offset(MbarId::SFull, buf));
                    tcgen05_commit(mbar_base + mbar_offset(MbarId::KVEmpty, k_slot));
                }

                for (int iter = 0; iter < kv_iters; iter++) {
                    const int g = gi + iter;
                    const int buf = g & 1;
                    const int ph = (g >> 1) & 1;
                    const int v_slot = 2 * (g % FA4_NUM_KV_PAIRS) + 1;
                    const int v_phase = (g / FA4_NUM_KV_PAIRS) & 1;
                    const int vbuf = g & 1;

                    // Prefetch the next tile's QK first so SFull(g+1) fires
                    // while softmax is still converting this tile.
                    if (iter + 1 < kv_iters) {
                        const int ng = g + 1;
                        const int nbuf = ng & 1;
                        const int nk_slot = 2 * (ng % FA4_NUM_KV_PAIRS);
                        const int nk_phase = (ng / FA4_NUM_KV_PAIRS) & 1;

                        mbarrier_wait(mbar_base + mbar_offset(MbarId::KVFull, nk_slot), nk_phase);
                        tcgen05_fence_after();
                        copy_sf_k_to_tmem(nk_slot, nbuf);
                        mbarrier_wait(mbar_base + mbar_offset(MbarId::SEmpty, nbuf), ((ng >> 1) & 1) ^ 1);
                        tcgen05_fence_after();
                        qk_gemm(nbuf, nk_slot, nbuf);
                        tcgen05_commit(mbar_base + mbar_offset(MbarId::SFull, nbuf));
                        tcgen05_commit(mbar_base + mbar_offset(MbarId::KVEmpty, nk_slot));
                    }

                    mbarrier_wait(mbar_base + mbar_offset(MbarId::KVFull, v_slot), v_phase);
                    tcgen05_fence_after();
                    copy_sf_v_to_tmem(v_slot, vbuf);

                    mbarrier_wait(mbar_base + mbar_offset(MbarId::P0Ready, buf), ph);
                    tcgen05_fence_after();
                    if (iter == 0) {
                        // Previous tile's epilogue must have released O/O_sum.
                        mbarrier_wait(mbar_base + mbar_offset(MbarId::OEmpty), (lt & 1) ^ 1);
                        tcgen05_fence_after();
                    }
                    pv_gemm(buf, 0, v_slot, vbuf, iter > 0);

                    mbarrier_wait(mbar_base + mbar_offset(MbarId::P1Ready, buf), ph);
                    tcgen05_fence_after();
                    pv_gemm(buf, 1, v_slot, vbuf, true);
                    tcgen05_commit(mbar_base + mbar_offset(MbarId::SEmpty, buf));
                    tcgen05_commit(mbar_base + mbar_offset(MbarId::KVEmpty, v_slot));
                }
                gi += kv_iters;

                tcgen05_commit(mbar_base + mbar_offset(MbarId::QEmpty));
                tcgen05_commit(mbar_base + mbar_offset(MbarId::OFull));
            }

            mbarrier_wait(mbar_base + mbar_offset(MbarId::TmemDealloc), 0);
        }

        __syncwarp();
        tcgen05_dealloc(tmem_base, TMEM_COLS);
    };

    if (warp_id < 4) {
        softmax_owner();
    } else if (warp_id < 8) {
        softmax_partner();
    } else if (warp_id == 12) {
        mma_loop();
    } else if (warp_id == 13) {
        if (elect_sync()) {
            epilogue_store();
        }
    } else if (warp_id == 14) {
        load_loop();
    }
}

namespace {

__global__
void fa4_pack_qk_sf_atoms_kernel(const uint8_t* __restrict__ sf,
                                 uint8_t* __restrict__ atoms,
                                 int seq_len,
                                 int tile_rows)
{
    const int tiles = seq_len / tile_rows;
    const int atom_id = blockIdx.x;
    const int k_chunk = atom_id % FA4_QK_K_ITERS;
    const int tile = (atom_id / FA4_QK_K_ITERS) % tiles;
    const int group = atom_id / (tiles * FA4_QK_K_ITERS);
    const int tid = threadIdx.x;
    const uint64_t atom_base = static_cast<uint64_t>(atom_id) * SF_ATOM_BYTES;

    atoms[atom_base + tid] = 0;
    __syncthreads();

    if (tid < tile_rows * 4) {
        const int row = tid / 4;
        const int kblock = tid & 3;
        const int src_row = tile * tile_rows + row;
        const int src_kblock = k_chunk * 4 + kblock;
        const int dst = (row & 31) * 16 + (row >> 5) * 4 + kblock;
        atoms[atom_base + dst] = sf[(group * seq_len + src_row) * (FA4_HEAD_DIM / 16) + src_kblock];
    }
}

__global__
void fa4_pack_v_sf_atoms_kernel(const uint8_t* __restrict__ sf_v_t,
                                uint8_t* __restrict__ atoms,
                                int kv_len)
{
    const int tiles = kv_len / 64;  // sf atoms stay per-64-kv columns
    const int atom_id = blockIdx.x;
    const int tile = atom_id % tiles;
    const int group = atom_id / tiles;
    const int tid = threadIdx.x;
    const uint64_t atom_base = static_cast<uint64_t>(atom_id) * SF_ATOM_BYTES;

    atoms[atom_base + tid] = 0;
    __syncthreads();

    if (tid < FA4_HEAD_DIM * 4) {
        const int row = tid / 4;
        const int kblock = tid & 3;
        const int src_kblock = tile * 4 + kblock;
        const int dst = (row & 31) * 16 + (row >> 5) * 4 + kblock;
        atoms[atom_base + dst] = sf_v_t[(group * FA4_HEAD_DIM + row) * (kv_len / 16) + src_kblock];
    }
}

__host__ __forceinline__
bool fa4_is_aligned(const void* ptr, uintptr_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

__host__
cudaError_t fa4_cu_result(CUresult err, const char* what)
{
    if (err == CUDA_SUCCESS) {
        return cudaSuccess;
    }

    const char* msg = nullptr;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS || msg == nullptr) {
        msg = "unknown driver error";
    }
    std::fprintf(stderr, "%s failed: %s\n", what, msg);
    return cudaErrorUnknown;
}

__host__
cudaError_t fa4_validate_launch_params(const void* q,
                                       const void* k,
                                       const void* v_t,
                                       const uint8_t* sf_q_atoms,
                                       const uint8_t* sf_k_atoms,
                                       const uint8_t* sf_v_atoms,
                                       __nv_bfloat16* o,
                                       int batch,
                                       int q_len,
                                       int kv_len,
                                       int num_q_heads,
                                       int num_kv_heads)
{
    if (q == nullptr || k == nullptr || v_t == nullptr || o == nullptr ||
        sf_q_atoms == nullptr || sf_k_atoms == nullptr || sf_v_atoms == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (batch <= 0 || q_len <= 0 || kv_len <= 0 || num_q_heads <= 0 || num_kv_heads <= 0) {
        return cudaErrorInvalidValue;
    }
    if ((q_len % FA4_Q_TILE_ROWS) != 0 || (kv_len % FA4_KV_TILE) != 0) {
        return cudaErrorInvalidValue;
    }
    if ((num_q_heads % num_kv_heads) != 0) {
        return cudaErrorInvalidValue;
    }
    if (SMEM_BYTES > 227 * 1024) {
        return cudaErrorInvalidConfiguration;
    }
    if (!fa4_is_aligned(q, 16) || !fa4_is_aligned(k, 16) || !fa4_is_aligned(v_t, 16) ||
        !fa4_is_aligned(o, 16) || !fa4_is_aligned(sf_q_atoms, 16) ||
        !fa4_is_aligned(sf_k_atoms, 16) || !fa4_is_aligned(sf_v_atoms, 16)) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

__host__
cudaError_t fa4_validate_sf_pack_params(const uint8_t* sf,
                                        uint8_t* atoms,
                                        int batch,
                                        int heads,
                                        int seq_len,
                                        int tile_rows)
{
    if (sf == nullptr || atoms == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (batch <= 0 || heads <= 0 || seq_len <= 0 || tile_rows <= 0 || (seq_len % tile_rows) != 0) {
        return cudaErrorInvalidValue;
    }
    if (!fa4_is_aligned(sf, 16) || !fa4_is_aligned(atoms, 16)) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

__host__
cudaError_t fa4_encode_fp4_tmap(CUtensorMap* tmap,
                                const void* ptr,
                                uint64_t global_width_fp4,
                                uint64_t global_rows,
                                uint32_t box_width_fp4,
                                uint32_t box_rows,
                                CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 2;
    uint64_t global_dim[rank] = {global_width_fp4, global_rows};
    uint64_t global_strides[rank - 1] = {global_width_fp4 / 2};
    uint32_t box_dim[rank] = {box_width_fp4, box_rows};
    uint32_t element_strides[rank] = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        rank,
        const_cast<void*>(ptr),
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    return fa4_cu_result(err, "cuTensorMapEncodeTiled(fp4)");
}

__host__
cudaError_t fa4_encode_o_tmap(CUtensorMap* tmap,
                              __nv_bfloat16* ptr,
                              uint64_t global_rows)
{
    constexpr uint32_t rank = 2;
    uint64_t global_dim[rank] = {FA4_HEAD_DIM, global_rows};
    uint64_t global_strides[rank - 1] = {FA4_HEAD_DIM * sizeof(__nv_bfloat16)};
    uint32_t box_dim[rank] = {FA4_HEAD_DIM, FA4_Q_STAGE_ROWS};
    uint32_t element_strides[rank] = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        ptr,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    return fa4_cu_result(err, "cuTensorMapEncodeTiled(O)");
}

}  // namespace

extern "C"
// sf_q: [batch, q_heads, q_len, head_dim / 16] raw e4m3 scale bytes.
// sf_q_atoms: [batch, q_heads, q_len / 128, 2, 512] in (32,4,4) atom layout.
cudaError_t nvfp4_sm100_pack_q_sf_atoms(const uint8_t* sf_q,
                                        uint8_t* sf_q_atoms,
                                        int batch,
                                        int num_q_heads,
                                        int q_len,
                                        cudaStream_t stream)
{
    cudaError_t err = fa4_validate_sf_pack_params(
        sf_q, sf_q_atoms, batch, num_q_heads, q_len, FA4_Q_STAGE_ROWS);
    if (err != cudaSuccess) {
        return err;
    }

    const int groups = batch * num_q_heads;
    const int tiles = q_len / FA4_Q_STAGE_ROWS;
    const int blocks = groups * tiles * FA4_QK_K_ITERS;
    fa4_pack_qk_sf_atoms_kernel<<<blocks, SF_ATOM_BYTES, 0, stream>>>(
        sf_q, sf_q_atoms, q_len, FA4_Q_STAGE_ROWS);
    return cudaGetLastError();
}

extern "C"
// sf_k: [batch, kv_heads, kv_len, head_dim / 16] raw e4m3 scale bytes.
// sf_k_atoms: [batch, kv_heads, kv_len / 64, 2, 512] in (32,4,4) atom layout.
cudaError_t nvfp4_sm100_pack_k_sf_atoms(const uint8_t* sf_k,
                                        uint8_t* sf_k_atoms,
                                        int batch,
                                        int num_kv_heads,
                                        int kv_len,
                                        cudaStream_t stream)
{
    cudaError_t err = fa4_validate_sf_pack_params(
        sf_k, sf_k_atoms, batch, num_kv_heads, kv_len, FA4_KV_TILE);
    if (err != cudaSuccess) {
        return err;
    }

    const int groups = batch * num_kv_heads;
    const int tiles = kv_len / FA4_KV_TILE;
    const int blocks = groups * tiles * FA4_QK_K_ITERS;
    fa4_pack_qk_sf_atoms_kernel<<<blocks, SF_ATOM_BYTES, 0, stream>>>(
        sf_k, sf_k_atoms, kv_len, FA4_KV_TILE);
    return cudaGetLastError();
}

extern "C"
// sf_v_t: [batch, kv_heads, head_dim, kv_len / 16] raw e4m3 scale bytes matching transposed V.
// sf_v_atoms: [batch, kv_heads, kv_len / 64, 1, 512] in (32,4,4) atom layout.
cudaError_t nvfp4_sm100_pack_v_sf_atoms(const uint8_t* sf_v_t,
                                        uint8_t* sf_v_atoms,
                                        int batch,
                                        int num_kv_heads,
                                        int kv_len,
                                        cudaStream_t stream)
{
    cudaError_t err = fa4_validate_sf_pack_params(
        sf_v_t, sf_v_atoms, batch, num_kv_heads, kv_len, FA4_KV_TILE);
    if (err != cudaSuccess) {
        return err;
    }

    const int groups = batch * num_kv_heads;
    const int tiles = kv_len / FA4_KV_TILE;
    const int blocks = groups * tiles * FA4_PV_K_ITERS;
    fa4_pack_v_sf_atoms_kernel<<<blocks, SF_ATOM_BYTES, 0, stream>>>(
        sf_v_t, sf_v_atoms, kv_len);
    return cudaGetLastError();
}

extern "C"
// q/k: [batch, heads, seq, head_dim / 2] packed fp4 row-major.
// v_t: [batch, kv_heads, head_dim, kv_len / 2] packed fp4, transposed.
// sf_*_atoms: prepacked contiguous (32,4,4) e4m3 atoms consumed by q_sf_src/k_sf_src/v_sf_src.
// softmax_scale_log2 includes log2(e) and any global Q/K descale factors.
cudaError_t nvfp4_sm100_attention_launch(const void* q,
                                         const void* k,
                                         const void* v_t,
                                         const uint8_t* sf_q_atoms,
                                         const uint8_t* sf_k_atoms,
                                         const uint8_t* sf_v_atoms,
                                         __nv_bfloat16* o,
                                         int batch,
                                         int q_len,
                                         int kv_len,
                                         int num_q_heads,
                                         int num_kv_heads,
                                         float softmax_scale_log2,
                                         float v_descale,
                                         cudaStream_t stream)
{
    cudaError_t err = fa4_validate_launch_params(
        q, k, v_t, sf_q_atoms, sf_k_atoms, sf_v_atoms, o,
        batch, q_len, kv_len, num_q_heads, num_kv_heads);
    if (err != cudaSuccess) {
        return err;
    }

    const uint64_t total_q_rows = static_cast<uint64_t>(batch) * num_q_heads * q_len;
    const uint64_t total_kv_rows = static_cast<uint64_t>(batch) * num_kv_heads * kv_len;
    const uint64_t total_v_rows = static_cast<uint64_t>(batch) * num_kv_heads * FA4_HEAD_DIM;

    CUtensorMap q_tmap;
    CUtensorMap k_tmap;
    CUtensorMap v_tmap;
    CUtensorMap o_tmap;

    err = fa4_encode_fp4_tmap(&q_tmap, q, FA4_HEAD_DIM, total_q_rows,
                              FA4_HEAD_DIM, FA4_Q_STAGE_ROWS,
                              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B);
    if (err != cudaSuccess) {
        return err;
    }
    err = fa4_encode_fp4_tmap(&k_tmap, k, FA4_HEAD_DIM, total_kv_rows,
                              FA4_HEAD_DIM, FA4_KV_TILE,
                              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B);
    if (err != cudaSuccess) {
        return err;
    }
    err = fa4_encode_fp4_tmap(&v_tmap, v_t, kv_len, total_v_rows,
                              FA4_MMA_K, FA4_HEAD_DIM,
                              CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B);
    if (err != cudaSuccess) {
        return err;
    }
    err = fa4_encode_o_tmap(&o_tmap, o, total_q_rows);
    if (err != cudaSuccess) {
        return err;
    }

    auto kernel = nvfp4_sm100_attention_kernel;
    err = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    if (err != cudaSuccess) {
        return err;
    }

    const int launch_q_blocks = q_len / FA4_Q_TILE_ROWS;
    const int num_tiles = batch * num_q_heads * launch_q_blocks;
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    if (sm_count <= 0) {
        sm_count = num_tiles;
    }
    dim3 grid(num_tiles < sm_count ? num_tiles : sm_count);
    kernel<<<grid, FA4_THREADS, SMEM_BYTES, stream>>>(
        q_tmap,
        k_tmap,
        v_tmap,
        o_tmap,
        sf_q_atoms,
        sf_k_atoms,
        sf_v_atoms,
        q_len,
        kv_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale_log2,
        v_descale,
        num_tiles);

    return cudaGetLastError();
}
