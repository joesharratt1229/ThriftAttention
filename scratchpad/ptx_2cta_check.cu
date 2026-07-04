// ptxas syntax validation for every 2-CTA PTX form the FA4_CTA_PAIR variant
// needs. Compiled for sm_100a; if this compiles, the strings are valid.
#include <cstdint>
#include <cuda.h>

__device__ __forceinline__ uint32_t cluster_ctarank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_arrive_relaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait()
{
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// Q5: remote mbarrier arrive on peer CTA's smem (mapa + arrive, cluster scope)
__device__ __forceinline__ void mbarrier_arrive_remote(uint32_t mbar_local, uint32_t peer_rank)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(mbar_local), "r"(peer_rank));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                 :: "r"(remote) : "memory");
}

// Q4: tcgen05.commit cta_group::2 with multicast to both CTAs' mbarriers
__device__ __forceinline__ void tcgen05_commit_mc2(uint32_t mbar, uint16_t mask)
{
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(mbar), "h"(mask) : "memory");
}

// plain cta_group::2 commit (local mbar only)
__device__ __forceinline__ void tcgen05_commit_2(uint32_t mbar)
{
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mbar) : "memory");
}

// Q6: alloc/dealloc/relinquish under cta_group::2
__device__ __forceinline__ void tcgen05_alloc2(uint32_t dst, uint32_t cols)
{
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(dst), "r"(cols));
}

__device__ __forceinline__ void tcgen05_dealloc2(uint32_t taddr, uint32_t cols)
{
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(cols));
}

__device__ __forceinline__ void tcgen05_relinquish2()
{
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}

// Q1: the 2-CTA block-scaled MMA, smem A (QK shape)
__device__ __forceinline__ void mma2_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                                        uint32_t idesc, uint32_t sfa, uint32_t sfb, int en)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(sfa), "r"(sfb), "r"(en));
}

// Q1b: the 2-CTA block-scaled MMA, tmem A (PV shape)
__device__ __forceinline__ void mma2_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc,
                                        uint32_t idesc, uint32_t sfa, uint32_t sfb, int en)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], [%1], %2, %3, [%4], [%5], p;\n\t"
        "}"
        :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(sfa), "r"(sfb), "r"(en));
}

// Q7a: TMA 2d G2S with cluster multicast (mask form)
__device__ __forceinline__ void tma_2d_mc(uint32_t dst, const void* tmap, int x, int y,
                                          uint32_t mbar, uint16_t mask, uint64_t pol)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".multicast::cluster.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5, %6;"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar), "h"(mask), "l"(pol)
        : "memory");
}

// Q7b/Q9: TMA 2d G2S signalling an mbarrier under cta_group::2 semantics
// (mbar may be the PEER CTA's barrier address via mapa)
__device__ __forceinline__ void tma_2d_cg2(uint32_t dst, const void* tmap, int x, int y,
                                           uint32_t mbar, uint64_t pol)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        ".cta_group::2.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar), "l"(pol)
        : "memory");
}

// plain bulk copy (non-tensor): dst in own smem, completion signalled on a
// REMOTE (peer-CTA) mbarrier via the shared::cluster dst/mbar form
__device__ __forceinline__ void tma_1d_remote_mbar(uint32_t dst, const void* src,
                                                   uint32_t size, uint32_t mbar_remote)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];"
        :: "r"(dst), "l"(src), "r"(size), "r"(mbar_remote)
        : "memory");
}

// tensor form with dst own smem + remote mbarrier (no cta_group; sm_90 style)
__device__ __forceinline__ void tma_2d_remote_mbar(uint32_t dst, const void* tmap, int x,
                                                   int y, uint32_t mbar_remote, uint64_t pol)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar_remote), "l"(pol)
        : "memory");
}

// tcgen05.cp under cta_group::2 (SF smem->tmem copy issued by the leader)
__device__ __forceinline__ void tcgen05_cp2(uint32_t taddr, uint64_t smem_desc)
{
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(smem_desc));
}

// mbarrier expect_tx + arrive on a REMOTE (peer) barrier
__device__ __forceinline__ void mbarrier_arrive_expect_tx_remote(uint32_t mbar_local,
                                                                 uint32_t peer_rank,
                                                                 uint32_t bytes)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(mbar_local), "r"(peer_rank));
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
        :: "r"(remote), "r"(bytes) : "memory");
}

extern "C" __global__ void __cluster_dims__(2, 1, 1) ptx_2cta_check(
    const CUtensorMap* tmap, uint32_t* out)
{
    __shared__ __align__(8) uint64_t mbar[2];
    __shared__ __align__(16) char buf[1024];
    __shared__ uint32_t tmem_addr;

    const uint32_t smem_mbar =
        static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    const uint32_t smem_buf =
        static_cast<uint32_t>(__cvta_generic_to_shared(buf));
    const uint32_t smem_tmem =
        static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr));

    const uint32_t rank = cluster_ctarank();
    cluster_arrive_relaxed();
    cluster_wait();

    if (threadIdx.x == 0) {
        tcgen05_alloc2(smem_tmem, 128);
        tcgen05_relinquish2();
        mbarrier_arrive_remote(smem_mbar, rank ^ 1);
        mbarrier_arrive_expect_tx_remote(smem_mbar, rank & ~1u, 4096);
        tcgen05_commit_mc2(smem_mbar, (uint16_t)0x3);
        tcgen05_commit_2(smem_mbar);
        mma2_ss(tmem_addr, 0, 0, (1u << 7) | (1u << 10) | (16u << 17) | (2u << 27),
                tmem_addr + 400, tmem_addr + 408, 1);
        mma2_ts(tmem_addr, tmem_addr + 128, 0,
                (1u << 7) | (1u << 10) | (16u << 17) | (2u << 27),
                tmem_addr + 400, tmem_addr + 408, 0);
        tma_2d_mc(smem_buf, tmap, 0, 0, smem_mbar, (uint16_t)0x3, 0x1000000000000000ull);
        tma_2d_cg2(smem_buf, tmap, 0, 0, smem_mbar, 0x1000000000000000ull);
        uint32_t mbar_remote;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(mbar_remote) : "r"(smem_mbar), "r"(rank & ~1u));
        tma_1d_remote_mbar(smem_buf, out, 256, mbar_remote);
        tma_2d_remote_mbar(smem_buf, tmap, 0, 0, mbar_remote, 0x1000000000000000ull);
        tcgen05_cp2(tmem_addr + 432, 0);
        tcgen05_dealloc2(tmem_addr, 128);
    }
    out[threadIdx.x + blockIdx.x] = rank;
}
