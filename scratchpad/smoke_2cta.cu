// Smoke tests for the flagged 2-CTA mechanics before the kernel uses them.
//   T1: split-half TMA with .cta_group::2 — BOTH CTAs issue their own half;
//       the mbarrier operand must be the PAIR LEADER's barrier in cluster
//       address space (mapa to even rank; FA4 clears the rank bit with an
//       AND).  Leader alone does expect_tx of the pair total and waits.
//   T2: leader-issued tcgen05.cp.cta_group::2 reads EACH CTA's local smem at
//       the same offset and writes EACH CTA's own tmem; commit with
//       .multicast::cluster wakes a local mbarrier in both CTAs.
// Each test runs under a host watchdog: a hang prints HANG instead of
// blocking (cuda-gdb-free triage, per HANG_FIX_NOTES methodology).
// Build: nvcc -o smoke_2cta smoke_2cta.cu -gencode=arch=compute_100a,code=sm_100a -std=c++17 -lcuda
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

__device__ __forceinline__ uint32_t cluster_rank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync()
{
    asm volatile("barrier.cluster.arrive;\n\tbarrier.cluster.wait;" ::: "memory");
}

__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

__device__ __forceinline__ void mbar_init(uint32_t mbar, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar), "r"(count));
}

__device__ __forceinline__ void mbar_wait(uint32_t mbar, int phase)
{
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni WAIT;\n\t"
        "DONE:\n\t}"
        :: "r"(mbar), "r"(phase), "r"(0x989680));
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t mbar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mbar), "r"(bytes) : "memory");
}

// Half-tile TMA: dst is the issuing CTA's LOCAL smem; mbar is a cluster-space
// address (pass the mapa'd leader barrier).
__device__ __forceinline__ void tma_2d_pair(uint32_t dst, const void* tmap, int x, int y, uint32_t mbar_cluster)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar_cluster) : "memory");
}

// T3: plain (non-tensor) cp.async.bulk in cluster space, as needed for the
// K/V/Q scale-factor atoms: dst = mapa'd SELF address (each CTA fills its
// own smem), completion = mapa'd LEADER barrier; leader expect_tx's both
// CTAs' bytes.  (Plain bulk rejects .cta_group::2, so this is the route.)
__device__ __forceinline__ void bulk_cluster(uint32_t dst_cluster, const void* src,
                                             uint32_t size, uint32_t mbar_cluster)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];"
        :: "r"(dst_cluster), "l"(src), "r"(size), "r"(mbar_cluster) : "memory");
}

__global__ void __cluster_dims__(2, 1, 1) t3_kernel(const uint8_t* __restrict__ src,
                                                    uint32_t* ok)
{
    __shared__ __align__(128) uint8_t buf[512];
    __shared__ __align__(8) uint64_t mbar_storage;
    const uint32_t rank = cluster_rank();
    const uint32_t mbar = (uint32_t)__cvta_generic_to_shared(&mbar_storage);
    const uint32_t smem = (uint32_t)__cvta_generic_to_shared(buf);

    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        buf[i] = 0xEE;
    }
    if (threadIdx.x == 0) {
        mbar_init(mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    cluster_sync();

    if (threadIdx.x == 0) {
        if (rank == 0) {
            mbar_expect_tx(mbar, 2 * 512);
        }
        // Each CTA copies a DIFFERENT 512B gmem slice into its OWN smem.
        bulk_cluster(mapa_shared(smem, rank), src + rank * 512, 512,
                     mapa_shared(mbar, rank & ~1u));
    }
    __syncthreads();
    if (rank == 0 && threadIdx.x == 0) {
        mbar_wait(mbar, 0);
    }
    __syncthreads();
    cluster_sync();

    if (threadIdx.x < 128) {
        const uint8_t expect = (uint8_t)((rank * 512 + threadIdx.x * 4 + 1) & 0xff);
        if (buf[threadIdx.x * 4 + 1] != expect) {
            atomicExch(ok, 300 + rank);
        }
    }
    cluster_sync();
}

__global__ void __cluster_dims__(2, 1, 1) t1_kernel(const __grid_constant__ CUtensorMap tmap,
                                                    uint32_t* ok)
{
    __shared__ __align__(128) uint8_t buf[64 * 64];
    __shared__ __align__(8) uint64_t mbar_storage;
    const uint32_t rank = cluster_rank();
    const uint32_t mbar = (uint32_t)__cvta_generic_to_shared(&mbar_storage);
    const uint32_t smem = (uint32_t)__cvta_generic_to_shared(buf);

    if (threadIdx.x == 0) {
        mbar_init(mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    cluster_sync();

    if (threadIdx.x == 0) {
        if (rank == 0) {
            mbar_expect_tx(mbar, 2 * 64 * 64);   // pair total, leader only
        }
        // Both CTAs issue their own 64-row half; completion routes to the
        // LEADER's barrier via the mapa'd cluster address.
        const uint32_t leader_mbar = mapa_shared(mbar, rank & ~1u);
        tma_2d_pair(smem, &tmap, 0, (int)rank * 64, leader_mbar);
    }
    __syncthreads();

    if (rank == 0 && threadIdx.x == 0) {
        mbar_wait(mbar, 0);
    }
    __syncthreads();
    cluster_sync();   // follower proceeds only after leader saw completion

    if (threadIdx.x < 64) {
        const int row = (int)rank * 64 + threadIdx.x;
        const uint8_t expect = (uint8_t)(row & 0xff);
        uint8_t got = buf[threadIdx.x * 64 + 17];
        if (got != expect) {
            atomicExch(ok, 100 + rank);
        }
    }
    cluster_sync();
}

// ---- T2: tcgen05.cp cta_group::2 + commit multicast --------------------
__device__ __forceinline__ uint64_t sf_desc(uint32_t addr)
{
    auto enc = [](uint64_t x) { return (x & 0x3ffffULL) >> 4ULL; };
    return enc(addr) | (enc(8 * 16) << 32ULL) | (1ULL << 46ULL);
}

__global__ void __cluster_dims__(2, 1, 1) t2_kernel(uint32_t* ok)
{
    __shared__ __align__(128) uint32_t sfbuf[128];
    __shared__ __align__(8) uint64_t mbar_storage;
    __shared__ uint32_t tmem_addr;
    const uint32_t rank = cluster_rank();
    const uint32_t mbar = (uint32_t)__cvta_generic_to_shared(&mbar_storage);
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        sfbuf[i] = 0xA0000000u + rank * 0x01000000u + i;   // per-CTA pattern
    }
    if (threadIdx.x == 0) {
        mbar_init(mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (warp_id == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_addr)), "r"(32));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
    __syncthreads();
    cluster_sync();

    if (rank == 0 && threadIdx.x == 0) {
        asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                     :: "r"(tmem_addr), "l"(sf_desc((uint32_t)__cvta_generic_to_shared(sfbuf))));
        uint16_t mask = 3;
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                     :: "r"(mbar), "h"(mask) : "memory");
    }
    __syncthreads();

    if (warp_id == 0) {
        if (lane_id == 0) {
            mbar_wait(mbar, 0);   // both CTAs: fired by the multicast commit
        }
        __syncwarp();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        float v[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3])
                     : "r"(tmem_addr + (uint32_t)(lane_id << 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const uint32_t got = __float_as_uint(v[0]);
        const uint32_t expect_hi = 0xA0000000u + rank * 0x01000000u;
        if ((got & 0xFF000000u) != (expect_hi & 0xFF000000u)) {
            atomicExch(ok, 200 + rank * 10);
        }
    }
    __syncthreads();
    cluster_sync();
    if (warp_id == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_addr), "r"(32));
    }
}

static int run_watchdog(const char* name, void (*launch)(uint32_t*), uint32_t* ok)
{
    CK(cudaMemset(ok, 0, 4));
    launch(ok);
    // Poll instead of sync so a deadlocked kernel reports HANG.
    for (int ms = 0; ms < 8000; ms += 50) {
        cudaError_t q = cudaStreamQuery(0);
        if (q == cudaSuccess) {
            uint32_t r;
            CK(cudaMemcpy(&r, ok, 4, cudaMemcpyDeviceToHost));
            printf("%s: %s (code %u)\n", name, r == 0 ? "PASS" : "FAIL", r);
            return r == 0 ? 0 : 1;
        }
        if (q != cudaErrorNotReady) {
            printf("%s: ERROR %s\n", name, cudaGetErrorString(q));
            return 1;
        }
        usleep(50 * 1000);
    }
    printf("%s: HANG (kernel never completed; process must be killed)\n", name);
    fflush(stdout);
    _exit(3);
}

static CUtensorMap g_tmap;
static uint8_t* g_src3;
static void launch_t1(uint32_t* ok) { t1_kernel<<<2, 128>>>(g_tmap, ok); }
static void launch_t2(uint32_t* ok) { t2_kernel<<<2, 128>>>(ok); }
static void launch_t3(uint32_t* ok) { t3_kernel<<<2, 128>>>(g_src3, ok); }

int main()
{
    uint8_t* h = new uint8_t[128 * 64];
    for (int r = 0; r < 128; r++) {
        memset(h + r * 64, r & 0xff, 64);
    }
    void* d;
    CK(cudaMalloc(&d, 128 * 64));
    CK(cudaMemcpy(d, h, 128 * 64, cudaMemcpyHostToDevice));

    uint64_t dims[2] = {64, 128};
    uint64_t strides[1] = {64};
    uint32_t box[2] = {64, 64};
    uint32_t elem_str[2] = {1, 1};
    CUresult cr = cuTensorMapEncodeTiled(
        &g_tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d, dims, strides, box, elem_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cr != CUDA_SUCCESS) { printf("tmap encode failed\n"); return 1; }

    uint8_t* h3 = new uint8_t[1024];
    for (int i = 0; i < 1024; i++) {
        h3[i] = (uint8_t)(i & 0xff);
    }
    CK(cudaMalloc(&g_src3, 1024));
    CK(cudaMemcpy(g_src3, h3, 1024, cudaMemcpyHostToDevice));

    uint32_t* ok;
    CK(cudaMalloc(&ok, 4));
    int rc = 0;
    rc |= run_watchdog("T2 (tcgen05.cp cta_group::2 + commit multicast)", launch_t2, ok);
    rc |= run_watchdog("T1 (split TMA cta_group::2 -> mapa'd leader mbar)", launch_t1, ok);
    rc |= run_watchdog("T3 (plain bulk, mapa-self dst -> mapa'd leader mbar)", launch_t3, ok);
    return rc;
}
