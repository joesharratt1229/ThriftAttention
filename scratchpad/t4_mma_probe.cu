// T4: which cta_group::2 block-scaled MMA encoding is legal on sm_100a?
// Issues ONE tcgen05.mma per variant inside a 2-CTA cluster (leader only),
// commits with multicast, waits under a host watchdog.  Data is garbage --
// only the encoding legality matters (illegal encodings trap the pipe).
#include <cstdio>
#include <cstdint>
#include <unistd.h>
#include <cuda_runtime.h>

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

__device__ __forceinline__ uint64_t make_desc_data_64b(uint32_t addr)
{
    auto enc = [](uint64_t x) { return (x & 0x3ffffULL) >> 4ULL; };
    return enc(addr) | (enc(8 * 64) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
}

// variant: 0 = scale_vec::4X idesc M=256   (the kernel's current form)
//          1 = scale_vec::4X idesc M=128
//          2 = scale_vec::2X idesc M=256
//          3 = control: cta_group::1 scale_vec::4X idesc M=128 (leader only)
__global__ void __cluster_dims__(2, 1, 1) t4_kernel(int variant, uint32_t* ok)
{
    __shared__ __align__(1024) uint8_t data[16384];
    __shared__ __align__(8) uint64_t mbar_storage;
    __shared__ uint32_t tmem_addr;
    const uint32_t rank = cluster_rank();
    const uint32_t mbar = (uint32_t)__cvta_generic_to_shared(&mbar_storage);
    const uint32_t smem = (uint32_t)__cvta_generic_to_shared(data);
    const int warp_id = threadIdx.x / 32;

    for (int i = threadIdx.x; i < 16384; i += blockDim.x) {
        data[i] = (uint8_t)(i * 7 + rank);
    }
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp_id == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_addr)), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
    __syncthreads();
    cluster_sync();

    if (rank == 0 && threadIdx.x == 0) {
        const uint64_t a_desc = make_desc_data_64b(smem);          // "Q" 128x64B
        const uint64_t b_desc = make_desc_data_64b(smem + 8192);   // "K half"
        const uint32_t d_tmem = tmem_addr + 128;                   // S at col 128
        const uint32_t sfa = tmem_addr + 392;
        const uint32_t sfb = tmem_addr + 400;
        const uint32_t idesc_m256 = (1u << 7) | (1u << 10) | (16u << 17) | (16u << 24);
        const uint32_t idesc_m128 = (1u << 7) | (1u << 10) | (16u << 17) | (8u << 24);
        if (variant == 0) {
            // Replicate the kernel's first-QK sequence exactly: sf cps on the
            // same pipe, then two chunk MMAs (second accumulates via pred).
            auto sf_desc = [](uint32_t a) {
                return ((uint64_t)(a & 0x3ffff) >> 4)
                    | (((uint64_t)((8 * 16) & 0x3ffff) >> 4) << 32) | (1ULL << 46);
            };
            asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                         :: "r"(tmem_addr + 432), "l"(sf_desc(smem + 12288)));
            asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                         :: "r"(sfa), "l"(sf_desc(smem + 12800)));
            asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                         :: "r"(sfa + 4), "l"(sf_desc(smem + 13312)));
            asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                         :: "r"(sfb), "l"(sf_desc(smem + 13824)));
            asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                         :: "r"(sfb + 4), "l"(sf_desc(smem + 14336)));
            #pragma unroll
            for (int d = 0; d < 2; d++) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %6, 0;\n\t"
                    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
                    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                    "}"
                    :: "r"(d_tmem), "l"(a_desc + (uint64_t)(d * 32 >> 4)),
                       "l"(b_desc + (uint64_t)(d * 32 >> 4)), "r"(idesc_m256),
                       "r"(sfa + d * 4), "r"(sfb + d * 4), "r"(d));
            }
        } else if (variant == 1) {
            asm volatile(
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], %1, %2, %3, [%4], [%5], 0;"
                :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc_m128), "r"(sfa), "r"(sfb));
        } else if (variant == 2) {
            asm volatile(
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X "
                "[%0], %1, %2, %3, [%4], [%5], 0;"
                :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc_m256), "r"(sfa), "r"(sfb));
        } else {
            asm volatile(
                "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
                "[%0], %1, %2, %3, [%4], [%5], 0;"
                :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc_m128 | (1u << 31)), "r"(sfa), "r"(sfb));
        }
        {
            uint16_t mask = 3;
            asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                         :: "r"(mbar), "h"(mask) : "memory");
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile(
            "{\n\t.reg .pred P1;\n\t"
            "WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], 0, %1;\n\t"
            "@P1 bra.uni DONE;\n\t"
            "bra.uni WAIT;\n\t"
            "DONE:\n\t}"
            :: "r"(mbar), "r"(0x989680));
        atomicExch(ok, 1);   // wait completed -> encoding executed
    }
    __syncthreads();
    cluster_sync();
    if (warp_id == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_addr), "r"(512));
    }
}

int main()
{
    uint32_t* ok;
    cudaMalloc(&ok, 4);
    const char* names[] = {
        "cta2 scale_vec::4X idesc M=256 (kernel form)",
        "cta2 scale_vec::4X idesc M=128",
        "cta2 scale_vec::2X idesc M=256",
        "cta2 scale_vec::4X idesc M=128 k_size=1",
    };
    for (int v = 0; v < 4; v++) {
        cudaMemset(ok, 0, 4);
        t4_kernel<<<2, 128>>>(v, ok);
        bool done = false;
        for (int ms = 0; ms < 5000 && !done; ms += 50) {
            cudaError_t q = cudaStreamQuery(0);
            if (q == cudaSuccess) {
                done = true;
            } else if (q != cudaErrorNotReady) {
                printf("V%d %-46s: TRAP %s\n", v, names[v], cudaGetErrorString(q));
                return 1;   // context poisoned; rerun per-variant if needed
            } else {
                usleep(50 * 1000);
            }
        }
        if (!done) {
            printf("V%d %-46s: HANG\n", v, names[v]);
            return 2;
        }
        uint32_t r;
        cudaMemcpy(&r, ok, 4, cudaMemcpyDeviceToHost);
        printf("V%d %-46s: %s\n", v, names[v], r ? "ISSUED+COMPLETED" : "??");
    }
    return 0;
}
