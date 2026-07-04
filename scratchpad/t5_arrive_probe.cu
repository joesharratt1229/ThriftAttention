// T5: is the remote cluster arrive (mapa + mbarrier.arrive.release.cluster.
// shared::cluster) legal against a barrier the leader is try_wait-spinning
// on?  Replicates the kernel's P0Ready pattern: count = 256 (128 lanes per
// CTA arrive once), leader waits parity 0.
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

__global__ void __cluster_dims__(2, 1, 1) t5_kernel(int variant, uint32_t* ok)
{
    __shared__ __align__(8) uint64_t mbar_storage;
    const uint32_t rank = cluster_rank();
    const uint32_t mbar = (uint32_t)__cvta_generic_to_shared(&mbar_storage);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" :: "r"(mbar));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    cluster_sync();

    // 128 lanes per CTA arrive on the LEADER's barrier.
    if (threadIdx.x < 128) {
        uint32_t remote;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(remote) : "r"(mbar), "r"(0u));
        if (variant == 0) {
            asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                         :: "r"(remote) : "memory");
        } else {
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0], 1;"
                         :: "r"(remote) : "memory");
        }
    }

    if (rank == 0 && threadIdx.x == 128) {   // separate warp spins like mma warp
        asm volatile(
            "{\n\t.reg .pred P1;\n\t"
            "WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], 0, %1;\n\t"
            "@P1 bra.uni DONE;\n\t"
            "bra.uni WAIT;\n\t"
            "DONE:\n\t}"
            :: "r"(mbar), "r"(0x989680));
        atomicExch(ok, 1);
    }
    __syncthreads();
    cluster_sync();
}

int main()
{
    uint32_t* ok;
    cudaMalloc(&ok, 4);
    const char* names[] = {
        "arrive.release.cluster.shared::cluster (kernel form)",
        "arrive.shared::cluster (DSL-emitted form)",
    };
    for (int v = 0; v < 2; v++) {
        cudaMemset(ok, 0, 4);
        t5_kernel<<<2, 256>>>(v, ok);
        bool done = false;
        for (int ms = 0; ms < 5000 && !done; ms += 50) {
            cudaError_t q = cudaStreamQuery(0);
            if (q == cudaSuccess) {
                done = true;
            } else if (q != cudaErrorNotReady) {
                printf("V%d %-52s: TRAP %s\n", v, names[v], cudaGetErrorString(q));
                return 1;
            } else {
                usleep(50 * 1000);
            }
        }
        if (!done) {
            printf("V%d %-52s: HANG\n", v, names[v]);
            return 2;
        }
        uint32_t r;
        cudaMemcpy(&r, ok, 4, cudaMemcpyDeviceToHost);
        printf("V%d %-52s: %s\n", v, names[v], r ? "PASS" : "??");
    }
    return 0;
}
