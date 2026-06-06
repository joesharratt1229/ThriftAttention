#include <cstdint>
#include <cuda_runtime.h>

#include "thriftattention/sm80/cuda_common.cuh"

__device__ inline uint32_t pack4_s8(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
    return (
        uint8_t)x0 |
        ((uint32_t)(uint8_t)x1 << 8) |
        ((uint32_t)(uint8_t)x2 << 16) |
        ((uint32_t)(uint8_t)x3 << 24);
}

namespace {
    __global__ void sm80_mma_m16n8k32_s8_test_kernel(
        const int8_t* A,
        const int8_t* B,
        int32_t* D
    ) {
        const int lane = threadIdx.x;
        uint32_t a_frag[4];
        uint32_t b_frag[2];
        int32_t acc[4] = {0, 0, 0, 0};

        const int tid4 = lane & 3;
        const int groupID = lane >> 2;
        const int row0 = groupID;
        const int row1 = groupID + 8;
        const int col0 = tid4 * 2;
        const int col1 = tid4 * 2 + 1;
    
        a_frag[0] = pack4_s8(
            A[row0 * 32 + tid4 * 4 + 0],
            A[row0 * 32 + tid4 * 4 + 1],
            A[row0 * 32 + tid4 * 4 + 2],
            A[row0 * 32 + tid4 * 4 + 3]
        );

        a_frag[1] = pack4_s8(
            A[row1 * 32 + tid4 * 4 + 0],
            A[row1 * 32 + tid4 * 4 + 1],
            A[row1 * 32 + tid4 * 4 + 2],
            A[row1 * 32 + tid4 * 4 + 3]
        );

        a_frag[2] = pack4_s8(
            A[row0 * 32 + 16 + tid4 * 4 + 0],
            A[row0 * 32 + 16 + tid4 * 4 + 1],
            A[row0 * 32 + 16 + tid4 * 4 + 2],
            A[row0 * 32 + 16 + tid4 * 4 + 3]
        );

        a_frag[3] = pack4_s8(
            A[row1 * 32 + 16 + tid4 * 4 + 0],
            A[row1 * 32 + 16 + tid4 * 4 + 1],
            A[row1 * 32 + 16 + tid4 * 4 + 2],
            A[row1 * 32 + 16 + tid4 * 4 + 3]
        );
        
        b_frag[0] = pack4_s8(
            B[(tid4 * 4 + 0) * 8 + groupID],
            B[(tid4 * 4 + 1) * 8 + groupID],
            B[(tid4 * 4 + 2) * 8 + groupID],
            B[(tid4 * 4 + 3) * 8 + groupID]
        );

        b_frag[1] = pack4_s8(
            B[(16 + tid4 * 4 + 0) * 8 + groupID],
            B[(16 + tid4 * 4 + 1) * 8 + groupID],
            B[(16 + tid4 * 4 + 2) * 8 + groupID],
            B[(16 + tid4 * 4 + 3) * 8 + groupID]
        );

        ta_mma_m16n8k32_s8(a_frag, b_frag, acc);

        D[row0 * 8 + col0] = acc[0];
        D[row0 * 8 + col1] = acc[1];
        D[row1 * 8 + col0] = acc[2];
        D[row1 * 8 + col1] = acc[3];

    }
}

void sm80_mma_m16n8k32_s8_test(
    const void* A,
    const void* B,
    void* D
) {
    sm80_mma_m16n8k32_s8_test_kernel<<<1, 32>>>(
        static_cast<const int8_t*>(A),
        static_cast<const int8_t*>(B),
        static_cast<int32_t*>(D)
    );
}