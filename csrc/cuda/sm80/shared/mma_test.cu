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

__device__ inline uint32_t pack8_s4(
    int8_t x0, int8_t x1, int8_t x2, int8_t x3,
    int8_t x4, int8_t x5, int8_t x6, int8_t x7) {
    return (static_cast<uint32_t>(x0) & 0xf) |
           ((static_cast<uint32_t>(x1) & 0xf) << 4) |
           ((static_cast<uint32_t>(x2) & 0xf) << 8) |
           ((static_cast<uint32_t>(x3) & 0xf) << 12) |
           ((static_cast<uint32_t>(x4) & 0xf) << 16) |
           ((static_cast<uint32_t>(x5) & 0xf) << 20) |
           ((static_cast<uint32_t>(x6) & 0xf) << 24) |
           ((static_cast<uint32_t>(x7) & 0xf) << 28);
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


    __global__ void sm80_mma_m16n8k64_s4_test_kernel(
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

        a_frag[0] = pack8_s4(
            A[row0 * 64 + tid4 * 8 + 0], A[row0 * 64 + tid4 * 8 + 1],
            A[row0 * 64 + tid4 * 8 + 2], A[row0 * 64 + tid4 * 8 + 3],
            A[row0 * 64 + tid4 * 8 + 4], A[row0 * 64 + tid4 * 8 + 5],
            A[row0 * 64 + tid4 * 8 + 6], A[row0 * 64 + tid4 * 8 + 7]);
        a_frag[1] = pack8_s4(
            A[row1 * 64 + tid4 * 8 + 0], A[row1 * 64 + tid4 * 8 + 1],
            A[row1 * 64 + tid4 * 8 + 2], A[row1 * 64 + tid4 * 8 + 3],
            A[row1 * 64 + tid4 * 8 + 4], A[row1 * 64 + tid4 * 8 + 5],
            A[row1 * 64 + tid4 * 8 + 6], A[row1 * 64 + tid4 * 8 + 7]);
        a_frag[2] = pack8_s4(
            A[row0 * 64 + 32 + tid4 * 8 + 0], A[row0 * 64 + 32 + tid4 * 8 + 1],
            A[row0 * 64 + 32 + tid4 * 8 + 2], A[row0 * 64 + 32 + tid4 * 8 + 3],
            A[row0 * 64 + 32 + tid4 * 8 + 4], A[row0 * 64 + 32 + tid4 * 8 + 5],
            A[row0 * 64 + 32 + tid4 * 8 + 6], A[row0 * 64 + 32 + tid4 * 8 + 7]);
        a_frag[3] = pack8_s4(
            A[row1 * 64 + 32 + tid4 * 8 + 0], A[row1 * 64 + 32 + tid4 * 8 + 1],
            A[row1 * 64 + 32 + tid4 * 8 + 2], A[row1 * 64 + 32 + tid4 * 8 + 3],
            A[row1 * 64 + 32 + tid4 * 8 + 4], A[row1 * 64 + 32 + tid4 * 8 + 5],
            A[row1 * 64 + 32 + tid4 * 8 + 6], A[row1 * 64 + 32 + tid4 * 8 + 7]);

        b_frag[0] = pack8_s4(
            B[(tid4 * 8 + 0) * 8 + groupID], B[(tid4 * 8 + 1) * 8 + groupID],
            B[(tid4 * 8 + 2) * 8 + groupID], B[(tid4 * 8 + 3) * 8 + groupID],
            B[(tid4 * 8 + 4) * 8 + groupID], B[(tid4 * 8 + 5) * 8 + groupID],
            B[(tid4 * 8 + 6) * 8 + groupID], B[(tid4 * 8 + 7) * 8 + groupID]);
        b_frag[1] = pack8_s4(
            B[(32 + tid4 * 8 + 0) * 8 + groupID], B[(32 + tid4 * 8 + 1) * 8 + groupID],
            B[(32 + tid4 * 8 + 2) * 8 + groupID], B[(32 + tid4 * 8 + 3) * 8 + groupID],
            B[(32 + tid4 * 8 + 4) * 8 + groupID], B[(32 + tid4 * 8 + 5) * 8 + groupID],
            B[(32 + tid4 * 8 + 6) * 8 + groupID], B[(32 + tid4 * 8 + 7) * 8 + groupID]);

        ta_mma_m16n8k64_s4(a_frag, b_frag, acc);

        D[row0 * 8 + col0] = acc[0];
        D[row0 * 8 + col1] = acc[1];
        D[row1 * 8 + col0] = acc[2];
        D[row1 * 8 + col1] = acc[3];
    }

    template <int HEAD_DIM>
    __global__ void sm80_mma_int8_scores_test_kernel(
        const int8_t* Q,
        const int8_t* K,
        const float* S_Q,
        const float* S_K,
        float* scores
    ) {
        constexpr int SCALE_DIM = HEAD_DIM / 32;

        const int lane = threadIdx.x;
        const int tid4 = lane & 3;
        const int groupID = lane >> 2;
        const int row0 = groupID;
        const int row1 = groupID + 8;
        const int col0 = tid4 * 2;
        const int col1 = tid4 * 2 + 1;

        float score0 = 0.0f;
        float score1 = 0.0f;
        float score2 = 0.0f;
        float score3 = 0.0f;

        #pragma unroll
        for (int group = 0; group < SCALE_DIM; group++) {
            const int head_offset = group * 32;
            uint32_t a_frag[4];
            uint32_t b_frag[2];
            int32_t acc[4] = {0, 0, 0, 0};

            a_frag[0] = pack4_s8(
                Q[row0 * HEAD_DIM + head_offset + tid4 * 4 + 0],
                Q[row0 * HEAD_DIM + head_offset + tid4 * 4 + 1],
                Q[row0 * HEAD_DIM + head_offset + tid4 * 4 + 2],
                Q[row0 * HEAD_DIM + head_offset + tid4 * 4 + 3]
            );

            a_frag[1] = pack4_s8(
                Q[row1 * HEAD_DIM + head_offset + tid4 * 4 + 0],
                Q[row1 * HEAD_DIM + head_offset + tid4 * 4 + 1],
                Q[row1 * HEAD_DIM + head_offset + tid4 * 4 + 2],
                Q[row1 * HEAD_DIM + head_offset + tid4 * 4 + 3]
            );

            a_frag[2] = pack4_s8(
                Q[row0 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 0],
                Q[row0 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 1],
                Q[row0 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 2],
                Q[row0 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 3]
            );

            a_frag[3] = pack4_s8(
                Q[row1 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 0],
                Q[row1 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 1],
                Q[row1 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 2],
                Q[row1 * HEAD_DIM + head_offset + 16 + tid4 * 4 + 3]
            );

            b_frag[0] = pack4_s8(
                K[groupID * HEAD_DIM + head_offset + tid4 * 4 + 0],
                K[groupID * HEAD_DIM + head_offset + tid4 * 4 + 1],
                K[groupID * HEAD_DIM + head_offset + tid4 * 4 + 2],
                K[groupID * HEAD_DIM + head_offset + tid4 * 4 + 3]
            );

            b_frag[1] = pack4_s8(
                K[groupID * HEAD_DIM + head_offset + 16 + tid4 * 4 + 0],
                K[groupID * HEAD_DIM + head_offset + 16 + tid4 * 4 + 1],
                K[groupID * HEAD_DIM + head_offset + 16 + tid4 * 4 + 2],
                K[groupID * HEAD_DIM + head_offset + 16 + tid4 * 4 + 3]
            );

            ta_mma_m16n8k32_s8(a_frag, b_frag, acc);

            score0 += static_cast<float>(acc[0]) * S_Q[row0 * SCALE_DIM + group] * S_K[col0 * SCALE_DIM + group];
            score1 += static_cast<float>(acc[1]) * S_Q[row0 * SCALE_DIM + group] * S_K[col1 * SCALE_DIM + group];
            score2 += static_cast<float>(acc[2]) * S_Q[row1 * SCALE_DIM + group] * S_K[col0 * SCALE_DIM + group];
            score3 += static_cast<float>(acc[3]) * S_Q[row1 * SCALE_DIM + group] * S_K[col1 * SCALE_DIM + group];
        }

        const float softmax_scale = rsqrtf(static_cast<float>(HEAD_DIM));
        scores[row0 * 8 + col0] = score0 * softmax_scale;
        scores[row0 * 8 + col1] = score1 * softmax_scale;
        scores[row1 * 8 + col0] = score2 * softmax_scale;
        scores[row1 * 8 + col1] = score3 * softmax_scale;
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

void sm80_mma_m16n8k64_s4_test(
    const void* A,
    const void* B,
    void* D
) {
    sm80_mma_m16n8k64_s4_test_kernel<<<1, 32>>>(
        static_cast<const int8_t*>(A),
        static_cast<const int8_t*>(B),
        static_cast<int32_t*>(D)
    );
}

void sm80_mma_int8_scores_test(
    const void* Q,
    const void* K,
    const void* S_Q,
    const void* S_K,
    void* scores,
    int head_dim
) {
    if (head_dim == 64) {
        sm80_mma_int8_scores_test_kernel<64><<<1, 32>>>(
            static_cast<const int8_t*>(Q),
            static_cast<const int8_t*>(K),
            static_cast<const float*>(S_Q),
            static_cast<const float*>(S_K),
            static_cast<float*>(scores));
    } else if (head_dim == 128) {
        sm80_mma_int8_scores_test_kernel<128><<<1, 32>>>(
            static_cast<const int8_t*>(Q),
            static_cast<const int8_t*>(K),
            static_cast<const float*>(S_Q),
            static_cast<const float*>(S_K),
            static_cast<float*>(scores));
    }
}