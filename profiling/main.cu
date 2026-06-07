#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                    \
    } while (0)

constexpr uint16_t BYTE_ID = 16;
constexpr uint16_t THREAD_ID = 16;
constexpr uint16_t TA_WARP_SIZE = 32;

constexpr uint64_t M = 64;
constexpr uint64_t P_K = 64;
constexpr uint64_t N = 64;
constexpr uint64_t SCALE_DIM = 16;

constexpr uint32_t A_BYTES = M * P_K * sizeof(__nv_fp4x2_e2m1);
constexpr uint32_t B_BYTES = N * P_K * sizeof(__nv_fp4x2_e2m1);
constexpr uint32_t SCALE_A_BYTES = M * SCALE_DIM * sizeof(__nv_fp8_e4m3);
constexpr uint32_t SCALE_B_BYTES = N * SCALE_DIM * sizeof(__nv_fp8_e4m3);
constexpr uint32_t A_SMEM = 0;
constexpr uint32_t B_SMEM = A_SMEM + A_BYTES;
constexpr uint32_t SCALE_A_SMEM = B_SMEM + B_BYTES;
constexpr uint32_t SCALE_B_SMEM = SCALE_A_SMEM + SCALE_A_BYTES;
constexpr uint32_t SMEM_BYTES = SCALE_B_SMEM + SCALE_B_BYTES;

constexpr int DEFAULT_ITERS = 10000;
constexpr int MMA_PER_ITER = 8;

__device__ __forceinline__
uint64_t read_clock64()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__
void mma_m16n8k64(uint32_t const a_reg[4],
                  uint32_t const b_reg[2],
                  uint32_t scale_a_reg,
                  uint32_t scale_b_reg,
                  float c_reg[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4"
        ".block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3}, "
        "{%10}, {%11, %12}, "
        "{%13}, {%14, %15};"
        : "+f"(c_reg[0]), "+f"(c_reg[1]), "+f"(c_reg[2]), "+f"(c_reg[3])
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
          "r"(b_reg[0]), "r"(b_reg[1]),
          "r"(scale_a_reg), "h"(BYTE_ID), "h"(THREAD_ID),
          "r"(scale_b_reg), "h"(BYTE_ID), "h"(THREAD_ID));
}

__device__ __forceinline__
void copy_bytes_to_smem(uint8_t* smem, uint32_t dst_offset,
                        const uint8_t* gmem, uint32_t bytes)
{
    for (uint32_t i = threadIdx.x; i < bytes; i += TA_WARP_SIZE) {
        smem[dst_offset + i] = gmem[i];
    }
}

__global__ __launch_bounds__(TA_WARP_SIZE)
void mma_m64k64n64_kernel(const __nv_fp4x2_e2m1* a,
                          const __nv_fp4x2_e2m1* b,
                          const __nv_fp8_e4m3* scale_a,
                          const __nv_fp8_e4m3* scale_b,
                          half* c,
                          uint64_t* mma_ticks,
                          uint64_t* empty_ticks,
                          int iters)
{
    const int bid = blockIdx.x;
    const int lane_id = threadIdx.x;

    a += bid * M * P_K;
    b += bid * N * P_K;
    scale_a += bid * M * SCALE_DIM;
    scale_b += bid * N * SCALE_DIM;

    extern __shared__ uint8_t smem[];

    copy_bytes_to_smem(smem, A_SMEM, reinterpret_cast<const uint8_t*>(a),
                       A_BYTES);
    copy_bytes_to_smem(smem, B_SMEM, reinterpret_cast<const uint8_t*>(b),
                       B_BYTES);
    copy_bytes_to_smem(smem, SCALE_A_SMEM,
                       reinterpret_cast<const uint8_t*>(scale_a),
                       SCALE_A_BYTES);
    copy_bytes_to_smem(smem, SCALE_B_SMEM,
                       reinterpret_cast<const uint8_t*>(scale_b),
                       SCALE_B_BYTES);
    __syncthreads();

    const uint32_t* a_smem = reinterpret_cast<const uint32_t*>(smem + A_SMEM);
    const uint32_t* b_smem = reinterpret_cast<const uint32_t*>(smem + B_SMEM);
    const uint32_t* scale_a_smem =
        reinterpret_cast<const uint32_t*>(smem + SCALE_A_SMEM);
    const uint32_t* scale_b_smem =
        reinterpret_cast<const uint32_t*>(smem + SCALE_B_SMEM);

    uint32_t a_reg[4];
    uint32_t b_reg[2];
    a_reg[0] = a_smem[lane_id * 4 + 0];
    a_reg[1] = a_smem[lane_id * 4 + 1];
    a_reg[2] = a_smem[lane_id * 4 + 2];
    a_reg[3] = a_smem[lane_id * 4 + 3];
    b_reg[0] = b_smem[lane_id * 2 + 0];
    b_reg[1] = b_smem[lane_id * 2 + 1];
    uint32_t scale_a_reg = scale_a_smem[lane_id];
    uint32_t scale_b_reg = scale_b_smem[lane_id];

    float c0[4] = {0.f, 1.f, 2.f, 3.f};
    float c1[4] = {4.f, 5.f, 6.f, 7.f};
    float c2[4] = {8.f, 9.f, 10.f, 11.f};
    float c3[4] = {12.f, 13.f, 14.f, 15.f};
    float c4[4] = {16.f, 17.f, 18.f, 19.f};
    float c5[4] = {20.f, 21.f, 22.f, 23.f};
    float c6[4] = {24.f, 25.f, 26.f, 27.f};
    float c7[4] = {28.f, 29.f, 30.f, 31.f};

    asm volatile(
        ""
        : "+r"(a_reg[0]), "+r"(a_reg[1]), "+r"(a_reg[2]), "+r"(a_reg[3]),
          "+r"(b_reg[0]), "+r"(b_reg[1]), "+r"(scale_a_reg),
          "+r"(scale_b_reg), "+f"(c0[0]), "+f"(c0[1]), "+f"(c0[2]),
          "+f"(c0[3])
        :
        : "memory");
    __syncwarp();

    uint32_t empty_dep = lane_id;
    uint64_t start = read_clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        asm volatile("add.u32 %0, %0, 1;" : "+r"(empty_dep));
    }
    uint64_t stop = read_clock64();
    const uint64_t empty = stop - start;

    start = read_clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c0);
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c1);
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c2);
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c3);
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c4);
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c5);
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c6);
        mma_m16n8k64(a_reg, b_reg, scale_a_reg, scale_b_reg, c7);
    }
    stop = read_clock64();

    if (lane_id == 0) {
        empty_ticks[bid] = empty;
        mma_ticks[bid] = stop - start;
        c[bid] = __float2half(c0[0] + c1[0] + c2[0] + c3[0] +
                              c4[0] + c5[0] + c6[0] + c7[0] +
                              static_cast<float>(empty_dep & 1));
    }
}

template <class T>
void fill_bytes(std::vector<T>& values, uint32_t seed)
{
    uint8_t* bytes = reinterpret_cast<uint8_t*>(values.data());
    const size_t nbytes = values.size() * sizeof(T);
    for (size_t i = 0; i < nbytes; ++i) {
        bytes[i] = static_cast<uint8_t>((i * 17 + seed) & 0xff);
    }
}

void setup_inputs(int blocks,
                  __nv_fp4x2_e2m1** d_a,
                  __nv_fp4x2_e2m1** d_b,
                  __nv_fp8_e4m3** d_scale_a,
                  __nv_fp8_e4m3** d_scale_b,
                  half** d_c,
                  uint64_t** d_mma_ticks,
                  uint64_t** d_empty_ticks)
{
    const size_t a_elems = blocks * M * P_K;
    const size_t b_elems = blocks * N * P_K;
    const size_t scale_a_elems = blocks * M * SCALE_DIM;
    const size_t scale_b_elems = blocks * N * SCALE_DIM;

    std::vector<__nv_fp4x2_e2m1> h_a(a_elems);
    std::vector<__nv_fp4x2_e2m1> h_b(b_elems);
    std::vector<__nv_fp8_e4m3> h_scale_a(scale_a_elems);
    std::vector<__nv_fp8_e4m3> h_scale_b(scale_b_elems);

    fill_bytes(h_a, 1);
    fill_bytes(h_b, 7);
    fill_bytes(h_scale_a, 13);
    fill_bytes(h_scale_b, 29);

    CUDA_CHECK(cudaMalloc(d_a, h_a.size() * sizeof(h_a[0])));
    CUDA_CHECK(cudaMalloc(d_b, h_b.size() * sizeof(h_b[0])));
    CUDA_CHECK(cudaMalloc(d_scale_a,
                          h_scale_a.size() * sizeof(h_scale_a[0])));
    CUDA_CHECK(cudaMalloc(d_scale_b,
                          h_scale_b.size() * sizeof(h_scale_b[0])));
    CUDA_CHECK(cudaMalloc(d_c, blocks * sizeof(half)));
    CUDA_CHECK(cudaMalloc(d_mma_ticks, blocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(d_empty_ticks, blocks * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(*d_a, h_a.data(), h_a.size() * sizeof(h_a[0]),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_b, h_b.data(), h_b.size() * sizeof(h_b[0]),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_scale_a, h_scale_a.data(),
                          h_scale_a.size() * sizeof(h_scale_a[0]),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_scale_b, h_scale_b.data(),
                          h_scale_b.size() * sizeof(h_scale_b[0]),
                          cudaMemcpyHostToDevice));
}

uint64_t min_tick(const std::vector<uint64_t>& ticks)
{
    return *std::min_element(ticks.begin(), ticks.end());
}

int main(int argc, char** argv)
{
    const int iters = argc > 1 ? std::atoi(argv[1]) : DEFAULT_ITERS;
    const int blocks = argc > 2 ? std::atoi(argv[2]) : 1;
    if (iters <= 0 || blocks <= 0) {
        std::fprintf(stderr, "usage: %s [iters=%d] [blocks=1]\n", argv[0],
                     DEFAULT_ITERS);
        return EXIT_FAILURE;
    }

    __nv_fp4x2_e2m1* d_a = nullptr;
    __nv_fp4x2_e2m1* d_b = nullptr;
    __nv_fp8_e4m3* d_scale_a = nullptr;
    __nv_fp8_e4m3* d_scale_b = nullptr;
    half* d_c = nullptr;
    uint64_t* d_mma_ticks = nullptr;
    uint64_t* d_empty_ticks = nullptr;

    setup_inputs(blocks, &d_a, &d_b, &d_scale_a, &d_scale_b, &d_c,
                 &d_mma_ticks, &d_empty_ticks);

    mma_m64k64n64_kernel<<<blocks, TA_WARP_SIZE, SMEM_BYTES>>>(
        d_a, d_b, d_scale_a, d_scale_b, d_c, d_mma_ticks, d_empty_ticks,
        iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> h_mma_ticks(blocks);
    std::vector<uint64_t> h_empty_ticks(blocks);
    CUDA_CHECK(cudaMemcpy(h_mma_ticks.data(), d_mma_ticks,
                          blocks * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_empty_ticks.data(), d_empty_ticks,
                          blocks * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    const uint64_t mma = min_tick(h_mma_ticks);
    const uint64_t empty = min_tick(h_empty_ticks);
    const double raw_cycles =
        static_cast<double>(mma) / (iters * MMA_PER_ITER);
    const double corrected_cycles =
        static_cast<double>(mma - empty) / (iters * MMA_PER_ITER);

    std::printf("iters=%d blocks=%d smem=%u bytes\n", iters, blocks,
                SMEM_BYTES);
    std::printf("raw mma region: %llu cycles\n",
                static_cast<unsigned long long>(mma));
    std::printf("empty region:   %llu cycles\n",
                static_cast<unsigned long long>(empty));
    std::printf("cycles/mma raw:       %.4f\n", raw_cycles);
    std::printf("cycles/mma corrected: %.4f\n", corrected_cycles);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_scale_a));
    CUDA_CHECK(cudaFree(d_scale_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mma_ticks));
    CUDA_CHECK(cudaFree(d_empty_ticks));
    return 0;
}
