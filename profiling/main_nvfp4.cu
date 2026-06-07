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

constexpr uint16_t BYTE_ID = 0;
constexpr uint16_t THREAD_ID = 0;
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
constexpr int WAVE_SWEEP[] = {1, 2, 4, 8};
constexpr int WAVE_SWEEP_COUNT = sizeof(WAVE_SWEEP) / sizeof(WAVE_SWEEP[0]);
constexpr int MAX_SWEEP_WAVES = 8;

constexpr double OPS_PER_MMA = 16.0 * 8.0 * 64.0 * 2.0;

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
        "{%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, "
        "{%17}, {%18, %19};"
        : "=f"(c_reg[0]), "=f"(c_reg[1]), "=f"(c_reg[2]), "=f"(c_reg[3])
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
          "r"(b_reg[0]), "r"(b_reg[1]),
          "f"(c_reg[0]), "f"(c_reg[1]), "f"(c_reg[2]), "f"(c_reg[3]),
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

    uint64_t start = read_clock64();
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
    uint64_t stop = read_clock64();

    if (lane_id == 0) {
        mma_ticks[bid] = stop - start;
        c[bid] = __float2half(c0[0] + c1[0] + c2[0] + c3[0] +
                              c4[0] + c5[0] + c6[0] + c7[0]);
    }
}

struct LaunchConfig {
    int sm_count;
    int active_blocks_per_sm;
    int saturation_blocks;
    int blocks;
    bool auto_blocks;
};

LaunchConfig configure_launch(int requested_blocks)
{
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    int active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, mma_m64k64n64_kernel, TA_WARP_SIZE,
        SMEM_BYTES));
    if (active_blocks_per_sm <= 0) {
        std::fprintf(stderr, "failed to compute active blocks per SM\n");
        std::exit(EXIT_FAILURE);
    }

    const int saturation_blocks =
        props.multiProcessorCount * active_blocks_per_sm;
    const bool auto_blocks = requested_blocks == 0;
    const int blocks = auto_blocks
                           ? saturation_blocks * MAX_SWEEP_WAVES
                           : requested_blocks;

    return {props.multiProcessorCount, active_blocks_per_sm,
            saturation_blocks, blocks, auto_blocks};
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
                  uint64_t** d_mma_ticks)
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

uint64_t median_tick(std::vector<uint64_t> ticks)
{
    std::sort(ticks.begin(), ticks.end());
    return ticks[ticks.size() / 2];
}

struct BenchmarkResult {
    double median_cycles_per_mma;
    double ops_per_sec;
};

BenchmarkResult run_benchmark(int blocks,
                              int iters,
                              const __nv_fp4x2_e2m1* d_a,
                              const __nv_fp4x2_e2m1* d_b,
                              const __nv_fp8_e4m3* d_scale_a,
                              const __nv_fp8_e4m3* d_scale_b,
                              half* d_c,
                              uint64_t* d_mma_ticks,
                              cudaEvent_t start_event,
                              cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(start_event));
    mma_m64k64n64_kernel<<<blocks, TA_WARP_SIZE, SMEM_BYTES>>>(
        d_a, d_b, d_scale_a, d_scale_b, d_c, d_mma_ticks, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

    std::vector<uint64_t> h_mma_ticks(blocks);
    CUDA_CHECK(cudaMemcpy(h_mma_ticks.data(), d_mma_ticks,
                          blocks * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    const uint64_t med_mma = median_tick(h_mma_ticks);
    const double denom = static_cast<double>(iters) * MMA_PER_ITER;
    const double total_mma_insts =
        static_cast<double>(blocks) * iters * MMA_PER_ITER;
    const double mma_inst_per_sec =
        elapsed_ms > 0.f
            ? total_mma_insts / (static_cast<double>(elapsed_ms) * 1.0e-3)
            : 0.0;

    return {static_cast<double>(med_mma) / denom,
            mma_inst_per_sec * OPS_PER_MMA};
}

int main(int argc, char** argv)
{
    const int iters = argc > 1 ? std::atoi(argv[1]) : DEFAULT_ITERS;
    const int requested_blocks = argc > 2 ? std::atoi(argv[2]) : 0;
    if (iters <= 0 || requested_blocks < 0) {
        std::fprintf(stderr,
                     "usage: %s [iters=%d] [blocks=auto|positive]\n",
                     argv[0], DEFAULT_ITERS);
        return EXIT_FAILURE;
    }

    const LaunchConfig launch = configure_launch(requested_blocks);
    const int max_blocks = launch.blocks;
    const int tail_blocks = max_blocks % launch.saturation_blocks;

    if (!launch.auto_blocks && max_blocks < launch.saturation_blocks) {
        std::fprintf(stderr,
                     "warning: blocks=%d underfills the GPU; need at least "
                     "%d blocks for one resident wave\n",
                     max_blocks, launch.saturation_blocks);
    } else if (!launch.auto_blocks && tail_blocks != 0) {
        std::fprintf(stderr,
                     "warning: blocks=%d leaves a partial final wave of %d "
                     "blocks; use a multiple of %d for clean saturation\n",
                     max_blocks, tail_blocks, launch.saturation_blocks);
    }

    __nv_fp4x2_e2m1* d_a = nullptr;
    __nv_fp4x2_e2m1* d_b = nullptr;
    __nv_fp8_e4m3* d_scale_a = nullptr;
    __nv_fp8_e4m3* d_scale_b = nullptr;
    half* d_c = nullptr;
    uint64_t* d_mma_ticks = nullptr;

    setup_inputs(max_blocks, &d_a, &d_b, &d_scale_a, &d_scale_b, &d_c,
                 &d_mma_ticks);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    const int warmup_blocks =
        launch.auto_blocks ? launch.saturation_blocks : max_blocks;
    run_benchmark(warmup_blocks, iters, d_a, d_b, d_scale_a, d_scale_b, d_c,
                  d_mma_ticks, start_event, stop_event);

    std::printf("iters=%d sms=%d active_blocks/sm=%d saturation_blocks=%d "
                "mode=%s\n",
                iters, launch.sm_count, launch.active_blocks_per_sm,
                launch.saturation_blocks,
                launch.auto_blocks ? "wave-sweep" : "manual");
    std::printf("%5s %8s %11s %11s %15s\n",
                "wave", "blocks", "cyc_med", "TOPS/s",
                "est_ops/cyc/sm");

    const auto print_result = [&](int run_blocks,
                                  const BenchmarkResult& result) {
        const int run_full_waves = run_blocks / launch.saturation_blocks;
        const double estimated_ops_per_cycle_per_sm =
            launch.active_blocks_per_sm * OPS_PER_MMA /
            result.median_cycles_per_mma;

        std::printf("%5d %8d %11.4f %11.3f %15.3f\n",
                    run_full_waves, run_blocks,
                    result.median_cycles_per_mma,
                    result.ops_per_sec * 1.0e-12,
                    estimated_ops_per_cycle_per_sm);
    };

    if (launch.auto_blocks) {
        for (int i = 0; i < WAVE_SWEEP_COUNT; ++i) {
            const int run_blocks = launch.saturation_blocks * WAVE_SWEEP[i];
            const BenchmarkResult result =
                run_benchmark(run_blocks, iters, d_a, d_b, d_scale_a,
                              d_scale_b, d_c, d_mma_ticks, start_event,
                              stop_event);
            print_result(run_blocks, result);
        }
    } else {
        const BenchmarkResult result =
            run_benchmark(max_blocks, iters, d_a, d_b, d_scale_a, d_scale_b,
                          d_c, d_mma_ticks, start_event, stop_event);
        print_result(max_blocks, result);
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_scale_a));
    CUDA_CHECK(cudaFree(d_scale_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mma_ticks));
    return 0;
}
