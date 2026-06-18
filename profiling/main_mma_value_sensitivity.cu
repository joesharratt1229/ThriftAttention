#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr uint16_t TA_WARP_SIZE = 32;

constexpr uint64_t M = 16;
constexpr uint64_t K = 16;
constexpr uint64_t N = 8;

constexpr uint32_t A_BYTES = M * K * sizeof(half);
constexpr uint32_t B_BYTES = N * K * sizeof(half);
constexpr uint32_t A_SMEM = 0;
constexpr uint32_t B_SMEM = A_SMEM + A_BYTES;
constexpr uint32_t SMEM_BYTES = B_SMEM + B_BYTES;

constexpr int DEFAULT_ITERS = 10000;
constexpr int MMA_PER_ITER = 8;
constexpr int WAVE_SWEEP[] = {1, 2, 4, 8};
constexpr int WAVE_SWEEP_COUNT = sizeof(WAVE_SWEEP) / sizeof(WAVE_SWEEP[0]);
constexpr int MAX_SWEEP_WAVES = 8;

constexpr double OPS_PER_MMA = 16.0 * 8.0 * 16.0 * 2.0;

__device__ __forceinline__
uint64_t read_clock64()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__
void mma_m16n8k16(const uint32_t a_reg[4],
                  const uint32_t b_reg[2],
                  float c_reg[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(c_reg[0]), "=f"(c_reg[1]), "=f"(c_reg[2]), "=f"(c_reg[3])
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
          "r"(b_reg[0]), "r"(b_reg[1]),
          "f"(c_reg[0]), "f"(c_reg[1]), "f"(c_reg[2]), "f"(c_reg[3]));
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
void mma_m16k16n8_kernel(const half* a,
                         const half* b,
                         half* c,
                         uint64_t* mma_ticks,
                         int iters)
{
    const int bid = blockIdx.x;
    const int lane_id = threadIdx.x;

    a += bid * M * K;
    b += bid * N * K;

    extern __shared__ uint8_t smem[];

    copy_bytes_to_smem(smem, A_SMEM, reinterpret_cast<const uint8_t*>(a),
                       A_BYTES);
    copy_bytes_to_smem(smem, B_SMEM, reinterpret_cast<const uint8_t*>(b),
                       B_BYTES);
    __syncthreads();

    const uint32_t* a_smem = reinterpret_cast<const uint32_t*>(smem + A_SMEM);
    const uint32_t* b_smem = reinterpret_cast<const uint32_t*>(smem + B_SMEM);

    uint32_t a_reg[4];
    uint32_t b_reg[2];
    a_reg[0] = a_smem[lane_id * 4 + 0];
    a_reg[1] = a_smem[lane_id * 4 + 1];
    a_reg[2] = a_smem[lane_id * 4 + 2];
    a_reg[3] = a_smem[lane_id * 4 + 3];
    b_reg[0] = b_smem[lane_id * 2 + 0];
    b_reg[1] = b_smem[lane_id * 2 + 1];

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
          "+r"(b_reg[0]), "+r"(b_reg[1]), "+f"(c0[0]), "+f"(c0[1]),
          "+f"(c0[2]), "+f"(c0[3])
        :
        : "memory");
    __syncwarp();

    uint64_t start = read_clock64();
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        mma_m16n8k16(a_reg, b_reg, c0);
        mma_m16n8k16(a_reg, b_reg, c1);
        mma_m16n8k16(a_reg, b_reg, c2);
        mma_m16n8k16(a_reg, b_reg, c3);
        mma_m16n8k16(a_reg, b_reg, c4);
        mma_m16n8k16(a_reg, b_reg, c5);
        mma_m16n8k16(a_reg, b_reg, c6);
        mma_m16n8k16(a_reg, b_reg, c7);
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
        &active_blocks_per_sm, mma_m16k16n8_kernel, TA_WARP_SIZE,
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

enum class ValuePattern {
    Zero,
    One,
    Pow2,
    Random
};

const char* pattern_name(ValuePattern pattern)
{
    switch (pattern) {
        case ValuePattern::Zero: return "zero";
        case ValuePattern::One: return "one";
        case ValuePattern::Pow2: return "pow2";
        case ValuePattern::Random: return "random";
    }
    return "unknown";
}

uint32_t lcg_next(uint32_t x)
{
    return x * 1664525u + 1013904223u;
}

void fill_half_values(std::vector<half>& values, ValuePattern pattern,
                      uint32_t seed)
{
    constexpr float pow2_values[] = {
        0.0625f, 0.125f, 0.25f, 0.5f,
        1.0f, 2.0f, 4.0f, 8.0f
    };

    uint32_t state = seed;
    for (size_t i = 0; i < values.size(); ++i) {
        float value = 0.f;
        switch (pattern) {
            case ValuePattern::Zero:
                value = 0.f;
                break;
            case ValuePattern::One:
                value = 1.f;
                break;
            case ValuePattern::Pow2:
                value = pow2_values[(i + seed) & 7];
                break;
            case ValuePattern::Random:
                state = lcg_next(state);
                value = 0.03125f * static_cast<float>((state & 0x3f) + 1);
                if (state & 0x40) {
                    value = -value;
                }
                break;
        }
        values[i] = __float2half(value);
    }
}

void allocate_buffers(int blocks,
                      half** d_a,
                      half** d_b,
                      half** d_c,
                      uint64_t** d_mma_ticks)
{
    const size_t a_elems = blocks * M * K;
    const size_t b_elems = blocks * N * K;

    CUDA_CHECK(cudaMalloc(d_a, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(d_b, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(d_c, blocks * sizeof(half)));
    CUDA_CHECK(cudaMalloc(d_mma_ticks, blocks * sizeof(uint64_t)));
}

void upload_inputs(int blocks, ValuePattern pattern, half* d_a, half* d_b)
{
    const size_t a_elems = blocks * M * K;
    const size_t b_elems = blocks * N * K;

    std::vector<half> h_a(a_elems);
    std::vector<half> h_b(b_elems);
    fill_half_values(h_a, pattern, 1);
    fill_half_values(h_b, pattern, 7);

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(h_a[0]),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(h_b[0]),
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
                              const half* d_a,
                              const half* d_b,
                              half* d_c,
                              uint64_t* d_mma_ticks,
                              cudaEvent_t start_event,
                              cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(start_event));
    mma_m16k16n8_kernel<<<blocks, TA_WARP_SIZE, SMEM_BYTES>>>(
        d_a, d_b, d_c, d_mma_ticks, iters);
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

    half* d_a = nullptr;
    half* d_b = nullptr;
    half* d_c = nullptr;
    uint64_t* d_mma_ticks = nullptr;
    allocate_buffers(max_blocks, &d_a, &d_b, &d_c, &d_mma_ticks);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    const int warmup_blocks =
        launch.auto_blocks ? launch.saturation_blocks : max_blocks;
    upload_inputs(max_blocks, ValuePattern::Random, d_a, d_b);
    run_benchmark(warmup_blocks, iters, d_a, d_b, d_c, d_mma_ticks,
                  start_event, stop_event);

    std::printf("iters=%d sms=%d active_blocks/sm=%d saturation_blocks=%d "
                "mode=%s\n",
                iters, launch.sm_count, launch.active_blocks_per_sm,
                launch.saturation_blocks,
                launch.auto_blocks ? "wave-sweep" : "manual");
    std::printf("%5s %9s %8s %11s %11s %15s %9s\n",
                "wave", "pattern", "blocks", "cyc_med", "TOPS/s",
                "est_ops/cyc/sm", "vs_rand");

    const ValuePattern patterns[] = {
        ValuePattern::Random,
        ValuePattern::Zero,
        ValuePattern::One,
        ValuePattern::Pow2,
    };

    const auto print_result = [&](int run_blocks, ValuePattern pattern,
                                  const BenchmarkResult& result,
                                  double random_cycles) {
        const int run_full_waves = run_blocks / launch.saturation_blocks;
        const double estimated_ops_per_cycle_per_sm =
            launch.active_blocks_per_sm * OPS_PER_MMA /
            result.median_cycles_per_mma;
        const double ratio = result.median_cycles_per_mma / random_cycles;

        std::printf("%5d %9s %8d %11.4f %11.3f %15.3f %9.4f\n",
                    run_full_waves, pattern_name(pattern), run_blocks,
                    result.median_cycles_per_mma,
                    result.ops_per_sec * 1.0e-12,
                    estimated_ops_per_cycle_per_sm, ratio);
    };

    const auto run_pattern_sweep = [&](int run_blocks) {
        double random_cycles = 0.0;
        BenchmarkResult results[4];
        for (int i = 0; i < 4; ++i) {
            upload_inputs(max_blocks, patterns[i], d_a, d_b);
            results[i] = run_benchmark(run_blocks, iters, d_a, d_b, d_c,
                                       d_mma_ticks, start_event, stop_event);
            if (patterns[i] == ValuePattern::Random) {
                random_cycles = results[i].median_cycles_per_mma;
            }
        }
        for (int i = 0; i < 4; ++i) {
            print_result(run_blocks, patterns[i], results[i], random_cycles);
        }
    };

    if (launch.auto_blocks) {
        for (int i = 0; i < WAVE_SWEEP_COUNT; ++i) {
            run_pattern_sweep(launch.saturation_blocks * WAVE_SWEEP[i]);
        }
    } else {
        run_pattern_sweep(max_blocks);
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mma_ticks));
    return 0;
}
