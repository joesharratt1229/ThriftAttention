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

constexpr int TA_WARP_SIZE = 32;
constexpr int INPUT_VALUES_PER_THREAD = 8;
constexpr int OUTPUT_VALUES_PER_THREAD = 4;
constexpr int INPUT_VALUES_PER_BLOCK =
    TA_WARP_SIZE * INPUT_VALUES_PER_THREAD;
constexpr int OUTPUT_VALUES_PER_BLOCK =
    TA_WARP_SIZE * OUTPUT_VALUES_PER_THREAD;

constexpr uint32_t INPUT_BYTES = INPUT_VALUES_PER_BLOCK * sizeof(float);
constexpr uint32_t SMEM_BYTES = INPUT_BYTES;

constexpr int DEFAULT_ITERS = 10000;
constexpr int CVT_CHAINS_PER_THREAD = 32;
constexpr int CVT_INSTR_PER_PACK = 4;
constexpr int VALUES_PER_PACK = 8;
constexpr int CVT_INSTR_PER_ROUND =
    TA_WARP_SIZE * CVT_INSTR_PER_PACK;
constexpr int VALUES_PER_ROUND = TA_WARP_SIZE * VALUES_PER_PACK;
constexpr int WAVE_SWEEP[] = {1, 2, 4, 8};
constexpr int WAVE_SWEEP_COUNT = sizeof(WAVE_SWEEP) / sizeof(WAVE_SWEEP[0]);
constexpr int MAX_SWEEP_WAVES = 8;

__device__ __forceinline__
uint64_t read_clock64()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__
uint32_t cvt_8xf32_to_e2m1_packed(float f0,
                                   float f1,
                                   float f2,
                                   float f3,
                                   float f4,
                                   float f5,
                                   float f6,
                                   float f7)
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
void copy_bytes_to_smem(uint8_t* smem, const uint8_t* gmem, uint32_t bytes)
{
    for (uint32_t i = threadIdx.x; i < bytes; i += TA_WARP_SIZE) {
        smem[i] = gmem[i];
    }
}

__global__ __launch_bounds__(TA_WARP_SIZE)
void cvt_e2m1_kernel(const float* x,
                     uint32_t* out,
                     uint64_t* ticks,
                     int iters)
{
    const int bid = blockIdx.x;
    const int lane_id = threadIdx.x;

    x += bid * INPUT_VALUES_PER_BLOCK;
    out += bid * OUTPUT_VALUES_PER_BLOCK;

    extern __shared__ uint8_t smem[];
    copy_bytes_to_smem(smem, reinterpret_cast<const uint8_t*>(x),
                       INPUT_BYTES);
    __syncthreads();

    const float* x_smem = reinterpret_cast<const float*>(smem);
    float x_reg[INPUT_VALUES_PER_THREAD];
#pragma unroll
    for (int i = 0; i < INPUT_VALUES_PER_THREAD; ++i) {
        x_reg[i] = x_smem[lane_id * INPUT_VALUES_PER_THREAD + i];
    }

    uint32_t acc[CVT_CHAINS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < CVT_CHAINS_PER_THREAD; ++i) {
        acc[i] = 0x9e3779b9u + static_cast<uint32_t>(i * 0x10101u);
    }

    asm volatile(
        ""
        : "+f"(x_reg[0]), "+f"(x_reg[1]), "+f"(x_reg[2]), "+f"(x_reg[3]),
          "+f"(x_reg[4]), "+f"(x_reg[5]), "+f"(x_reg[6]), "+f"(x_reg[7])
        :
        : "memory");
    __syncwarp();

    uint64_t start = read_clock64();
#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
        for (int i = 0; i < CVT_CHAINS_PER_THREAD; ++i) {
            const uint32_t packed = cvt_8xf32_to_e2m1_packed(
                x_reg[(i + 0) & 7], x_reg[(i + 1) & 7],
                x_reg[(i + 2) & 7], x_reg[(i + 3) & 7],
                x_reg[(i + 4) & 7], x_reg[(i + 5) & 7],
                x_reg[(i + 6) & 7], x_reg[(i + 7) & 7]);
            acc[i] ^= packed;
        }
    }
    uint64_t stop = read_clock64();

    uint32_t y[OUTPUT_VALUES_PER_THREAD] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < CVT_CHAINS_PER_THREAD; ++i) {
        y[i & (OUTPUT_VALUES_PER_THREAD - 1)] ^= acc[i];
    }

#pragma unroll
    for (int i = 0; i < OUTPUT_VALUES_PER_THREAD; ++i) {
        out[lane_id * OUTPUT_VALUES_PER_THREAD + i] = y[i];
    }

    if (lane_id == 0) {
        ticks[bid] = stop - start;
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
        &active_blocks_per_sm, cvt_e2m1_kernel, TA_WARP_SIZE,
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

void fill_float_values(std::vector<float>& values)
{
    for (size_t i = 0; i < values.size(); ++i) {
        const int v = static_cast<int>((i * 17 + 11) % 65) - 32;
        values[i] = 0.125f * static_cast<float>(v);
    }
}

void setup_inputs(int blocks,
                  float** d_x,
                  uint32_t** d_out,
                  uint64_t** d_ticks)
{
    std::vector<float> h_x(blocks * INPUT_VALUES_PER_BLOCK);
    fill_float_values(h_x);

    CUDA_CHECK(cudaMalloc(d_x, h_x.size() * sizeof(h_x[0])));
    CUDA_CHECK(cudaMalloc(d_out,
                          blocks * OUTPUT_VALUES_PER_BLOCK *
                              sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(d_ticks, blocks * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(*d_x, h_x.data(), h_x.size() * sizeof(h_x[0]),
                          cudaMemcpyHostToDevice));
}

uint64_t median_tick(std::vector<uint64_t> ticks)
{
    std::sort(ticks.begin(), ticks.end());
    return ticks[ticks.size() / 2];
}

struct BenchmarkResult {
    double median_cycles_per_round;
    double cvt_inst_per_sec;
    double values_per_sec;
};

BenchmarkResult run_benchmark(int blocks,
                              int iters,
                              const float* d_x,
                              uint32_t* d_out,
                              uint64_t* d_ticks,
                              cudaEvent_t start_event,
                              cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(start_event));
    cvt_e2m1_kernel<<<blocks, TA_WARP_SIZE, SMEM_BYTES>>>(
        d_x, d_out, d_ticks, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

    std::vector<uint64_t> h_ticks(blocks);
    CUDA_CHECK(cudaMemcpy(h_ticks.data(), d_ticks,
                          blocks * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    const uint64_t med_ticks = median_tick(h_ticks);
    const double total_rounds =
        static_cast<double>(blocks) * iters * CVT_CHAINS_PER_THREAD;
    const double elapsed_sec = static_cast<double>(elapsed_ms) * 1.0e-3;
    const double cvt_inst_per_sec =
        elapsed_sec > 0.0
            ? total_rounds * CVT_INSTR_PER_ROUND / elapsed_sec
            : 0.0;
    const double values_per_sec =
        elapsed_sec > 0.0
            ? total_rounds * VALUES_PER_ROUND / elapsed_sec
            : 0.0;

    return {static_cast<double>(med_ticks) /
                (static_cast<double>(iters) * CVT_CHAINS_PER_THREAD),
            cvt_inst_per_sec,
            values_per_sec};
}

int resident_blocks_per_sm(int run_blocks,
                           int sm_count,
                           int active_blocks_per_sm)
{
    const int blocks_per_sm = (run_blocks + sm_count - 1) / sm_count;
    return std::min(blocks_per_sm, active_blocks_per_sm);
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

    float* d_x = nullptr;
    uint32_t* d_out = nullptr;
    uint64_t* d_ticks = nullptr;

    setup_inputs(max_blocks, &d_x, &d_out, &d_ticks);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    const int warmup_blocks =
        launch.auto_blocks ? launch.saturation_blocks : max_blocks;
    run_benchmark(warmup_blocks, iters, d_x, d_out, d_ticks,
                  start_event, stop_event);

    std::printf("iters=%d sms=%d active_blocks/sm=%d saturation_blocks=%d "
                "round=warp:%d_packs:%d_cvt:%d_values mode=%s "
                "impl=cvt.rn.satfinite.e2m1x2.f32\n",
                iters, launch.sm_count, launch.active_blocks_per_sm,
                launch.saturation_blocks, TA_WARP_SIZE, CVT_INSTR_PER_ROUND,
                VALUES_PER_ROUND, launch.auto_blocks ? "wave-sweep" : "manual");
    std::printf("%5s %8s %12s %12s %12s %14s %14s\n",
                "wave", "blocks", "cyc/round", "cvt_G/s", "val_G/s",
                "cvt/cyc/sm", "val/cyc/sm");

    const auto print_result = [&](int run_blocks,
                                  const BenchmarkResult& result) {
        const int run_full_waves = run_blocks / launch.saturation_blocks;
        const int resident_blocks = resident_blocks_per_sm(
            run_blocks, launch.sm_count, launch.active_blocks_per_sm);
        const double cvt_per_cycle_per_sm =
            resident_blocks * CVT_INSTR_PER_ROUND /
            result.median_cycles_per_round;
        const double values_per_cycle_per_sm =
            resident_blocks * VALUES_PER_ROUND /
            result.median_cycles_per_round;

        std::printf("%5d %8d %12.4f %12.3f %12.3f %14.3f %14.3f\n",
                    run_full_waves, run_blocks,
                    result.median_cycles_per_round,
                    result.cvt_inst_per_sec * 1.0e-9,
                    result.values_per_sec * 1.0e-9,
                    cvt_per_cycle_per_sm,
                    values_per_cycle_per_sm);
    };

    if (launch.auto_blocks) {
        for (int i = 0; i < WAVE_SWEEP_COUNT; ++i) {
            const int run_blocks = launch.saturation_blocks * WAVE_SWEEP[i];
            const BenchmarkResult result =
                run_benchmark(run_blocks, iters, d_x, d_out, d_ticks,
                              start_event, stop_event);
            print_result(run_blocks, result);
        }
    } else {
        const BenchmarkResult result =
            run_benchmark(max_blocks, iters, d_x, d_out, d_ticks,
                          start_event, stop_event);
        print_result(max_blocks, result);
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ticks));
    return 0;
}
