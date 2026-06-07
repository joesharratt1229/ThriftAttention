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
constexpr int TILE_M = 16;
constexpr int TILE_N = 8;
constexpr int VALUES_PER_THREAD = 4;
constexpr int EXP_PER_TILE = TILE_M * TILE_N;
constexpr int EXP_CHAINS_PER_THREAD = 32;
constexpr int EXP_PER_ITER = TA_WARP_SIZE * EXP_CHAINS_PER_THREAD;

constexpr uint32_t INPUT_BYTES = EXP_PER_TILE * sizeof(float);
constexpr uint32_t SMEM_BYTES = INPUT_BYTES;

constexpr int DEFAULT_ITERS = 10000;
constexpr int TILES_PER_ITER = EXP_PER_ITER / EXP_PER_TILE;
constexpr int WAVE_SWEEP[] = {1, 2, 4, 8};
constexpr int WAVE_SWEEP_COUNT = sizeof(WAVE_SWEEP) / sizeof(WAVE_SWEEP[0]);
constexpr int MAX_SWEEP_WAVES = 8;

static_assert(EXP_PER_ITER % EXP_PER_TILE == 0,
              "exp iteration must cover a whole number of tiles");

__device__ __forceinline__
uint64_t read_clock64()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__
float ex2_approx_ftz(float x)
{
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__
void copy_bytes_to_smem(uint8_t* smem, const uint8_t* gmem, uint32_t bytes)
{
    for (uint32_t i = threadIdx.x; i < bytes; i += TA_WARP_SIZE) {
        smem[i] = gmem[i];
    }
}

__global__ __launch_bounds__(TA_WARP_SIZE)
void exp_m16n8_kernel(const float* x,
                      float* out,
                      uint64_t* exp_ticks,
                      int iters)
{
    const int bid = blockIdx.x;
    const int lane_id = threadIdx.x;

    x += bid * EXP_PER_TILE;
    out += bid * EXP_PER_TILE;

    extern __shared__ uint8_t smem[];
    copy_bytes_to_smem(smem, reinterpret_cast<const uint8_t*>(x),
                       INPUT_BYTES);
    __syncthreads();

    const float* x_smem = reinterpret_cast<const float*>(smem);
    float x_reg[VALUES_PER_THREAD];
    x_reg[0] = x_smem[lane_id * VALUES_PER_THREAD + 0];
    x_reg[1] = x_smem[lane_id * VALUES_PER_THREAD + 1];
    x_reg[2] = x_smem[lane_id * VALUES_PER_THREAD + 2];
    x_reg[3] = x_smem[lane_id * VALUES_PER_THREAD + 3];

    __syncwarp();

    float chain[EXP_CHAINS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
        const float seed = 0.000244140625f * static_cast<float>(i + 1);
        chain[i] = x_reg[i % VALUES_PER_THREAD] + seed;
    }

    __syncwarp();

    uint64_t start = read_clock64();
#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
        for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
            chain[i] = ex2_approx_ftz(chain[i]);
        }
    }
    uint64_t stop = read_clock64();

    float y[VALUES_PER_THREAD] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
        y[i % VALUES_PER_THREAD] += chain[i];
    }

    out[lane_id * VALUES_PER_THREAD + 0] = y[0];
    out[lane_id * VALUES_PER_THREAD + 1] = y[1];
    out[lane_id * VALUES_PER_THREAD + 2] = y[2];
    out[lane_id * VALUES_PER_THREAD + 3] = y[3];

    if (lane_id == 0) {
        exp_ticks[bid] = stop - start;
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
        &active_blocks_per_sm, exp_m16n8_kernel, TA_WARP_SIZE,
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

void setup_inputs(int blocks,
                  float** d_x,
                  float** d_out,
                  uint64_t** d_exp_ticks)
{
    const size_t elems = blocks * EXP_PER_TILE;
    std::vector<float> h_x(elems);

    for (size_t i = 0; i < h_x.size(); ++i) {
        const float x = 0.01f * static_cast<float>((i % 17) - 8);
        h_x[i] = x * 1.4426950408889634f;
    }

    CUDA_CHECK(cudaMalloc(d_x, h_x.size() * sizeof(h_x[0])));
    CUDA_CHECK(cudaMalloc(d_out, h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_exp_ticks, blocks * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(*d_x, h_x.data(), h_x.size() * sizeof(h_x[0]),
                          cudaMemcpyHostToDevice));
}

uint64_t median_tick(std::vector<uint64_t> ticks)
{
    std::sort(ticks.begin(), ticks.end());
    return ticks[ticks.size() / 2];
}

struct BenchmarkResult {
    double median_cycles_per_tile;
    double exp_per_sec;
};

BenchmarkResult run_benchmark(int blocks,
                              int iters,
                              const float* d_x,
                              float* d_out,
                              uint64_t* d_exp_ticks,
                              cudaEvent_t start_event,
                              cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(start_event));
    exp_m16n8_kernel<<<blocks, TA_WARP_SIZE, SMEM_BYTES>>>(
        d_x, d_out, d_exp_ticks, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

    std::vector<uint64_t> h_exp_ticks(blocks);
    CUDA_CHECK(cudaMemcpy(h_exp_ticks.data(), d_exp_ticks,
                          blocks * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    const uint64_t med_ticks = median_tick(h_exp_ticks);
    const double total_tiles =
        static_cast<double>(blocks) * iters * TILES_PER_ITER;
    const double exp_per_sec =
        elapsed_ms > 0.f
            ? total_tiles * EXP_PER_TILE /
                  (static_cast<double>(elapsed_ms) * 1.0e-3)
            : 0.0;

    return {static_cast<double>(med_ticks) /
                (static_cast<double>(iters) * TILES_PER_ITER),
            exp_per_sec};
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
    float* d_out = nullptr;
    uint64_t* d_exp_ticks = nullptr;

    setup_inputs(max_blocks, &d_x, &d_out, &d_exp_ticks);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    const int warmup_blocks =
        launch.auto_blocks ? launch.saturation_blocks : max_blocks;
    run_benchmark(warmup_blocks, iters, d_x, d_out, d_exp_ticks,
                  start_event, stop_event);

    std::printf("iters=%d sms=%d active_blocks/sm=%d saturation_blocks=%d "
                "tile=m16n8 exp/tile=%d mode=%s\n",
                iters, launch.sm_count, launch.active_blocks_per_sm,
                launch.saturation_blocks, EXP_PER_TILE,
                launch.auto_blocks ? "wave-sweep" : "manual");
    std::printf("%5s %8s %13s %11s %16s\n",
                "wave", "blocks", "cyc_med/tile", "Gexp/s",
                "est_exp/cyc/sm");

    const auto print_result = [&](int run_blocks,
                                  const BenchmarkResult& result) {
        const int run_full_waves = run_blocks / launch.saturation_blocks;
        const double estimated_exp_per_cycle_per_sm =
            launch.active_blocks_per_sm * EXP_PER_TILE /
            result.median_cycles_per_tile;

        std::printf("%5d %8d %13.4f %11.3f %16.3f\n",
                    run_full_waves, run_blocks,
                    result.median_cycles_per_tile,
                    result.exp_per_sec * 1.0e-9,
                    estimated_exp_per_cycle_per_sm);
    };

    if (launch.auto_blocks) {
        for (int i = 0; i < WAVE_SWEEP_COUNT; ++i) {
            const int run_blocks = launch.saturation_blocks * WAVE_SWEEP[i];
            const BenchmarkResult result =
                run_benchmark(run_blocks, iters, d_x, d_out, d_exp_ticks,
                              start_event, stop_event);
            print_result(run_blocks, result);
        }
    } else {
        const BenchmarkResult result =
            run_benchmark(max_blocks, iters, d_x, d_out, d_exp_ticks,
                          start_event, stop_event);
        print_result(max_blocks, result);
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_exp_ticks));
    return 0;
}
