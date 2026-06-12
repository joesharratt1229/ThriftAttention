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

constexpr uint32_t INPUT_BYTES = EXP_PER_TILE * sizeof(int32_t);
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
int32_t next_power(int32_t power)
{
    const uint32_t wrapped = static_cast<uint32_t>(power + 9) & 15u;
    return static_cast<int32_t>(wrapped) - 8;
}

__device__ __forceinline__
uint32_t pow2_int_bits_ptx(int32_t power)
{
    uint32_t bits;
    asm volatile(
        "{\n\t"
        ".reg .s32 biased;\n\t"
        "add.s32 biased, %1, 127;\n\t"
        "shl.b32 %0, biased, 23;\n\t"
        "}\n"
        : "=r"(bits)
        : "r"(power));
    return bits;
}

__device__ __forceinline__
uint32_t mufu_ex2_int_bits(int32_t power)
{
    const float x = static_cast<float>(power);
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return __float_as_uint(y);
}

__device__ __forceinline__
void copy_bytes_to_smem(uint8_t* smem, const uint8_t* gmem, uint32_t bytes)
{
    for (uint32_t i = threadIdx.x; i < bytes; i += TA_WARP_SIZE) {
        smem[i] = gmem[i];
    }
}

__global__ __launch_bounds__(TA_WARP_SIZE)
void pow2_int_m16n8_kernel(const int32_t* x,
                           uint32_t* out,
                           uint64_t* ticks,
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

    const int32_t* x_smem = reinterpret_cast<const int32_t*>(smem);
    int32_t power[EXP_CHAINS_PER_THREAD];
    uint32_t acc[EXP_CHAINS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
        power[i] = x_smem[lane_id * VALUES_PER_THREAD +
                          (i % VALUES_PER_THREAD)];
        acc[i] = static_cast<uint32_t>(0x9e3779b9u + i);
    }

    __syncwarp();

    uint64_t start = read_clock64();
#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
        for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
            power[i] = next_power(power[i]);
            acc[i] ^= pow2_int_bits_ptx(power[i]);
        }
    }
    uint64_t stop = read_clock64();

    uint32_t y[VALUES_PER_THREAD] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
        y[i % VALUES_PER_THREAD] ^= acc[i];
    }

    out[lane_id * VALUES_PER_THREAD + 0] = y[0];
    out[lane_id * VALUES_PER_THREAD + 1] = y[1];
    out[lane_id * VALUES_PER_THREAD + 2] = y[2];
    out[lane_id * VALUES_PER_THREAD + 3] = y[3];

    if (lane_id == 0) {
        ticks[bid] = stop - start;
    }
}

__global__ __launch_bounds__(TA_WARP_SIZE)
void mufu_int_m16n8_kernel(const int32_t* x,
                           uint32_t* out,
                           uint64_t* ticks,
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

    const int32_t* x_smem = reinterpret_cast<const int32_t*>(smem);
    int32_t power[EXP_CHAINS_PER_THREAD];
    uint32_t acc[EXP_CHAINS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
        power[i] = x_smem[lane_id * VALUES_PER_THREAD +
                          (i % VALUES_PER_THREAD)];
        acc[i] = static_cast<uint32_t>(0x9e3779b9u + i);
    }

    __syncwarp();

    uint64_t start = read_clock64();
#pragma unroll 1
    for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
        for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
            power[i] = next_power(power[i]);
            acc[i] ^= mufu_ex2_int_bits(power[i]);
        }
    }
    uint64_t stop = read_clock64();

    uint32_t y[VALUES_PER_THREAD] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int i = 0; i < EXP_CHAINS_PER_THREAD; ++i) {
        y[i % VALUES_PER_THREAD] ^= acc[i];
    }

    out[lane_id * VALUES_PER_THREAD + 0] = y[0];
    out[lane_id * VALUES_PER_THREAD + 1] = y[1];
    out[lane_id * VALUES_PER_THREAD + 2] = y[2];
    out[lane_id * VALUES_PER_THREAD + 3] = y[3];

    if (lane_id == 0) {
        ticks[bid] = stop - start;
    }
}

struct LaunchConfig {
    int sm_count;
    int pow2_active_blocks_per_sm;
    int mufu_active_blocks_per_sm;
    int common_active_blocks_per_sm;
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

    int pow2_active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &pow2_active_blocks_per_sm, pow2_int_m16n8_kernel, TA_WARP_SIZE,
        SMEM_BYTES));
    int mufu_active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &mufu_active_blocks_per_sm, mufu_int_m16n8_kernel, TA_WARP_SIZE,
        SMEM_BYTES));
    if (pow2_active_blocks_per_sm <= 0 || mufu_active_blocks_per_sm <= 0) {
        std::fprintf(stderr, "failed to compute active blocks per SM\n");
        std::exit(EXIT_FAILURE);
    }

    const int common_active_blocks_per_sm =
        std::min(pow2_active_blocks_per_sm, mufu_active_blocks_per_sm);
    const int saturation_blocks =
        props.multiProcessorCount * common_active_blocks_per_sm;
    const bool auto_blocks = requested_blocks == 0;
    const int blocks = auto_blocks
                           ? saturation_blocks * MAX_SWEEP_WAVES
                           : requested_blocks;

    return {props.multiProcessorCount, pow2_active_blocks_per_sm,
            mufu_active_blocks_per_sm, common_active_blocks_per_sm,
            saturation_blocks, blocks, auto_blocks};
}

void setup_inputs(int blocks,
                  int32_t** d_x,
                  uint32_t** d_out,
                  uint64_t** d_pow2_ticks,
                  uint64_t** d_mufu_ticks)
{
    const size_t elems = blocks * EXP_PER_TILE;
    std::vector<int32_t> h_x(elems);

    for (size_t i = 0; i < h_x.size(); ++i) {
        h_x[i] = static_cast<int32_t>(i % 16) - 8;
    }

    CUDA_CHECK(cudaMalloc(d_x, h_x.size() * sizeof(h_x[0])));
    CUDA_CHECK(cudaMalloc(d_out, h_x.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(d_pow2_ticks, blocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(d_mufu_ticks, blocks * sizeof(uint64_t)));

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

BenchmarkResult make_result(int blocks,
                            int iters,
                            float elapsed_ms,
                            const uint64_t* d_ticks)
{
    std::vector<uint64_t> h_ticks(blocks);
    CUDA_CHECK(cudaMemcpy(h_ticks.data(), d_ticks,
                          blocks * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    const uint64_t med_ticks = median_tick(h_ticks);
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

BenchmarkResult run_pow2_benchmark(int blocks,
                                   int iters,
                                   const int32_t* d_x,
                                   uint32_t* d_out,
                                   uint64_t* d_ticks,
                                   cudaEvent_t start_event,
                                   cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(start_event));
    pow2_int_m16n8_kernel<<<blocks, TA_WARP_SIZE, SMEM_BYTES>>>(
        d_x, d_out, d_ticks, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    return make_result(blocks, iters, elapsed_ms, d_ticks);
}

BenchmarkResult run_mufu_benchmark(int blocks,
                                   int iters,
                                   const int32_t* d_x,
                                   uint32_t* d_out,
                                   uint64_t* d_ticks,
                                   cudaEvent_t start_event,
                                   cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(start_event));
    mufu_int_m16n8_kernel<<<blocks, TA_WARP_SIZE, SMEM_BYTES>>>(
        d_x, d_out, d_ticks, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    return make_result(blocks, iters, elapsed_ms, d_ticks);
}

int resident_blocks_per_sm(int run_blocks,
                           int sm_count,
                           int active_blocks_per_sm)
{
    const int blocks_per_sm =
        (run_blocks + sm_count - 1) / sm_count;
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
                     "%d blocks for one common resident wave\n",
                     max_blocks, launch.saturation_blocks);
    } else if (!launch.auto_blocks && tail_blocks != 0) {
        std::fprintf(stderr,
                     "warning: blocks=%d leaves a partial final common wave "
                     "of %d blocks; use a multiple of %d for clean "
                     "saturation\n",
                     max_blocks, tail_blocks, launch.saturation_blocks);
    }

    int32_t* d_x = nullptr;
    uint32_t* d_out = nullptr;
    uint64_t* d_pow2_ticks = nullptr;
    uint64_t* d_mufu_ticks = nullptr;

    setup_inputs(max_blocks, &d_x, &d_out, &d_pow2_ticks, &d_mufu_ticks);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    const int warmup_blocks =
        launch.auto_blocks ? launch.saturation_blocks : max_blocks;
    run_pow2_benchmark(warmup_blocks, iters, d_x, d_out, d_pow2_ticks,
                       start_event, stop_event);
    run_mufu_benchmark(warmup_blocks, iters, d_x, d_out, d_mufu_ticks,
                       start_event, stop_event);

    std::printf("iters=%d sms=%d pow2_active_blocks/sm=%d "
                "mufu_active_blocks/sm=%d common_saturation_blocks=%d "
                "tile=m16n8 exp/tile=%d mode=%s impl=int-ptx-vs-mufu\n",
                iters, launch.sm_count, launch.pow2_active_blocks_per_sm,
                launch.mufu_active_blocks_per_sm, launch.saturation_blocks,
                EXP_PER_TILE, launch.auto_blocks ? "wave-sweep" : "manual");
    std::printf("%5s %8s %13s %13s %11s %11s %13s %13s\n",
                "wave", "blocks", "pow2_cyc/t", "mufu_cyc/t",
                "pow2_G/s", "mufu_G/s", "pow2_op/cyc",
                "mufu_op/cyc");

    const auto print_result = [&](int run_blocks,
                                  const BenchmarkResult& pow2_result,
                                  const BenchmarkResult& mufu_result) {
        const int run_full_waves = run_blocks / launch.saturation_blocks;
        const int pow2_resident_blocks = resident_blocks_per_sm(
            run_blocks, launch.sm_count, launch.pow2_active_blocks_per_sm);
        const int mufu_resident_blocks = resident_blocks_per_sm(
            run_blocks, launch.sm_count, launch.mufu_active_blocks_per_sm);
        const double pow2_exp_per_cycle_per_sm =
            pow2_resident_blocks * EXP_PER_TILE /
            pow2_result.median_cycles_per_tile;
        const double mufu_exp_per_cycle_per_sm =
            mufu_resident_blocks * EXP_PER_TILE /
            mufu_result.median_cycles_per_tile;

        std::printf("%5d %8d %13.4f %13.4f %11.3f %11.3f "
                    "%13.3f %13.3f\n",
                    run_full_waves, run_blocks,
                    pow2_result.median_cycles_per_tile,
                    mufu_result.median_cycles_per_tile,
                    pow2_result.exp_per_sec * 1.0e-9,
                    mufu_result.exp_per_sec * 1.0e-9,
                    pow2_exp_per_cycle_per_sm,
                    mufu_exp_per_cycle_per_sm);
    };

    if (launch.auto_blocks) {
        for (int i = 0; i < WAVE_SWEEP_COUNT; ++i) {
            const int run_blocks = launch.saturation_blocks * WAVE_SWEEP[i];
            const BenchmarkResult pow2_result =
                run_pow2_benchmark(run_blocks, iters, d_x, d_out,
                                   d_pow2_ticks, start_event, stop_event);
            const BenchmarkResult mufu_result =
                run_mufu_benchmark(run_blocks, iters, d_x, d_out,
                                   d_mufu_ticks, start_event, stop_event);
            print_result(run_blocks, pow2_result, mufu_result);
        }
    } else {
        const BenchmarkResult pow2_result =
            run_pow2_benchmark(max_blocks, iters, d_x, d_out, d_pow2_ticks,
                               start_event, stop_event);
        const BenchmarkResult mufu_result =
            run_mufu_benchmark(max_blocks, iters, d_x, d_out, d_mufu_ticks,
                               start_event, stop_event);
        print_result(max_blocks, pow2_result, mufu_result);
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_pow2_ticks));
    CUDA_CHECK(cudaFree(d_mufu_ticks));
    return 0;
}
