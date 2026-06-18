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
constexpr int VALUES_PER_THREAD = 4;
constexpr int DEFAULT_ITERS = 20000;
constexpr int BODY_ITERS = 64;
constexpr int WAVE_SWEEP[] = {1, 2, 4, 8};
constexpr int WAVE_SWEEP_COUNT = sizeof(WAVE_SWEEP) / sizeof(WAVE_SWEEP[0]);
constexpr int MAX_SWEEP_WAVES = 8;

static_assert((BODY_ITERS & (BODY_ITERS - 1)) == 0,
              "BODY_ITERS should be friendly to full unrolling");

__device__ __forceinline__
uint64_t read_clock64()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__
void ld_global_to_reg(const float* addr, float reg[VALUES_PER_THREAD])
{
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(reg[0]), "=f"(reg[1]), "=f"(reg[2]), "=f"(reg[3])
        : "l"(addr)
        : "memory");
}

template <int P, int ITERS>
__device__ __forceinline__
float body_ffma(float seed, const float in[VALUES_PER_THREAD])
{
    float x0 = seed + in[0] + 0.11f + P * 0.001f;
    float x1 = seed + in[1] + 0.22f + P * 0.002f;
    float x2 = seed + in[2] + 0.33f + P * 0.003f;
    float x3 = seed + in[3] + 0.44f + P * 0.004f;
    float x4 = seed + in[0] + 0.55f + P * 0.005f;
    float x5 = seed + in[1] + 0.66f + P * 0.006f;
    float x6 = seed + in[2] + 0.77f + P * 0.007f;
    float x7 = seed + in[3] + 0.88f + P * 0.008f;

#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        if constexpr ((P & 1) == 0) {
            x0 = fmaf(x0, 1.0001f + P * 0.00001f, x1);
            x1 = fmaf(x1, 1.0002f + P * 0.00001f, x2);
            x2 = fmaf(x2, 1.0003f + P * 0.00001f, x3);
            x3 = fmaf(x3, 1.0004f + P * 0.00001f, x4);
            x4 = fmaf(x4, 1.0005f + P * 0.00001f, x5);
            x5 = fmaf(x5, 1.0006f + P * 0.00001f, x6);
            x6 = fmaf(x6, 1.0007f + P * 0.00001f, x7);
            x7 = fmaf(x7, 1.0008f + P * 0.00001f, x0);
        } else {
            x7 = fmaf(x7, 1.0008f + P * 0.00001f, x6);
            x6 = fmaf(x6, 1.0007f + P * 0.00001f, x5);
            x5 = fmaf(x5, 1.0006f + P * 0.00001f, x4);
            x4 = fmaf(x4, 1.0005f + P * 0.00001f, x3);
            x3 = fmaf(x3, 1.0004f + P * 0.00001f, x2);
            x2 = fmaf(x2, 1.0003f + P * 0.00001f, x1);
            x1 = fmaf(x1, 1.0002f + P * 0.00001f, x0);
            x0 = fmaf(x0, 1.0001f + P * 0.00001f, x7);
        }
    }

    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

template <int K, int ITERS>
__global__ __launch_bounds__(TA_WARP_SIZE)
void kernel_divergence_test(const float* input,
                            float* output,
                            uint64_t* ticks,
                            int outer_iters)
{
    const int bid = blockIdx.x;
    const int lane = threadIdx.x & (TA_WARP_SIZE - 1);
    const int elem = (bid * TA_WARP_SIZE + lane) * VALUES_PER_THREAD;

    float in[VALUES_PER_THREAD];
    ld_global_to_reg(input + elem, in);

    const int path = lane & (K - 1);
    const float seed =
        in[0] + 0.03125f * static_cast<float>(lane + 1);

    __syncwarp();
    const uint64_t start = read_clock64();

    float acc = 0.f;
#pragma unroll 1
    for (int iter = 0; iter < outer_iters; ++iter) {
        switch (path) {
            case 0: acc += body_ffma<0, ITERS>(seed + acc, in); break;
            case 1: acc += body_ffma<1, ITERS>(seed + acc, in); break;
            case 2: acc += body_ffma<2, ITERS>(seed + acc, in); break;
            case 3: acc += body_ffma<3, ITERS>(seed + acc, in); break;
            case 4: acc += body_ffma<4, ITERS>(seed + acc, in); break;
            case 5: acc += body_ffma<5, ITERS>(seed + acc, in); break;
            case 6: acc += body_ffma<6, ITERS>(seed + acc, in); break;
            case 7: acc += body_ffma<7, ITERS>(seed + acc, in); break;
            case 8: acc += body_ffma<8, ITERS>(seed + acc, in); break;
            case 9: acc += body_ffma<9, ITERS>(seed + acc, in); break;
            case 10: acc += body_ffma<10, ITERS>(seed + acc, in); break;
            case 11: acc += body_ffma<11, ITERS>(seed + acc, in); break;
            case 12: acc += body_ffma<12, ITERS>(seed + acc, in); break;
            case 13: acc += body_ffma<13, ITERS>(seed + acc, in); break;
            case 14: acc += body_ffma<14, ITERS>(seed + acc, in); break;
            case 15: acc += body_ffma<15, ITERS>(seed + acc, in); break;
            case 16: acc += body_ffma<16, ITERS>(seed + acc, in); break;
            case 17: acc += body_ffma<17, ITERS>(seed + acc, in); break;
            case 18: acc += body_ffma<18, ITERS>(seed + acc, in); break;
            case 19: acc += body_ffma<19, ITERS>(seed + acc, in); break;
            case 20: acc += body_ffma<20, ITERS>(seed + acc, in); break;
            case 21: acc += body_ffma<21, ITERS>(seed + acc, in); break;
            case 22: acc += body_ffma<22, ITERS>(seed + acc, in); break;
            case 23: acc += body_ffma<23, ITERS>(seed + acc, in); break;
            case 24: acc += body_ffma<24, ITERS>(seed + acc, in); break;
            case 25: acc += body_ffma<25, ITERS>(seed + acc, in); break;
            case 26: acc += body_ffma<26, ITERS>(seed + acc, in); break;
            case 27: acc += body_ffma<27, ITERS>(seed + acc, in); break;
            case 28: acc += body_ffma<28, ITERS>(seed + acc, in); break;
            case 29: acc += body_ffma<29, ITERS>(seed + acc, in); break;
            case 30: acc += body_ffma<30, ITERS>(seed + acc, in); break;
            case 31: acc += body_ffma<31, ITERS>(seed + acc, in); break;
        }
    }

    const uint64_t stop = read_clock64();
    output[bid * TA_WARP_SIZE + lane] = acc;

    if (lane == 0) {
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
        &active_blocks_per_sm, kernel_divergence_test<32, BODY_ITERS>,
        TA_WARP_SIZE, 0));
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
                  float** d_input,
                  float** d_output,
                  uint64_t** d_ticks)
{
    const size_t input_elems =
        static_cast<size_t>(blocks) * TA_WARP_SIZE * VALUES_PER_THREAD;
    const size_t output_elems = static_cast<size_t>(blocks) * TA_WARP_SIZE;

    std::vector<float> h_input(input_elems);
    for (size_t i = 0; i < h_input.size(); ++i) {
        h_input[i] =
            0.0009765625f * static_cast<float>((static_cast<int>(i) % 31) + 1);
    }

    CUDA_CHECK(cudaMalloc(d_input, h_input.size() * sizeof(h_input[0])));
    CUDA_CHECK(cudaMalloc(d_output, output_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_ticks, blocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(*d_input, h_input.data(),
                          h_input.size() * sizeof(h_input[0]),
                          cudaMemcpyHostToDevice));
}

uint64_t median_tick(std::vector<uint64_t> ticks)
{
    std::sort(ticks.begin(), ticks.end());
    return ticks[ticks.size() / 2];
}

struct BenchmarkResult {
    float elapsed_ms;
    double median_cycles_per_outer_iter;
};

template <int K>
BenchmarkResult run_benchmark(int blocks,
                              int iters,
                              const float* d_input,
                              float* d_output,
                              uint64_t* d_ticks,
                              cudaEvent_t start_event,
                              cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(start_event));
    kernel_divergence_test<K, BODY_ITERS><<<blocks, TA_WARP_SIZE>>>(
        d_input, d_output, d_ticks, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

    std::vector<uint64_t> h_ticks(blocks);
    CUDA_CHECK(cudaMemcpy(h_ticks.data(), d_ticks,
                          h_ticks.size() * sizeof(h_ticks[0]),
                          cudaMemcpyDeviceToHost));

    return {elapsed_ms,
            static_cast<double>(median_tick(h_ticks)) /
                static_cast<double>(iters)};
}

BenchmarkResult run_benchmark_for_k(int k,
                                    int blocks,
                                    int iters,
                                    const float* d_input,
                                    float* d_output,
                                    uint64_t* d_ticks,
                                    cudaEvent_t start_event,
                                    cudaEvent_t stop_event)
{
    switch (k) {
        case 1:
            return run_benchmark<1>(blocks, iters, d_input, d_output, d_ticks,
                                    start_event, stop_event);
        case 2:
            return run_benchmark<2>(blocks, iters, d_input, d_output, d_ticks,
                                    start_event, stop_event);
        case 4:
            return run_benchmark<4>(blocks, iters, d_input, d_output, d_ticks,
                                    start_event, stop_event);
        case 8:
            return run_benchmark<8>(blocks, iters, d_input, d_output, d_ticks,
                                    start_event, stop_event);
        case 16:
            return run_benchmark<16>(blocks, iters, d_input, d_output, d_ticks,
                                     start_event, stop_event);
        case 32:
            return run_benchmark<32>(blocks, iters, d_input, d_output, d_ticks,
                                     start_event, stop_event);
        default:
            std::fprintf(stderr, "unsupported K=%d\n", k);
            std::exit(EXIT_FAILURE);
    }
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

    float* d_input = nullptr;
    float* d_output = nullptr;
    uint64_t* d_ticks = nullptr;
    setup_inputs(max_blocks, &d_input, &d_output, &d_ticks);

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    const int warmup_blocks =
        launch.auto_blocks ? launch.saturation_blocks : max_blocks;
    run_benchmark_for_k(1, warmup_blocks, iters, d_input, d_output, d_ticks,
                        start_event, stop_event);

    std::printf("iters=%d body_iters=%d sms=%d active_blocks/sm=%d "
                "saturation_blocks=%d mode=%s\n",
                iters, BODY_ITERS, launch.sm_count,
                launch.active_blocks_per_sm, launch.saturation_blocks,
                launch.auto_blocks ? "wave-sweep" : "manual");
    std::printf("%5s %5s %8s %11s %11s %9s\n",
                "wave", "K", "blocks", "elapsed_ms", "cyc/iter",
                "vs_K1");

    const auto print_sweep = [&](int run_blocks) {
        double baseline_ms = 0.0;
        for (int k : {1, 2, 4, 8, 16, 32}) {
            const BenchmarkResult result =
                run_benchmark_for_k(k, run_blocks, iters, d_input, d_output,
                                    d_ticks, start_event, stop_event);
            if (k == 1) {
                baseline_ms = result.elapsed_ms;
            }
            const int run_full_waves = run_blocks / launch.saturation_blocks;
            const double ratio =
                baseline_ms > 0.0
                    ? static_cast<double>(result.elapsed_ms) / baseline_ms
                    : 0.0;
            std::printf("%5d %5d %8d %11.3f %11.2f %9.3f\n",
                        run_full_waves, k, run_blocks, result.elapsed_ms,
                        result.median_cycles_per_outer_iter, ratio);
        }
    };

    if (launch.auto_blocks) {
        for (int i = 0; i < WAVE_SWEEP_COUNT; ++i) {
            print_sweep(launch.saturation_blocks * WAVE_SWEEP[i]);
        }
    } else {
        print_sweep(max_blocks);
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_ticks));
    return 0;
}
