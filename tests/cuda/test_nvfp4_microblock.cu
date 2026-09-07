// Standalone regression test (no PyTorch required). From the repository root:
// nvcc -std=c++17 -O3 -use_fast_math -arch=sm_120a -I csrc/include tests/cuda/test_nvfp4_microblock.cu csrc/cuda/sm120/nvfp4/fp4_attention.cu -o /tmp/test_nvfp4_microblock
// /tmp/test_nvfp4_microblock
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Attention = void (*)(const void*, const void*, const void*, const void*,
                          const void*, const void*, void*, int, int, int, int,
                          int, int, int, bool, bool, bool);
void fp4_attention_causal_nvfp4(const void*, const void*, const void*, const void*,
                              const void*, const void*, void*, int, int, int, int,
                              int, int, int, bool, bool, bool);
void fp4_attention_noncausal_nvfp4(const void*, const void*, const void*, const void*,
                                 const void*, const void*, void*, int, int, int, int,
                                 int, int, int, bool, bool, bool);

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}
struct DeviceBuffer {
    void* ptr;
    explicit DeviceBuffer(size_t bytes) { check(cudaMalloc(&ptr, bytes)); }
    explicit DeviceBuffer(const std::vector<uint8_t>& host) : DeviceBuffer(host.size()) {
        check(cudaMemcpy(ptr, host.data(), host.size(), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
constexpr float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
float decode(int code) { return levels[code & 7] * ((code & 8) ? -1 : 1); }
float round_fp4(float x) {
    int best = 0;
    for (int i = 1; i < 8; ++i) {
        float distance = std::abs(x - levels[i]);
        float previous = std::abs(x - levels[best]);
        if (distance < previous || (distance == previous && (i & 1) == 0)) best = i;
    }
    return levels[best];
}
void pack(std::vector<uint8_t>& bytes, size_t i, int code) {
    bytes[i / 2] |= uint8_t(code << (4 * (i % 2)));
}
int q_code(int head, int row) {
    constexpr int codes[] = {7, 4, 2, 0, 12, 6};
    return codes[(row + head) % 6];
}
int k_code(int batch, int key) {
    // Different maxima in all four microblocks and larger maxima in later
    // tiles. The final weak block is > log(24) below the running maximum.
    constexpr int codes[] = {3, 1, 2, 4, 1, 5, 2, 6, 2, 7, 4, 1};
    return std::min(7, codes[key / 16] + ((key + batch) % 3 == 0));
}
int v_code(int batch, int key, int col) {
    return ((key * 7 + col * 3 + batch) % 8) | (((key + col) % 3 == 0) ? 8 : 0);
}

// Dense scalar reference in logical key order. It intentionally does not
// share warp reductions, scale packing, or MMA layouts with the kernel.
std::vector<float> reference(int dim, int head, int row, bool causal, bool approx) {
    constexpr int kv_len = 192;
    const int batch = head / 2;
    const float scale = (1.f / std::sqrt(float(dim))) * (approx ? 1.4426950408889634f : 1.f);
    std::vector<float> out(dim, 0.f);
    float m = -INFINITY, denominator = 0;
    for (int start = 0; start < kv_len && (!causal || start <= row); start += 64) {
        float scores[64], block_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        for (int j = 0; j < 64; ++j) {
            int key = start + j;
            scores[j] = causal && key > row ? -INFINITY
                : 16.f * decode(q_code(head, row)) * decode(k_code(batch, key)) * .125f * scale;
            block_max[j / 16] = std::max(block_max[j / 16], scores[j]);
        }
        float next_m = *std::max_element(block_max, block_max + 4);
        if (approx) next_m = std::nearbyint(next_m + .5f);
        next_m = std::max(m, next_m);
        float rescale = approx ? std::exp2(m - next_m) : std::exp(m - next_m);
        for (float& x : out) x *= rescale;
        denominator *= rescale;
        m = next_m;
        for (int g = 0; g < 4; ++g) {
            if (!std::isfinite(block_max[g])) continue;
            float b = approx ? std::nearbyint(block_max[g] + .5f) : block_max[g];
            float sf = float(__nv_fp8_e4m3(448.f * (approx ? std::exp2(b - m) : std::exp(b - m))));
            for (int j = g * 16; j < (g + 1) * 16; ++j) {
                if (!std::isfinite(scores[j])) continue;
                float code_input = approx ? 4.f * std::exp2(std::nearbyint(scores[j] - b))
                                          : 6.f * std::exp(scores[j] - b);
                float weight = round_fp4(code_input) * sf;
                if (!approx) weight /= 448.f * 6.f;
                denominator += approx ? weight : std::exp(scores[j] - m);
                for (int col = 0; col < dim; ++col)
                    out[col] += weight * decode(v_code(batch, start + j, col)) * .5f;
            }
        }
    }
    for (float& x : out) x /= denominator;
    return out;
}

void run_case(int dim, bool causal, bool approx, bool bf16) {
    constexpr int batch = 2, q_heads = 2, kv_heads = 1, q_len = 192, kv_len = 192;
    constexpr int capacity = 256; // Exercise padded KV strides as well.
    std::vector<uint8_t> q(batch * q_heads * q_len * dim / 2);
    std::vector<uint8_t> k(batch * kv_heads * capacity * dim / 2);
    std::vector<uint8_t> v(batch * kv_heads * capacity * dim / 2);
    std::vector<uint8_t> sq(batch * q_heads * q_len * dim / 16, 0x38); // 1
    std::vector<uint8_t> sk(batch * kv_heads * capacity * dim / 16, 0x20); // 1/8
    std::vector<uint8_t> sv(batch * kv_heads * capacity * dim / 16, 0x30); // 1/2
    for (int h = 0; h < batch * q_heads; ++h)
        for (int row = 0; row < q_len; ++row)
            for (int col = 0; col < 16; ++col)
                pack(q, (h * q_len + row) * dim + col, q_code(h, row));
    for (int h = 0; h < batch * kv_heads; ++h)
        for (int physical = 0; physical < kv_len; ++physical) {
            int x = physical % 32;
            int logical = (physical / 32) * 32 + (x / 8) * 2 + ((x % 8) / 2) * 8 + x % 2;
            for (int col = 0; col < 16; ++col)
                pack(k, (h * capacity + physical) * dim + col, k_code(h, logical));
            for (int col = 0; col < dim; ++col)
                pack(v, (h * dim + col) * capacity + physical, v_code(h, physical, col));
        }
    DeviceBuffer dq(q), dk(k), dv(v), dsq(sq), dsk(sk), dsv(sv);
    size_t count = batch * q_heads * q_len * dim;
    DeviceBuffer dout(count * 2);
    Attention attention = causal ? fp4_attention_causal_nvfp4 : fp4_attention_noncausal_nvfp4;
    std::vector<uint16_t> first(count), actual(count);
    float worst = 0;
    for (bool legacy_flag : {false, true}) {
        attention(dq.ptr, dk.ptr, dv.ptr, dsq.ptr, dsk.ptr, dsv.ptr, dout.ptr,
                  batch * q_heads, q_len, kv_len, capacity, q_heads, kv_heads,
                  dim, bf16, approx, legacy_flag);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(actual.data(), dout.ptr, count * 2, cudaMemcpyDeviceToHost));
        if (legacy_flag) {
            if (first != actual) { std::fprintf(stderr, "legacy flag changed output\n"); std::exit(1); }
            continue;
        }
        first = actual;
        for (int h = 0; h < batch * q_heads; ++h)
            for (int row = 0; row < q_len; ++row) {
                auto expected = reference(dim, h, row, causal, approx);
                for (int col = 0; col < dim; ++col) {
                    size_t i = (h * q_len + row) * dim + col;
                    float got = bf16 ? float(reinterpret_cast<__nv_bfloat16*>(actual.data())[i])
                                     : float(reinterpret_cast<half*>(actual.data())[i]);
                    float error = std::abs(got - expected[col]);
                    worst = std::max(worst, error);
                    float tolerance = .00015f + (bf16 ? .004f : .0006f) * std::abs(expected[col]);
                    if (!std::isfinite(got) || error > tolerance) {
                        std::fprintf(stderr, "d=%d causal=%d approx=%d bf16=%d h=%d row=%d col=%d got=%g expected=%g\n",
                                     dim, causal, approx, bf16, h, row, col, got, expected[col]);
                        std::exit(1);
                    }
                }
            }
    }
    std::printf("PASS d=%d causal=%d approx=%d bf16=%d max_error=%g\n", dim, causal, approx, bf16, worst);
}
int main() {
    for (int dim : {64, 128})
        for (bool causal : {false, true})
            for (bool approx : {false, true})
                for (bool bf16 : {false, true}) run_case(dim, causal, approx, bf16);
}
