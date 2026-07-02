#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void nvfp4_quantise(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool is_bf16);

void nvfp4_quantise_permute_seq(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool inverse,
    bool is_bf16);

void nvfp4_quantise_transpose(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool is_bf16);

void nvfp4_quantise_transpose_permute_seq(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool inverse,
    bool is_bf16);

static bool check_cuda(cudaError_t err, const char* expr) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "CUDA error at %s: %s\n", expr, cudaGetErrorString(err));
    return false;
}

#define CHECK_CUDA(expr) \
    do { \
        if (!check_cuda((expr), #expr)) { \
            return false; \
        } \
    } while (0)

static void print_bytes(const char* label, const std::vector<uint8_t>& values) {
    std::printf("%s:", label);
    for (uint8_t value : values) {
        std::printf(" %02x", value);
    }
    std::printf("\n");
}

static bool has_nonzero(const std::vector<uint8_t>& values) {
    return std::any_of(values.begin(), values.end(), [](uint8_t value) {
        return value != 0;
    });
}

static bool run_one(
    const char* name,
    void (*kernel)(void*, void*, void*, int, int, int, bool),
    void* d_x,
    int bs,
    int seq_len,
    int head_dim,
    bool transpose) {
    const int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;

    const size_t packed_bytes = transpose
        ? static_cast<size_t>(bs) * head_dim * (padded_seq / 2)
        : static_cast<size_t>(bs) * seq_len * (head_dim / 2);
    const size_t scale_bytes = transpose
        ? static_cast<size_t>(bs) * head_dim * (padded_seq / 16)
        : static_cast<size_t>(bs) * seq_len * (head_dim / 16);

    uint8_t* d_packed = nullptr;
    uint8_t* d_scale = nullptr;
    CHECK_CUDA(cudaMalloc(&d_packed, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_scale, scale_bytes));
    CHECK_CUDA(cudaMemset(d_packed, 0, packed_bytes));
    CHECK_CUDA(cudaMemset(d_scale, 0, scale_bytes));

    kernel(d_x, d_packed, d_scale, bs, seq_len, head_dim, false);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const size_t sample_packed = std::min<size_t>(16, packed_bytes);
    const size_t sample_scale = std::min<size_t>(16, scale_bytes);
    std::vector<uint8_t> h_packed(sample_packed);
    std::vector<uint8_t> h_scale(sample_scale);
    CHECK_CUDA(cudaMemcpy(h_packed.data(), d_packed, sample_packed, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_scale.data(), d_scale, sample_scale, cudaMemcpyDeviceToHost));

    std::printf("%s ok\n", name);
    print_bytes("  packed", h_packed);
    print_bytes("  scale ", h_scale);

    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_scale));

    if (!has_nonzero(h_packed) || !has_nonzero(h_scale)) {
        std::fprintf(stderr, "%s produced an all-zero output sample\n", name);
        return false;
    }
    return true;
}

static bool run_one_permuted(
    const char* name,
    void (*kernel)(void*, void*, void*, int, int, int, bool, bool),
    void* d_x,
    int bs,
    int seq_len,
    int head_dim,
    bool transpose) {
    const int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;

    const size_t packed_bytes = transpose
        ? static_cast<size_t>(bs) * head_dim * (padded_seq / 2)
        : static_cast<size_t>(bs) * seq_len * (head_dim / 2);
    const size_t scale_bytes = transpose
        ? static_cast<size_t>(bs) * head_dim * (padded_seq / 16)
        : static_cast<size_t>(bs) * seq_len * (head_dim / 16);

    uint8_t* d_packed = nullptr;
    uint8_t* d_scale = nullptr;
    CHECK_CUDA(cudaMalloc(&d_packed, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_scale, scale_bytes));
    CHECK_CUDA(cudaMemset(d_packed, 0, packed_bytes));
    CHECK_CUDA(cudaMemset(d_scale, 0, scale_bytes));

    kernel(d_x, d_packed, d_scale, bs, seq_len, head_dim, false, false);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const size_t sample_packed = std::min<size_t>(16, packed_bytes);
    const size_t sample_scale = std::min<size_t>(16, scale_bytes);
    std::vector<uint8_t> h_packed(sample_packed);
    std::vector<uint8_t> h_scale(sample_scale);
    CHECK_CUDA(cudaMemcpy(h_packed.data(), d_packed, sample_packed, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_scale.data(), d_scale, sample_scale, cudaMemcpyDeviceToHost));

    std::printf("%s ok\n", name);
    print_bytes("  packed", h_packed);
    print_bytes("  scale ", h_scale);

    CHECK_CUDA(cudaFree(d_packed));
    CHECK_CUDA(cudaFree(d_scale));

    if (!has_nonzero(h_packed) || !has_nonzero(h_scale)) {
        std::fprintf(stderr, "%s produced an all-zero output sample\n", name);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int bs = 1;
    int seq_len = 128;
    int head_dim = 128;

    if (argc > 1) {
        bs = std::atoi(argv[1]);
    }
    if (argc > 2) {
        seq_len = std::atoi(argv[2]);
    }
    if (argc > 3) {
        head_dim = std::atoi(argv[3]);
    }

    if (bs <= 0 || seq_len <= 0 || (head_dim != 64 && head_dim != 128)) {
        std::fprintf(stderr, "usage: %s [bs] [seq_len] [head_dim=64|128]\n", argv[0]);
        return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice")) {
        return 1;
    }
    if (!check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties")) {
        return 1;
    }
    std::printf(
        "device=%s capability=%d.%d bs=%d seq_len=%d head_dim=%d\n",
        prop.name,
        prop.major,
        prop.minor,
        bs,
        seq_len,
        head_dim);

    const size_t input_elems = static_cast<size_t>(bs) * seq_len * head_dim;
    std::vector<half> h_x(input_elems);
    for (size_t i = 0; i < input_elems; ++i) {
        const float value = static_cast<float>(static_cast<int>(i % 257) - 128) / 17.0f;
        h_x[i] = __float2half(value);
    }

    half* d_x = nullptr;
    if (!check_cuda(cudaMalloc(&d_x, input_elems * sizeof(half)), "cudaMalloc(d_x)")) {
        return 1;
    }
    if (!check_cuda(
            cudaMemcpy(d_x, h_x.data(), input_elems * sizeof(half), cudaMemcpyHostToDevice),
            "cudaMemcpy(d_x)")) {
        cudaFree(d_x);
        return 1;
    }

    bool ok = true;
    ok = run_one("nvfp4_quantise", nvfp4_quantise, d_x, bs, seq_len, head_dim, false) && ok;
    ok = run_one_permuted(
        "nvfp4_quantise_permute_seq",
        nvfp4_quantise_permute_seq,
        d_x,
        bs,
        seq_len,
        head_dim,
        false) && ok;
    ok = run_one(
        "nvfp4_quantise_transpose",
        nvfp4_quantise_transpose,
        d_x,
        bs,
        seq_len,
        head_dim,
        true) && ok;
    ok = run_one_permuted(
        "nvfp4_quantise_transpose_permute_seq",
        nvfp4_quantise_transpose_permute_seq,
        d_x,
        bs,
        seq_len,
        head_dim,
        true) && ok;

    cudaFree(d_x);
    return ok ? 0 : 1;
}
