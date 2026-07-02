#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_bf16.h>
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

void nvfp4_quantise_transpose(
    void* X_raw,
    void* X_fp4_raw,
    void* X_scale_raw,
    int bs,
    int seq_len,
    int head_dim,
    bool is_bf16);

extern "C" cudaError_t nvfp4_sm100_pack_q_sf_atoms(
    const uint8_t* sf_q,
    uint8_t* sf_q_atoms,
    int batch,
    int num_q_heads,
    int q_len,
    cudaStream_t stream);

extern "C" cudaError_t nvfp4_sm100_pack_k_sf_atoms(
    const uint8_t* sf_k,
    uint8_t* sf_k_atoms,
    int batch,
    int num_kv_heads,
    int kv_len,
    cudaStream_t stream);

extern "C" cudaError_t nvfp4_sm100_pack_v_sf_atoms(
    const uint8_t* sf_v_t,
    uint8_t* sf_v_atoms,
    int batch,
    int num_kv_heads,
    int kv_len,
    cudaStream_t stream);

extern "C" cudaError_t nvfp4_sm100_attention_launch(
    const void* q,
    const void* k,
    const void* v_t,
    const uint8_t* sf_q_atoms,
    const uint8_t* sf_k_atoms,
    const uint8_t* sf_v_atoms,
    __nv_bfloat16* o,
    int batch,
    int q_len,
    int kv_len,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale_log2,
    float v_descale,
    cudaStream_t stream);

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

static void fill_input(std::vector<half>& x, int seed) {
    for (size_t i = 0; i < x.size(); ++i) {
        const int lane = static_cast<int>((i + 17 * seed) % 257) - 128;
        const float wave = static_cast<float>((static_cast<int>(i / 37) + seed) % 11) - 5.0f;
        x[i] = __float2half(static_cast<float>(lane) / 23.0f + wave / 19.0f);
    }
}

static bool has_nonzero_u16(const std::vector<uint16_t>& values) {
    return std::any_of(values.begin(), values.end(), [](uint16_t value) {
        return value != 0;
    });
}

static void print_u16_sample(const char* label, const std::vector<uint16_t>& values) {
    std::printf("%s:", label);
    for (uint16_t value : values) {
        std::printf(" %04x", value);
    }
    std::printf("\n");
}

static bool run_attention_smoke(
    int batch,
    int q_len,
    int kv_len,
    int num_q_heads,
    int num_kv_heads) {
    constexpr int head_dim = 128;
    constexpr int q_tile_rows = 256;
    constexpr int kv_tile = 64;
    constexpr int sf_atom_bytes = 512;
    constexpr float log2e = 1.4426950408889634f;

    if (batch <= 0 || q_len <= 0 || kv_len <= 0 || num_q_heads <= 0 || num_kv_heads <= 0) {
        std::fprintf(stderr, "all dimensions must be positive\n");
        return false;
    }
    if ((q_len % q_tile_rows) != 0 || (kv_len % kv_tile) != 0) {
        std::fprintf(stderr, "q_len must be divisible by 256 and kv_len by 64\n");
        return false;
    }
    if ((kv_len % 128) != 0) {
        std::fprintf(stderr, "this smoke test requires kv_len divisible by 128 for the existing V transpose quantizer\n");
        return false;
    }
    if ((num_q_heads % num_kv_heads) != 0) {
        std::fprintf(stderr, "num_q_heads must be divisible by num_kv_heads\n");
        return false;
    }

    const int q_groups = batch * num_q_heads;
    const int kv_groups = batch * num_kv_heads;
    const size_t q_input_elems = static_cast<size_t>(q_groups) * q_len * head_dim;
    const size_t kv_input_elems = static_cast<size_t>(kv_groups) * kv_len * head_dim;

    std::vector<half> h_q(q_input_elems);
    std::vector<half> h_k(kv_input_elems);
    std::vector<half> h_v(kv_input_elems);
    fill_input(h_q, 1);
    fill_input(h_k, 3);
    fill_input(h_v, 5);

    half* d_q = nullptr;
    half* d_k = nullptr;
    half* d_v = nullptr;
    uint8_t* d_q_fp4 = nullptr;
    uint8_t* d_k_fp4 = nullptr;
    uint8_t* d_v_t_fp4 = nullptr;
    uint8_t* d_q_sf = nullptr;
    uint8_t* d_k_sf = nullptr;
    uint8_t* d_v_t_sf = nullptr;
    uint8_t* d_q_atoms = nullptr;
    uint8_t* d_k_atoms = nullptr;
    uint8_t* d_v_atoms = nullptr;
    __nv_bfloat16* d_o = nullptr;

    const size_t q_fp4_bytes = static_cast<size_t>(q_groups) * q_len * (head_dim / 2);
    const size_t k_fp4_bytes = static_cast<size_t>(kv_groups) * kv_len * (head_dim / 2);
    const size_t v_t_fp4_bytes = static_cast<size_t>(kv_groups) * head_dim * (kv_len / 2);
    const size_t q_sf_bytes = static_cast<size_t>(q_groups) * q_len * (head_dim / 16);
    const size_t k_sf_bytes = static_cast<size_t>(kv_groups) * kv_len * (head_dim / 16);
    const size_t v_t_sf_bytes = static_cast<size_t>(kv_groups) * head_dim * (kv_len / 16);
    const size_t q_atom_bytes = static_cast<size_t>(q_groups) * (q_len / 128) * 2 * sf_atom_bytes;
    const size_t k_atom_bytes = static_cast<size_t>(kv_groups) * (kv_len / 64) * 2 * sf_atom_bytes;
    const size_t v_atom_bytes = static_cast<size_t>(kv_groups) * (kv_len / 64) * sf_atom_bytes;
    const size_t o_bytes = static_cast<size_t>(q_groups) * q_len * head_dim * sizeof(__nv_bfloat16);

    CHECK_CUDA(cudaMalloc(&d_q, q_input_elems * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_k, kv_input_elems * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_v, kv_input_elems * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_q_fp4, q_fp4_bytes));
    CHECK_CUDA(cudaMalloc(&d_k_fp4, k_fp4_bytes));
    CHECK_CUDA(cudaMalloc(&d_v_t_fp4, v_t_fp4_bytes));
    CHECK_CUDA(cudaMalloc(&d_q_sf, q_sf_bytes));
    CHECK_CUDA(cudaMalloc(&d_k_sf, k_sf_bytes));
    CHECK_CUDA(cudaMalloc(&d_v_t_sf, v_t_sf_bytes));
    CHECK_CUDA(cudaMalloc(&d_q_atoms, q_atom_bytes));
    CHECK_CUDA(cudaMalloc(&d_k_atoms, k_atom_bytes));
    CHECK_CUDA(cudaMalloc(&d_v_atoms, v_atom_bytes));
    CHECK_CUDA(cudaMalloc(&d_o, o_bytes));

    CHECK_CUDA(cudaMemcpy(d_q, h_q.data(), q_input_elems * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k.data(), kv_input_elems * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v.data(), kv_input_elems * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_o, 0, o_bytes));

    nvfp4_quantise(d_q, d_q_fp4, d_q_sf, q_groups, q_len, head_dim, false);
    CHECK_CUDA(cudaGetLastError());
    nvfp4_quantise(d_k, d_k_fp4, d_k_sf, kv_groups, kv_len, head_dim, false);
    CHECK_CUDA(cudaGetLastError());
    nvfp4_quantise_transpose(d_v, d_v_t_fp4, d_v_t_sf, kv_groups, kv_len, head_dim, false);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(nvfp4_sm100_pack_q_sf_atoms(d_q_sf, d_q_atoms, batch, num_q_heads, q_len, 0));
    CHECK_CUDA(nvfp4_sm100_pack_k_sf_atoms(d_k_sf, d_k_atoms, batch, num_kv_heads, kv_len, 0));
    CHECK_CUDA(nvfp4_sm100_pack_v_sf_atoms(d_v_t_sf, d_v_atoms, batch, num_kv_heads, kv_len, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    const float softmax_scale_log2 = log2e / std::sqrt(static_cast<float>(head_dim));
    const float v_descale = 1.0f;
    CHECK_CUDA(nvfp4_sm100_attention_launch(
        d_q_fp4,
        d_k_fp4,
        d_v_t_fp4,
        d_q_atoms,
        d_k_atoms,
        d_v_atoms,
        d_o,
        batch,
        q_len,
        kv_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale_log2,
        v_descale,
        0));
    CHECK_CUDA(cudaDeviceSynchronize());

    const size_t sample_words = std::min<size_t>(32, o_bytes / sizeof(uint16_t));
    std::vector<uint16_t> h_o(sample_words);
    CHECK_CUDA(cudaMemcpy(h_o.data(), d_o, sample_words * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    print_u16_sample("attention bf16 sample", h_o);

    bool ok = has_nonzero_u16(h_o);
    if (!ok) {
        std::fprintf(stderr, "attention output sample is all zero\n");
    }

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q_fp4);
    cudaFree(d_k_fp4);
    cudaFree(d_v_t_fp4);
    cudaFree(d_q_sf);
    cudaFree(d_k_sf);
    cudaFree(d_v_t_sf);
    cudaFree(d_q_atoms);
    cudaFree(d_k_atoms);
    cudaFree(d_v_atoms);
    cudaFree(d_o);
    return ok;
}

int main(int argc, char** argv) {
    int batch = 1;
    int q_len = 256;
    int kv_len = 128;
    int num_q_heads = 1;
    int num_kv_heads = 1;

    if (argc > 1) {
        batch = std::atoi(argv[1]);
    }
    if (argc > 2) {
        q_len = std::atoi(argv[2]);
    }
    if (argc > 3) {
        kv_len = std::atoi(argv[3]);
    }
    if (argc > 4) {
        num_q_heads = std::atoi(argv[4]);
    }
    if (argc > 5) {
        num_kv_heads = std::atoi(argv[5]);
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
        "device=%s capability=%d.%d batch=%d q_len=%d kv_len=%d q_heads=%d kv_heads=%d\n",
        prop.name,
        prop.major,
        prop.minor,
        batch,
        q_len,
        kv_len,
        num_q_heads,
        num_kv_heads);

    return run_attention_smoke(batch, q_len, kv_len, num_q_heads, num_kv_heads) ? 0 : 1;
}
