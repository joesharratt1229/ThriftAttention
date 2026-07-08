#include <cmath>
#include <stdexcept>
#include <vector>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
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

namespace {

void check_cuda(cudaError_t err, const char* what) {
    TORCH_CHECK(err == cudaSuccess, what, " failed: ", cudaGetErrorString(err));
}

void check_qkv(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, and v must be contiguous");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q, k, and v must have the same dtype");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
                "q, k, and v must be float16 or bfloat16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, and v must be [batch, heads, seq, head_dim]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v head count mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "k/v sequence length mismatch");
    TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128 && v.size(3) == 128, "head_dim must be 128 for fp4_attention_sm100.cu");
    TORCH_CHECK(q.size(2) % 128 == 0, "q_len must be a multiple of 128");
    TORCH_CHECK(k.size(2) % 128 == 0, "kv_len must be a multiple of 128 for V transpose quantization and attention packing");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_q_heads must be divisible by num_kv_heads");
}

}

pybind11::dict quantise_and_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double softmax_scale) {
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();
    check_qkv(q, k, v);

    const int batch = static_cast<int>(q.size(0));
    const int num_q_heads = static_cast<int>(q.size(1));
    const int num_kv_heads = static_cast<int>(k.size(1));
    const int q_len = static_cast<int>(q.size(2));
    const int kv_len = static_cast<int>(k.size(2));
    constexpr int head_dim = 128;
    const bool is_bf16 = q.scalar_type() == torch::kBFloat16;

    auto byte_opts = q.options().dtype(torch::kUInt8);
    auto out_opts = q.options().dtype(torch::kBFloat16);

    auto q_fp4 = torch::empty({batch, num_q_heads, q_len, head_dim / 2}, byte_opts);
    auto k_fp4 = torch::empty({batch, num_kv_heads, kv_len, head_dim / 2}, byte_opts);
    auto v_t_fp4 = torch::empty({batch, num_kv_heads, head_dim, kv_len / 2}, byte_opts);

    auto q_sf = torch::empty({batch, num_q_heads, q_len, head_dim / 16}, byte_opts);
    auto k_sf = torch::empty({batch, num_kv_heads, kv_len, head_dim / 16}, byte_opts);
    auto v_t_sf = torch::empty({batch, num_kv_heads, head_dim, kv_len / 16}, byte_opts);

    nvfp4_quantise(q.data_ptr(), q_fp4.data_ptr(), q_sf.data_ptr(), batch * num_q_heads, q_len, head_dim, is_bf16);
    nvfp4_quantise(k.data_ptr(), k_fp4.data_ptr(), k_sf.data_ptr(), batch * num_kv_heads, kv_len, head_dim, is_bf16);
    nvfp4_quantise_transpose(v.data_ptr(), v_t_fp4.data_ptr(), v_t_sf.data_ptr(), batch * num_kv_heads, kv_len, head_dim, is_bf16);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    check_cuda(cudaGetLastError(), "nvfp4 quantize launch");

    auto q_sf_atoms = torch::empty({batch, num_q_heads, q_len / 128, 2, 512}, byte_opts);
    auto k_sf_atoms = torch::empty({batch, num_kv_heads, kv_len / 128, 2, 512}, byte_opts);
    auto v_sf_atoms = torch::empty({batch, num_kv_heads, kv_len / 64, 1, 512}, byte_opts);

    check_cuda(nvfp4_sm100_pack_q_sf_atoms(
                   static_cast<const uint8_t*>(q_sf.data_ptr()),
                   static_cast<uint8_t*>(q_sf_atoms.data_ptr()),
                   batch, num_q_heads, q_len, stream),
               "nvfp4_sm100_pack_q_sf_atoms");
    check_cuda(nvfp4_sm100_pack_k_sf_atoms(
                   static_cast<const uint8_t*>(k_sf.data_ptr()),
                   static_cast<uint8_t*>(k_sf_atoms.data_ptr()),
                   batch, num_kv_heads, kv_len, stream),
               "nvfp4_sm100_pack_k_sf_atoms");
    check_cuda(nvfp4_sm100_pack_v_sf_atoms(
                   static_cast<const uint8_t*>(v_t_sf.data_ptr()),
                   static_cast<uint8_t*>(v_sf_atoms.data_ptr()),
                   batch, num_kv_heads, kv_len, stream),
               "nvfp4_sm100_pack_v_sf_atoms");

    auto out = torch::empty({batch, num_q_heads, q_len, head_dim}, out_opts);
    const float softmax_scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
    check_cuda(nvfp4_sm100_attention_launch(
                   q_fp4.data_ptr(),
                   k_fp4.data_ptr(),
                   v_t_fp4.data_ptr(),
                   static_cast<const uint8_t*>(q_sf_atoms.data_ptr()),
                   static_cast<const uint8_t*>(k_sf_atoms.data_ptr()),
                   static_cast<const uint8_t*>(v_sf_atoms.data_ptr()),
                   reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                   batch, q_len, kv_len, num_q_heads, num_kv_heads,
                   softmax_scale_log2, 1.0f, stream),
               "nvfp4_sm100_attention_launch");
    check_cuda(cudaGetLastError(), "nvfp4 attention launch status");

    pybind11::dict result;
    result["out"] = out;
    result["q_fp4"] = q_fp4;
    result["k_fp4"] = k_fp4;
    result["v_t_fp4"] = v_t_fp4;
    result["q_sf"] = q_sf;
    result["k_sf"] = k_sf;
    result["v_t_sf"] = v_t_sf;
    result["q_sf_atoms"] = q_sf_atoms;
    result["k_sf_atoms"] = k_sf_atoms;
    result["v_sf_atoms"] = v_sf_atoms;
    return result;
}

torch::Tensor attention_only(
    torch::Tensor q_fp4,
    torch::Tensor k_fp4,
    torch::Tensor v_t_fp4,
    torch::Tensor q_sf_atoms,
    torch::Tensor k_sf_atoms,
    torch::Tensor v_sf_atoms,
    double softmax_scale) {
    const int batch = static_cast<int>(q_fp4.size(0));
    const int num_q_heads = static_cast<int>(q_fp4.size(1));
    const int q_len = static_cast<int>(q_fp4.size(2));
    const int num_kv_heads = static_cast<int>(k_fp4.size(1));
    const int kv_len = static_cast<int>(k_fp4.size(2));
    constexpr int head_dim = 128;

    auto out = torch::empty({batch, num_q_heads, q_len, head_dim},
                            q_fp4.options().dtype(torch::kBFloat16));
    const float softmax_scale_log2 = static_cast<float>(softmax_scale * 1.4426950408889634);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    check_cuda(nvfp4_sm100_attention_launch(
                   q_fp4.data_ptr(),
                   k_fp4.data_ptr(),
                   v_t_fp4.data_ptr(),
                   static_cast<const uint8_t*>(q_sf_atoms.data_ptr()),
                   static_cast<const uint8_t*>(k_sf_atoms.data_ptr()),
                   static_cast<const uint8_t*>(v_sf_atoms.data_ptr()),
                   reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                   batch, q_len, kv_len, num_q_heads, num_kv_heads,
                   softmax_scale_log2, 1.0f, stream),
               "nvfp4_sm100_attention_launch");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantise_and_attention", &quantise_and_attention,
          "Quantize Q/K/V to NVFP4, pack scale atoms, and run SM100 attention");
    m.def("attention_only", &attention_only,
          "Run SM100 NVFP4 attention on pre-quantised inputs (for kernel-only benchmarking)");
}
