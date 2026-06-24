#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <vector>

namespace {

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

void check_packed_qkv(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t) {
    TORCH_CHECK(q_packed.is_cuda() && k_packed.is_cuda() && v_packed_t.is_cuda(),
                "packed Q/K/V must be CUDA tensors");
    TORCH_CHECK(q_scale.is_cuda() && k_scale.is_cuda() && v_scale_t.is_cuda(),
                "Q/K/V scales must be CUDA tensors");
    TORCH_CHECK(q_packed.device() == k_packed.device() && q_packed.device() == v_packed_t.device() &&
                q_packed.device() == q_scale.device() && q_packed.device() == k_scale.device() &&
                q_packed.device() == v_scale_t.device(),
                "packed Q/K/V and scales must be on the same CUDA device");
    TORCH_CHECK(q_packed.scalar_type() == at::kByte && k_packed.scalar_type() == at::kByte &&
                v_packed_t.scalar_type() == at::kByte,
                "packed Q/K/V must use torch.uint8 storage");
    TORCH_CHECK(q_scale.scalar_type() == at::kFloat8_e4m3fn &&
                k_scale.scalar_type() == at::kFloat8_e4m3fn &&
                v_scale_t.scalar_type() == at::kFloat8_e4m3fn,
                "Q/K/V scales must use torch.float8_e4m3fn storage");
    TORCH_CHECK(q_packed.dim() == 4 && k_packed.dim() == 4 && v_packed_t.dim() == 4,
                "packed Q/K/V must be 4D tensors");
    TORCH_CHECK(q_scale.dim() == 4 && k_scale.dim() == 4 && v_scale_t.dim() == 4,
                "Q/K/V scales must be 4D tensors");
    TORCH_CHECK(q_packed.size(0) > 0 && q_packed.size(1) > 0 && k_packed.size(1) > 0 &&
                v_packed_t.size(1) > 0,
                "packed Q/K/V batch and head dimensions must be positive");
    TORCH_CHECK(q_packed.size(0) == k_packed.size(0), "packed Q/K batch dimensions must match");
    TORCH_CHECK(q_packed.size(0) == v_packed_t.size(0), "packed Q/V batch dimensions must match");
    TORCH_CHECK(q_packed.size(1) % k_packed.size(1) == 0,
                "Q heads must be divisible by KV heads");
    TORCH_CHECK(q_packed.size(3) == k_packed.size(3), "packed Q/K head dimensions must match");
    TORCH_CHECK(q_packed.size(3) == 32 || q_packed.size(3) == 64,
                "packed head_dim must represent head_dim 64 or 128");

    const int head_dim_packed = q_packed.size(3);
    const int head_dim = head_dim_packed * 2;
    const int scale_dim = head_dim / 16;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    TORCH_CHECK(q_len > 0 && kv_len > 0, "packed Q/K sequence lengths must be positive");
    TORCH_CHECK(k_packed.size(1) == v_packed_t.size(1), "packed K/V head dimensions must match");
    TORCH_CHECK(v_packed_t.size(2) == head_dim, "packed transposed V head_dim mismatch");
    TORCH_CHECK(q_scale.size(0) == q_packed.size(0) && q_scale.size(1) == q_packed.size(1) &&
                q_scale.size(2) == q_len && q_scale.size(3) == scale_dim,
                "Q scale shape must match packed Q layout");
    TORCH_CHECK(k_scale.size(0) == k_packed.size(0) && k_scale.size(1) == k_packed.size(1) &&
                k_scale.size(2) == kv_len && k_scale.size(3) == scale_dim,
                "K scale shape must match packed K layout");
    TORCH_CHECK(v_scale_t.size(0) == v_packed_t.size(0) && v_scale_t.size(1) == v_packed_t.size(1) &&
                v_scale_t.size(2) == head_dim,
                "V scale shape must match packed V layout");
    TORCH_CHECK(q_packed.is_contiguous() && q_scale.is_contiguous(),
                "packed Q and Q scale must be contiguous");
    TORCH_CHECK(k_packed.stride(3) == 1 && k_packed.stride(2) == head_dim_packed,
                "packed K must have contiguous per-token head_dim storage");
    TORCH_CHECK(k_packed.stride(1) % head_dim_packed == 0,
                "packed K head stride must be a whole number of KV tokens");
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / head_dim_packed;
    TORCH_CHECK(kv_capacity >= kv_len, "packed K capacity must cover packed K length");
    TORCH_CHECK(k_packed.stride(0) == k_packed.size(1) * k_packed.stride(1),
                "packed K batch stride must match packed KV-head capacity");
    TORCH_CHECK(k_scale.stride(3) == 1 && k_scale.stride(2) == scale_dim,
                "K scale must have contiguous per-token scale storage");
    TORCH_CHECK(k_scale.stride(1) == kv_capacity * scale_dim,
                "K scale head stride must match packed K capacity");
    TORCH_CHECK(k_scale.stride(0) == k_scale.size(1) * k_scale.stride(1),
                "K scale batch stride must match packed KV-head capacity");

    const int padded_kv = ceil_div(kv_capacity, 128) * 128;
    TORCH_CHECK(v_packed_t.size(3) >= ceil_div(kv_len, 2),
                "packed transposed V length must cover packed K length");
    TORCH_CHECK(v_packed_t.stride(3) == 1 && v_packed_t.stride(2) == padded_kv / 2,
                "packed transposed V stride must match padded KV capacity");
    TORCH_CHECK(v_packed_t.stride(1) == head_dim * (padded_kv / 2),
                "packed transposed V head stride must match padded KV capacity");
    TORCH_CHECK(v_packed_t.stride(0) == v_packed_t.size(1) * v_packed_t.stride(1),
                "packed transposed V batch stride must match padded KV capacity");
    TORCH_CHECK(v_scale_t.size(3) >= ceil_div(kv_len, 16),
                "V scale length must cover packed K length");
    TORCH_CHECK(v_scale_t.stride(3) == 1 && v_scale_t.stride(2) == padded_kv / 16,
                "V scale stride must match padded KV capacity");
    TORCH_CHECK(v_scale_t.stride(1) == head_dim * (padded_kv / 16),
                "V scale head stride must match padded KV capacity");
    TORCH_CHECK(v_scale_t.stride(0) == v_scale_t.size(1) * v_scale_t.stride(1),
                "V scale batch stride must match padded KV capacity");
}

void check_nvfp4_input(const at::Tensor& x, bool is_bf16) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");
    TORCH_CHECK(x.size(3) == 64 || x.size(3) == 128, "head_dim must be 64 or 128, got ", x.size(3));
    const at::ScalarType expected_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(x.scalar_type() == expected_dtype,
                "x dtype must match is_bf16; expected ", expected_dtype, ", got ", x.scalar_type());
}

}  // namespace

void sm100_fp4_attention_causal_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void sm100_fp4_attention_noncausal_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void sm100_fp4_attention_single_query_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);

void nvfp4_quantise(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void nvfp4_quantise_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);
void nvfp4_quantise_transpose(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);

static at::Tensor sm100_fp4_attention_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool single_query,
    bool is_bf16) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(!single_query || (q_len >= 1 && q_len <= 16),
                "single-query SM100 attention expects grouped q_len in [1, 16], got ", q_len);
    TORCH_CHECK(single_query || kv_len % 64 == 0,
                "tiled packed SM100 attention requires KV length divisible by 64");
    TORCH_CHECK(!causal || single_query || q_len == kv_len,
                "causal packed SM100 attention requires q_len == kv_len");
    const c10::cuda::CUDAGuard device_guard(q_packed.device());

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    if (single_query) {
        sm100_fp4_attention_single_query_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else if (causal) {
        sm100_fp4_attention_causal_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        sm100_fp4_attention_noncausal_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor sm100_fp4_attention_causal_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return sm100_fp4_attention_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, false, is_bf16);
}

static at::Tensor sm100_fp4_attention_noncausal_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return sm100_fp4_attention_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, false, is_bf16);
}

static at::Tensor sm100_fp4_attention_single_query_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return sm100_fp4_attention_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, true, is_bf16);
}

static std::vector<at::Tensor> nvfp4_quantize(const at::Tensor& x, bool is_bf16 = false) {
    check_nvfp4_input(x, is_bf16);
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 16}, opts_f8);

    nvfp4_quantise(x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
                   batch * heads, seq_len, head_dim, is_bf16);
    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_permuted(const at::Tensor& x, bool is_bf16 = false) {
    check_nvfp4_input(x, is_bf16);
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 16}, opts_f8);

    nvfp4_quantise_permute_seq(x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
                               batch * heads, seq_len, head_dim, false, is_bf16);
    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_transposed(const at::Tensor& x, bool is_bf16 = false) {
    check_nvfp4_input(x, is_bf16);
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 16}, opts_f8);

    nvfp4_quantise_transpose(x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
                             batch * heads, seq_len, head_dim, is_bf16);
    return {x_packed, x_scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sm100_fp4_attention_causal_nvfp4_packed", &sm100_fp4_attention_causal_nvfp4_packed,
          pybind11::arg("q_packed"), pybind11::arg("k_packed"), pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"), pybind11::arg("k_scale"), pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false);
    m.def("sm100_fp4_attention_noncausal_nvfp4_packed", &sm100_fp4_attention_noncausal_nvfp4_packed,
          pybind11::arg("q_packed"), pybind11::arg("k_packed"), pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"), pybind11::arg("k_scale"), pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false);
    m.def("sm100_fp4_attention_single_query_nvfp4_packed", &sm100_fp4_attention_single_query_nvfp4_packed,
          pybind11::arg("q_packed"), pybind11::arg("k_packed"), pybind11::arg("v_packed_t"),
          pybind11::arg("q_scale"), pybind11::arg("k_scale"), pybind11::arg("v_scale_t"),
          pybind11::arg("is_bf16") = false);
    m.def("nvfp4_quantize", &nvfp4_quantize, pybind11::arg("x"), pybind11::arg("is_bf16") = false);
    m.def("nvfp4_quantize_permuted", &nvfp4_quantize_permuted,
          pybind11::arg("x"), pybind11::arg("is_bf16") = false);
    m.def("nvfp4_quantize_transposed", &nvfp4_quantize_transposed,
          pybind11::arg("x"), pybind11::arg("is_bf16") = false);
}
