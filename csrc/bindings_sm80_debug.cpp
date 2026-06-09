#include <torch/extension.h>

void sm80_mma_m16n8k32_s8_test(
    const void *A,
    const void *B,
    void *D);

void int8_attention_noncausal(
    const void *Q_raw,
    const void *K_raw,
    const void *V_raw,
    const void *S_Q_raw,
    const void *S_K_raw,
    const void *S_V_raw,
    void *O_raw,
    int bs,
    int q_len,
    int kv_len,
    int kv_capacity,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool is_bf16);

static at::Tensor sm80_mma_m16n8k32_s8_test_impl(
    const at::Tensor &a,
    const at::Tensor &b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(a.dtype() == at::kChar, "A must have dtype torch.int8");
    TORCH_CHECK(b.dtype() == at::kChar, "B must have dtype torch.int8");
    TORCH_CHECK(a.sizes() == at::IntArrayRef({16, 32}), "A must be [16, 32]");
    TORCH_CHECK(b.sizes() == at::IntArrayRef({32, 8}), "B must be [32, 8]");

    auto d = at::empty({16, 8}, at::TensorOptions().dtype(at::kInt).device(a.device()));
    sm80_mma_m16n8k32_s8_test(a.data_ptr(), b.data_ptr(), d.data_ptr());
    return d;
}

static at::Tensor sm80_int8_attention_noncausal(
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &V,
    const at::Tensor &S_Q,
    const at::Tensor &S_K,
    const at::Tensor &S_V,
    const bool is_bf16)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(S_Q.is_cuda() && S_K.is_cuda() && S_V.is_cuda(), "S_Q/S_K/S_V must be CUDA tensors");

    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(S_Q.is_contiguous() && S_K.is_contiguous() && S_V.is_contiguous(), "S_Q/S_K/S_V must be contiguous");

    TORCH_CHECK(Q.dtype() == at::kChar, "Q must be torch.int8");
    TORCH_CHECK(K.dtype() == at::kChar, "K must be torch.int8");
    TORCH_CHECK(V.dtype() == at::kChar, "V must be torch.int8");

    TORCH_CHECK(S_Q.dtype() == at::kFloat, "S_Q must be torch.float32");
    TORCH_CHECK(S_K.dtype() == at::kFloat, "S_K must be torch.float32");
    TORCH_CHECK(S_V.dtype() == at::kFloat, "S_V must be torch.float32");

    const int bs = Q.size(0);
    const int q_len = Q.size(1);
    const int num_q_heads = Q.size(2);
    const int head_dim = Q.size(3);

    const int kv_len = K.size(1);
    const int num_kv_heads = K.size(2);
    const int kv_capacity = kv_len;

    TORCH_CHECK(Q.dim() == 4, "Q must have shape [bs, q_len, num_q_heads, head_dim]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [bs, kv_len, num_kv_heads, head_dim]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [bs, kv_len, num_kv_heads, head_dim]");

    TORCH_CHECK(K.size(0) == bs && V.size(0) == bs, "K/V batch size must match Q");
    TORCH_CHECK(K.size(1) == V.size(1), "K/V kv_len size must match");
    TORCH_CHECK(K.size(2) == V.size(2), "K/V num_kv_heads must match");
    TORCH_CHECK(K.size(3) == head_dim && V.size(3) == head_dim, "K/V head_dim must match Q");

    const int scale_dim = head_dim / 32;

    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(S_Q.sizes() == at::IntArrayRef({bs, q_len, num_q_heads, scale_dim}), "S_Q must have shape [bs, q_len, num_q_heads, head_dim / 32]");
    TORCH_CHECK(S_K.sizes() == at::IntArrayRef({bs, kv_len, num_kv_heads, scale_dim}), "S_K must have shape [bs, kv_len, num_kv_heads, head_dim / 32]");
    TORCH_CHECK(S_V.sizes() == at::IntArrayRef({bs, kv_len, num_kv_heads, scale_dim}), "S_V must have shape [bs, kv_len, num_kv_heads, head_dim / 32]");
    
    auto out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    auto O = at::empty(
        {bs, q_len, num_q_heads, head_dim},
        at::TensorOptions().dtype(out_dtype).device(Q.device())
    );

    int8_attention_noncausal(
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        S_Q.data_ptr(),
        S_K.data_ptr(),
        S_V.data_ptr(),
        O.data_ptr(),
        bs,
        q_len,
        kv_len,
        kv_capacity,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_bf16
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sm80_mma_m16n8k32_s8_test", &sm80_mma_m16n8k32_s8_test_impl);
    m.def("sm80_int8_attention_noncausal", &sm80_int8_attention_noncausal);
}