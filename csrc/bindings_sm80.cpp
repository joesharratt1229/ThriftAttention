#include <torch/extension.h>

#include <vector>

void sm80_mma_m16n8k32_s8_test(
    const void *A,
    const void *B,
    void *D);

void sm80_mma_m16n8k64_s4_test(
    const void *A,
    const void *B,
    void *D);

void sm80_mma_int8_scores_test(
    const void *Q,
    const void *K,
    const void *S_Q,
    const void *S_K,
    void *scores,
    int head_dim);

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

void int8_attention_causal(
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

void int4_attention_noncausal(
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

void int4_attention_causal(
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

void int4_quantize(
    const void *X_raw,
    void *X_packed_raw,
    void *X_scale_raw,
    int rows,
    int head_dim,
    bool is_bf16);


static int64_t sm80_capacity_from_stride(
    const at::Tensor &x,
    int64_t elements_per_token,
    const char *name)
{
    TORCH_CHECK(x.stride(3) == 1, name, " last dimension must be contiguous");
    TORCH_CHECK(x.stride(2) == x.size(3), name, " head stride is unsupported");
    TORCH_CHECK(x.stride(1) == elements_per_token, name, " sequence stride is unsupported");
    TORCH_CHECK(x.stride(0) % elements_per_token == 0, name, " batch stride is unsupported");
    return x.stride(0) / elements_per_token;
}

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


static at::Tensor sm80_mma_m16n8k64_s4_test_impl(
    const at::Tensor &a,
    const at::Tensor &b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(a.dtype() == at::kChar, "A must have dtype torch.int8 containing signed int4 values [-8, 7]");
    TORCH_CHECK(b.dtype() == at::kChar, "B must have dtype torch.int8 containing signed int4 values [-8, 7]");
    TORCH_CHECK(a.sizes() == at::IntArrayRef({16, 64}), "A must be [16, 64]");
    TORCH_CHECK(b.sizes() == at::IntArrayRef({64, 8}), "B must be [64, 8]");
    TORCH_CHECK(a.min().item<int8_t>() >= -8 && a.max().item<int8_t>() <= 7, "A values must be in signed int4 range [-8, 7]");
    TORCH_CHECK(b.min().item<int8_t>() >= -8 && b.max().item<int8_t>() <= 7, "B values must be in signed int4 range [-8, 7]");

    auto d = at::empty({16, 8}, at::TensorOptions().dtype(at::kInt).device(a.device()));
    sm80_mma_m16n8k64_s4_test(a.data_ptr(), b.data_ptr(), d.data_ptr());
    return d;
}

static at::Tensor sm80_mma_int8_scores_test_impl(
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &S_Q,
    const at::Tensor &S_K)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda(), "Q and K must be CUDA tensors");
    TORCH_CHECK(S_Q.is_cuda() && S_K.is_cuda(), "S_Q and S_K must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous(), "Q and K must be contiguous");
    TORCH_CHECK(S_Q.is_contiguous() && S_K.is_contiguous(), "S_Q and S_K must be contiguous");
    TORCH_CHECK(Q.dtype() == at::kChar && K.dtype() == at::kChar, "Q and K must be torch.int8");
    TORCH_CHECK(S_Q.dtype() == at::kFloat && S_K.dtype() == at::kFloat, "S_Q and S_K must be torch.float32");
    TORCH_CHECK(Q.dim() == 2 && K.dim() == 2, "Q and K must be rank-2 tensors");
    TORCH_CHECK(S_Q.dim() == 2 && S_K.dim() == 2, "S_Q and S_K must be rank-2 tensors");

    const int head_dim = Q.size(1);
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(Q.sizes() == at::IntArrayRef({16, head_dim}), "Q must have shape [16, head_dim]");
    TORCH_CHECK(K.sizes() == at::IntArrayRef({8, head_dim}), "K must have shape [8, head_dim]");

    const int scale_dim = head_dim / 32;
    TORCH_CHECK(S_Q.sizes() == at::IntArrayRef({16, scale_dim}), "S_Q must have shape [16, head_dim / 32]");
    TORCH_CHECK(S_K.sizes() == at::IntArrayRef({8, scale_dim}), "S_K must have shape [8, head_dim / 32]");

    auto scores = at::empty(
        {16, 8},
        at::TensorOptions().dtype(at::kFloat).device(Q.device()));

    sm80_mma_int8_scores_test(
        Q.data_ptr(), K.data_ptr(), S_Q.data_ptr(), S_K.data_ptr(),
        scores.data_ptr(), head_dim);
    return scores;
}



static std::vector<at::Tensor> sm80_int4_quantize(const at::Tensor &X, const bool is_bf16)
{
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(X.dim() == 4, "X must have shape [bs, seq_len, heads, head_dim]");
    TORCH_CHECK(X.dtype() == (is_bf16 ? at::kBFloat16 : at::kHalf), "X dtype must match is_bf16");

    const int bs = X.size(0);
    const int seq_len = X.size(1);
    const int heads = X.size(2);
    const int head_dim = X.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");

    const int rows = bs * seq_len * heads;
    const int scale_dim = head_dim / 64;
    auto packed = at::empty({bs, seq_len, heads, head_dim / 2}, at::TensorOptions().dtype(at::kByte).device(X.device()));
    auto scale = at::empty({bs, seq_len, heads, scale_dim}, at::TensorOptions().dtype(at::kFloat).device(X.device()));

    int4_quantize(X.data_ptr(), packed.data_ptr(), scale.data_ptr(), rows, head_dim, is_bf16);
    return {packed, scale};
}

static void check_sm80_int4_attention_inputs(
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &V,
    const at::Tensor &S_Q,
    const at::Tensor &S_K,
    const at::Tensor &S_V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(S_Q.is_cuda() && S_K.is_cuda() && S_V.is_cuda(), "S_Q/S_K/S_V must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(S_Q.is_contiguous(), "S_Q must be contiguous");
    TORCH_CHECK(Q.dtype() == at::kByte, "Q must be torch.uint8 packed signed int4");
    TORCH_CHECK(K.dtype() == at::kByte, "K must be torch.uint8 packed signed int4");
    TORCH_CHECK(V.dtype() == at::kByte, "V must be torch.uint8 packed signed int4");
    TORCH_CHECK(S_Q.dtype() == at::kFloat, "S_Q must be torch.float32");
    TORCH_CHECK(S_K.dtype() == at::kFloat, "S_K must be torch.float32");
    TORCH_CHECK(S_V.dtype() == at::kFloat, "S_V must be torch.float32");
    TORCH_CHECK(Q.dim() == 4, "Q must have shape [bs, q_len, num_q_heads, head_dim / 2]");
    TORCH_CHECK(K.dim() == 4, "K must have shape [bs, kv_len, num_kv_heads, head_dim / 2]");
    TORCH_CHECK(V.dim() == 4, "V must have shape [bs, kv_len, num_kv_heads, head_dim / 2]");

    const int bs = Q.size(0);
    const int q_len = Q.size(1);
    const int num_q_heads = Q.size(2);
    const int packed_head_dim = Q.size(3);
    const int head_dim = packed_head_dim * 2;
    const int kv_len = K.size(1);
    const int num_kv_heads = K.size(2);
    const int scale_dim = head_dim / 64;

    TORCH_CHECK(head_dim == 64 || head_dim == 128, "head_dim must be 64 or 128");
    TORCH_CHECK(K.size(0) == bs && V.size(0) == bs, "K/V batch size must match Q");
    TORCH_CHECK(K.size(1) == V.size(1), "K/V kv_len must match");
    TORCH_CHECK(K.size(2) == V.size(2), "K/V num_kv_heads must match");
    TORCH_CHECK(K.size(3) == packed_head_dim && V.size(3) == packed_head_dim, "K/V packed head_dim must match Q");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(S_Q.sizes() == at::IntArrayRef({bs, q_len, num_q_heads, scale_dim}), "S_Q must have shape [bs, q_len, num_q_heads, head_dim / 64]");
    TORCH_CHECK(S_K.sizes() == at::IntArrayRef({bs, kv_len, num_kv_heads, scale_dim}), "S_K must have shape [bs, kv_len, num_kv_heads, head_dim / 64]");
    TORCH_CHECK(S_V.sizes() == at::IntArrayRef({bs, kv_len, num_kv_heads, scale_dim}), "S_V must have shape [bs, kv_len, num_kv_heads, head_dim / 64]");

    const int64_t kv_capacity = sm80_capacity_from_stride(K, num_kv_heads * packed_head_dim, "K");
    TORCH_CHECK(sm80_capacity_from_stride(V, num_kv_heads * packed_head_dim, "V") == kv_capacity, "V capacity must match K");
    TORCH_CHECK(sm80_capacity_from_stride(S_K, num_kv_heads * scale_dim, "S_K") == kv_capacity, "S_K capacity must match K");
    TORCH_CHECK(sm80_capacity_from_stride(S_V, num_kv_heads * scale_dim, "S_V") == kv_capacity, "S_V capacity must match K");
    TORCH_CHECK(kv_capacity >= kv_len, "KV capacity must be >= kv_len");
}

static at::Tensor sm80_int4_attention_packed(
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &V,
    const at::Tensor &S_Q,
    const at::Tensor &S_K,
    const at::Tensor &S_V,
    const bool causal,
    const bool is_bf16)
{
    check_sm80_int4_attention_inputs(Q, K, V, S_Q, S_K, S_V);

    const int bs = Q.size(0);
    const int q_len = Q.size(1);
    const int num_q_heads = Q.size(2);
    const int head_dim = Q.size(3) * 2;
    const int kv_len = K.size(1);
    const int num_kv_heads = K.size(2);
    const int kv_capacity = sm80_capacity_from_stride(K, num_kv_heads * Q.size(3), "K");

    auto out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    auto O = at::empty(
        {bs, q_len, num_q_heads, head_dim},
        at::TensorOptions().dtype(out_dtype).device(Q.device()));

    if (causal) {
        int4_attention_causal(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(),
            S_Q.data_ptr(), S_K.data_ptr(), S_V.data_ptr(), O.data_ptr(),
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads,
            head_dim, is_bf16);
    } else {
        int4_attention_noncausal(
            Q.data_ptr(), K.data_ptr(), V.data_ptr(),
            S_Q.data_ptr(), S_K.data_ptr(), S_V.data_ptr(), O.data_ptr(),
            bs, q_len, kv_len, kv_capacity, num_q_heads, num_kv_heads,
            head_dim, is_bf16);
    }

    return O;
}

static at::Tensor sm80_int4_attention_noncausal(
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &V,
    const at::Tensor &S_Q,
    const at::Tensor &S_K,
    const at::Tensor &S_V,
    const bool is_bf16)
{
    return sm80_int4_attention_packed(Q, K, V, S_Q, S_K, S_V, false, is_bf16);
}

static at::Tensor sm80_int4_attention_causal(
    const at::Tensor &Q,
    const at::Tensor &K,
    const at::Tensor &V,
    const at::Tensor &S_Q,
    const at::Tensor &S_K,
    const at::Tensor &S_V,
    const bool is_bf16)
{
    return sm80_int4_attention_packed(Q, K, V, S_Q, S_K, S_V, true, is_bf16);
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

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(S_Q.is_contiguous(), "S_Q must be contiguous");

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
    const int kv_capacity = sm80_capacity_from_stride(K, num_kv_heads * head_dim, "K");

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
    TORCH_CHECK(sm80_capacity_from_stride(V, num_kv_heads * head_dim, "V") == kv_capacity, "V capacity must match K");
    TORCH_CHECK(sm80_capacity_from_stride(S_K, num_kv_heads * scale_dim, "S_K") == kv_capacity, "S_K capacity must match K");
    TORCH_CHECK(sm80_capacity_from_stride(S_V, num_kv_heads * scale_dim, "S_V") == kv_capacity, "S_V capacity must match K");
    TORCH_CHECK(kv_capacity >= kv_len, "KV capacity must be >= kv_len");
    
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

static at::Tensor sm80_int8_attention_causal(
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

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(S_Q.is_contiguous(), "S_Q must be contiguous");

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
    const int kv_capacity = sm80_capacity_from_stride(K, num_kv_heads * head_dim, "K");

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
    TORCH_CHECK(sm80_capacity_from_stride(V, num_kv_heads * head_dim, "V") == kv_capacity, "V capacity must match K");
    TORCH_CHECK(sm80_capacity_from_stride(S_K, num_kv_heads * scale_dim, "S_K") == kv_capacity, "S_K capacity must match K");
    TORCH_CHECK(sm80_capacity_from_stride(S_V, num_kv_heads * scale_dim, "S_V") == kv_capacity, "S_V capacity must match K");
    TORCH_CHECK(kv_capacity >= kv_len, "KV capacity must be >= kv_len");
    
    auto out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    auto O = at::empty(
        {bs, q_len, num_q_heads, head_dim},
        at::TensorOptions().dtype(out_dtype).device(Q.device())
    );

    int8_attention_causal(
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
    m.def("sm80_mma_m16n8k64_s4_test", &sm80_mma_m16n8k64_s4_test_impl);
    m.def("sm80_mma_int8_scores_test", &sm80_mma_int8_scores_test_impl);
    m.def("sm80_int8_attention_noncausal", &sm80_int8_attention_noncausal);
    m.def("sm80_int8_attention_causal", &sm80_int8_attention_causal);
    m.def("sm80_int4_quantize", &sm80_int4_quantize);
    m.def("sm80_int4_attention_noncausal", &sm80_int4_attention_noncausal);
    m.def("sm80_int4_attention_causal", &sm80_int4_attention_causal);
}