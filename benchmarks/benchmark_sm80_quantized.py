import statistics

import torch

from thriftattention._extension import get_extension


def quantize_int8_per_32(x):
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    scale = grouped.abs().amax(-1).clamp(min=1e-6) / 127.0
    quantized = torch.round(grouped / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return quantized.reshape_as(x).contiguous(), scale.contiguous()


def benchmark(fn, warmup=20, repetitions=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "median": statistics.median(times),
        "mean": statistics.fmean(times),
        "minimum": min(times),
    }


def print_stats(label, stats):
    print(
        f"{label}: median={stats['median']:.4f} ms, "
        f"mean={stats['mean']:.4f} ms, min={stats['minimum']:.4f} ms"
    )


def make_inputs(bs, q_len, kv_len, q_heads, kv_heads, head_dim):
    q = torch.randn(bs, q_len, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(bs, kv_len, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    return q, k, v


def run_case(bs, q_len, kv_len, q_heads, kv_heads, head_dim):
    ext = get_extension()
    q, k, v = make_inputs(bs, q_len, kv_len, q_heads, kv_heads, head_dim)

    q_i8, s_q8 = quantize_int8_per_32(q)
    k_i8, s_k8 = quantize_int8_per_32(k)
    v_i8, s_v8 = quantize_int8_per_32(v)

    q_i4, s_q4 = ext.sm80_int4_quantize(q.contiguous(), False)
    k_i4, s_k4 = ext.sm80_int4_quantize(k.contiguous(), False)
    v_i4, s_v4 = ext.sm80_int4_quantize(v.contiguous(), False)

    prefix = f"bs={bs} q={q_len} kv={kv_len} heads={q_heads}/{kv_heads} dim={head_dim}"

    print_stats(
        f"{prefix} int8 noncausal",
        benchmark(lambda: ext.sm80_int8_attention_noncausal(q_i8, k_i8, v_i8, s_q8, s_k8, s_v8, False)),
    )
    print_stats(
        f"{prefix} int8 causal",
        benchmark(lambda: ext.sm80_int8_attention_causal(q_i8, k_i8, v_i8, s_q8, s_k8, s_v8, False)),
    )
    print_stats(
        f"{prefix} int4 noncausal",
        benchmark(lambda: ext.sm80_int4_attention_noncausal(q_i4, k_i4, v_i4, s_q4, s_k4, s_v4, False)),
    )
    print_stats(
        f"{prefix} int4 causal",
        benchmark(lambda: ext.sm80_int4_attention_causal(q_i4, k_i4, v_i4, s_q4, s_k4, s_v4, False)),
    )


if __name__ == "__main__":
    torch.manual_seed(1234)
    run_case(1, 64, 64, 1, 1, 64)
    run_case(1, 128, 128, 1, 1, 128)
    run_case(1, 256, 256, 8, 4, 128)
    run_case(1, 512, 512, 8, 4, 128)
    run_case(1, 1024, 1024, 8, 4, 128)
