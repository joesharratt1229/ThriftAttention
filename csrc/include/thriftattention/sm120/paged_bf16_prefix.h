#pragma once

#include <cstdint>

// A bfloat16 prefix per sequence for the varlen NVFP4 kernel (fp4_attention.cu, paged_prefix_attend): keys
// 0 .. prefix_len[seq] - 1 of each sequence, held in bfloat16 by a paged KV cache, attended in 16 bits after the
// sequence's FP4 keys, which then start at its key prefix_len[seq]. Host and device code share it, so the bfloat16
// pointers are untyped here.
//   q_bf16: [out_tokens, num_q_heads, head_dim] bfloat16, the query of each written row at its output index
//   key_cache, value_cache: element (block, slot, kv head, d) at block * block_stride + slot * token_stride
//       + kv head * head_stride + d (elements); the sequence's key j is in block
//       block_table[seq * block_table_stride + j / block_size], slot j % block_size
//   key_mean (or null): [num_kv_heads, head_dim] float, the vector subtracted from the keys before FP4 quantization
//       (the FP4 logits are q . (k - key_mean) with the key bias); the 16-bit logits q . k are brought to that frame
//       by subtracting q . key_mean
struct PagedBf16PrefixArgs {
    const void* q_bf16 = nullptr;
    const void* key_cache = nullptr;
    const void* value_cache = nullptr;
    int64_t block_stride = 0;
    int64_t token_stride = 0;
    int64_t head_stride = 0;
    const int* block_table = nullptr;
    int block_table_stride = 0;
    int block_size = 0;
    const int* prefix_len = nullptr;
    const float* key_mean = nullptr;
};
