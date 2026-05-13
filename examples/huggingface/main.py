#!/usr/bin/env python3
from __future__ import annotations

import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import thriftattention as ta


MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "ThriftAttention patches Hugging Face attention for long context prefill."
SEQ_LEN = 32768


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ThriftAttention")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()

    ta.patch_model(model, backend="hf", mode="thrift", causal=True, fp16_fraction=0.05)

    token_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
    token_ids = (token_ids * math.ceil(SEQ_LEN / len(token_ids)))[:SEQ_LEN]
    input_ids = torch.tensor([token_ids], device="cuda")

    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits

    print(tokenizer.decode(logits[:, -1].argmax(dim=-1)))


if __name__ == "__main__":
    main()
