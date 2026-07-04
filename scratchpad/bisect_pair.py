import os, sys, math, torch
sys.path.insert(0, "/workspace/ThriftAttention/sm100-nvfp4/sketch")
from profile_breakdown import build_variant
name, defines = sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else []
base = ["FA4_ABLATE_SLOAD=1", "FA4_ABLATE_REDMAX=1", "FA4_ABLATE_CONVERT=1", "FA4_ABLATE_RESCALE=1"]
ext = build_variant(name, base + defines)
torch.manual_seed(0)
q = torch.randn(1, 1, 512, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 1, 512, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 1, 512, 128, device="cuda", dtype=torch.bfloat16)
scale = 1.0 / math.sqrt(128)
os.environ["FA4_FORCE_PAIR"] = "0"
pre = ext.quantise_and_attention(q, k, v, scale)
args = (pre["q_fp4"], pre["k_fp4"], pre["v_t_fp4"], pre["q_sf_atoms"], pre["k_sf_atoms"], pre["v_sf_atoms"])
os.environ["FA4_FORCE_PAIR"] = "1"
out = ext.attention_only(*args, scale)
torch.cuda.synchronize()
print(f"{name}: NO TRAP")
