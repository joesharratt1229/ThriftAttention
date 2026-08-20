"""Per-block quantisation-error probe for ThriftAttention.

Registers a transformers attention implementation that returns the ordinary
full-precision (baseline) attention output -- so the forward pass and NLL are
identical to the fp16 run -- while measuring, on the side, how much softmax
quantisation error each 64x64 attention tile carries and how much of that error
the ThriftAttention block selection captures at each budget.

For every layer and head it accumulates, per document:
  err_total          sum over tiles of ||P_fp16 - P_fp4||_1
  err_captured[b]    the part of err_total inside blocks selected at budget b
  mass_captured[b]   attention mass (P_fp16) inside the selected blocks -- the
                     control for "is error concentration just attention
                     concentration?"
  err_resid[b]       only with resid=True: ||P_fp16 - P_mixed||_1 where P_mixed
                     uses fp16 scores on selected blocks and fp4 scores
                     elsewhere. Capture credits thrift for error it merely
                     locates; the residual is what the mixed-precision kernel
                     actually leaves behind, since the softmax normaliser stays
                     contaminated by fp4 scores outside the selection. Costs one
                     extra full-width softmax per budget, so it is off by
                     default: use it to check that the capture-to-residual gap
                     stays flat as context length grows.
  curve_err_oracle   cumulative error at GRID fractions, tiles sorted by error
  curve_err_ranked   the same, tiles sorted by the block-mean heuristic score
  curve_mass_oracle  cumulative attention mass, tiles sorted by mass

P_fp16 and P_fp4 come from full-row softmaxes (the normaliser couples tiles, so
per-tile softmax would be wrong): scores are materialised for 64-row query
chunks against the whole causal KV width. P_fp4 uses q/k quantised by the
deployed `nvfp4_quantize` CUDA kernel and expanded exactly with
`nvfp4_dequantize` -- E2M1 x E4M3 products are exactly representable in
bf16/fp16, so both matmuls run on tensor cores with fp32 accumulation, matching
how the real kernels compute scores.

Everything is reduced online; nothing per-tile is kept unless `save_tiles` is
set (meant for one short-context document, for heatmaps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
import torch.distributed as dist

from thriftattention.quantization import nvfp4_dequantize, nvfp4_quantize
from thriftattention.selection import select_block_pairs
from thriftattention.selection.block_mean import block_means

BLOCK = 64
# fraction-of-blocks grid the cumulative curves are sampled at
GRID = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
KV_BUCKET = 2048  # causal KV widths are padded to this granularity to limit recompiles


def _chunk_stats(
    q16: torch.Tensor,       # [H, R, D] baseline-precision query rows
    q4: torch.Tensor,        # [H, R, D] dequantised nvfp4 query rows
    k16: torch.Tensor,       # [H, W, D] baseline-precision keys (GQA-expanded)
    k4: torch.Tensor,        # [H, W, D] dequantised nvfp4 keys
    keep: torch.Tensor,      # [B, H, R/64, W/64] selected-block mask per budget
    row0: int,               # global index of the first query row in this chunk
    scaling: float,
    resid_count: int,        # budgets to compute the mixed-precision residual for
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads, rows, _ = q16.shape
    width = k16.shape[1]
    s16 = torch.bmm(q16, k16.transpose(1, 2)).float() * scaling
    s4 = torch.bmm(q4, k4.transpose(1, 2)).float() * scaling
    cols = torch.arange(width, device=q16.device)
    causal = cols.unsqueeze(0) > (row0 + torch.arange(rows, device=q16.device)).unsqueeze(1)
    s16.masked_fill_(causal, float("-inf"))
    s4.masked_fill_(causal, float("-inf"))

    p16 = torch.softmax(s16, dim=-1)
    p4 = torch.softmax(s4, dim=-1)

    def tile_sum(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(heads, rows // BLOCK, BLOCK, width // BLOCK, BLOCK).sum(dim=(2, 4))

    err_tiles = tile_sum((p16 - p4).abs())
    mass_tiles = tile_sum(p16)

    # each residual budget costs another full-width softmax, so it is opt-in
    resid_tiles = []
    for b in range(resid_count):
        keep_el = keep[b].repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=2)
        p_mixed = torch.softmax(torch.where(keep_el, s16, s4), dim=-1)
        resid_tiles.append(tile_sum((p16 - p_mixed).abs()))
    resid = torch.stack(resid_tiles) if resid_tiles else err_tiles.new_zeros((0, *err_tiles.shape))
    return err_tiles, mass_tiles, resid


@dataclass
class ProbeConfig:
    budgets: tuple[float, ...]
    baseline_impl: str
    num_layers: int
    head_chunk: int = 8
    row_chunk: int = 256
    layer_stride: int = 1     # measure every n-th layer
    row_stride: int = 1       # measure every n-th row chunk
    save_tiles: bool = False  # keep the full per-tile error map for the current doc
    compile: bool = True
    resid: bool = False       # also measure the mixed-precision residual (~1/3 slower)


class ProbeState:
    """Accumulates per-(layer, head) error statistics across one document."""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self._chunk_fn = torch.compile(_chunk_stats, dynamic=True) if config.compile else _chunk_stats
        self._acc: dict[str, torch.Tensor] | None = None
        self._grid: torch.Tensor | None = None
        self.tiles: dict[int, torch.Tensor] = {}
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.baseline_fn = ALL_ATTENTION_FUNCTIONS[config.baseline_impl]

    def _ensure_acc(self, num_heads: int, device: torch.device) -> None:
        if self._acc is not None:
            return
        layers, budgets, grid = self.config.num_layers, len(self.config.budgets), len(GRID)
        shape = {
            "err_total": (layers, num_heads),
            "mass_total": (layers, num_heads),
            "err_captured": (layers, num_heads, budgets),
            "err_resid": (layers, num_heads, budgets),
            "mass_captured": (layers, num_heads, budgets),
            "curve_err_oracle": (layers, num_heads, grid),
            "curve_err_ranked": (layers, num_heads, grid),
            "curve_mass_oracle": (layers, num_heads, grid),
        }
        self._acc = {k: torch.zeros(s, dtype=torch.float64, device=device) for k, s in shape.items()}
        self._grid = torch.tensor(GRID, device=device)

    @torch.no_grad()
    def record(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, scaling: float) -> None:
        cfg = self.config
        if layer_idx % cfg.layer_stride:
            return
        if query.shape[0] != 1:
            raise RuntimeError("block error probe expects batch size 1")
        seq_len = query.shape[2]
        if seq_len % max(cfg.row_chunk, BLOCK) or cfg.row_chunk % BLOCK:
            raise RuntimeError(f"sequence and row chunk must be multiples of {BLOCK}")

        is_bf16 = query.dtype == torch.bfloat16
        q = query.contiguous()
        k = key.contiguous()
        num_heads, kv_heads = q.shape[1], k.shape[1]
        groups = num_heads // kv_heads
        self._ensure_acc(num_heads, q.device)

        q_hat = nvfp4_dequantize(*nvfp4_quantize(q, is_bf16=is_bf16), is_bf16=is_bf16)
        k_hat = nvfp4_dequantize(*nvfp4_quantize(k, is_bf16=is_bf16), is_bf16=is_bf16)

        # deployed selection per budget: [heads, q_blocks, top_k] with -1 padding
        num_blocks = seq_len // BLOCK
        selections = [
            select_block_pairs(q, k, causal=True, fraction=b, block_size=BLOCK, is_bf16=is_bf16)
            .long()
            .view(num_heads, num_blocks, -1)
            for b in cfg.budgets
        ]
        # block-mean heuristic scores, for the thrift-ranked cumulative curve
        q_mean = block_means(q, block_size=BLOCK, is_bf16=is_bf16)[0].float()
        k_mean = block_means(k, block_size=BLOCK, is_bf16=is_bf16)[0].float()
        k_mean = k_mean.repeat_interleave(groups, dim=0)

        tiles = torch.zeros(num_heads, num_blocks, num_blocks) if cfg.save_tiles else None

        for h0 in range(0, num_heads, cfg.head_chunk):
            h1 = min(h0 + cfg.head_chunk, num_heads)
            kv_index = torch.arange(h0, h1, device=q.device) // groups
            k16_g = k[0].index_select(0, kv_index)
            k4_g = k_hat[0].index_select(0, kv_index)

            for chunk_i, r0 in enumerate(range(0, seq_len, cfg.row_chunk)):
                if chunk_i % cfg.row_stride:
                    continue
                r1 = min(r0 + cfg.row_chunk, seq_len)
                width = min(seq_len, -(-r1 // KV_BUCKET) * KV_BUCKET)
                qb0, qb1, wb = r0 // BLOCK, r1 // BLOCK, width // BLOCK

                keep = torch.zeros(
                    len(cfg.budgets), h1 - h0, qb1 - qb0, wb + 1, dtype=torch.bool, device=q.device
                )
                for bi, sel in enumerate(selections):
                    sel_c = sel[h0:h1, qb0:qb1].clamp(min=-1)
                    keep[bi].scatter_(2, sel_c.masked_fill(sel_c < 0, wb).clamp(max=wb), True)
                keep = keep[..., :-1].contiguous()

                err_t, mass_t, resid_t = self._chunk_fn(
                    q[0, h0:h1, r0:r1],
                    q_hat[0, h0:h1, r0:r1],
                    k16_g[:, :width],
                    k4_g[:, :width],
                    keep,
                    r0,
                    scaling,
                    len(cfg.budgets) if cfg.resid else 0,
                )
                err_t, mass_t, resid_t = err_t.double(), mass_t.double(), resid_t.double()

                acc = self._acc
                acc["err_total"][layer_idx, h0:h1] += err_t.sum(dim=(1, 2))
                acc["mass_total"][layer_idx, h0:h1] += mass_t.sum(dim=(1, 2))
                acc["err_captured"][layer_idx, h0:h1] += (err_t * keep).sum(dim=(2, 3)).T
                if resid_t.shape[0]:
                    acc["err_resid"][layer_idx, h0:h1] += resid_t.sum(dim=(2, 3)).T
                acc["mass_captured"][layer_idx, h0:h1] += (mass_t * keep).sum(dim=(2, 3)).T

                scores = torch.bmm(q_mean[h0:h1, qb0:qb1], k_mean[h0:h1, :wb].transpose(1, 2))
                block_causal = torch.arange(wb, device=q.device).unsqueeze(0) > torch.arange(
                    qb0, qb1, device=q.device
                ).unsqueeze(1)
                scores.masked_fill_(block_causal.unsqueeze(0), float("-inf"))
                self._accumulate_curves(layer_idx, h0, h1, qb0, err_t, mass_t, scores)

                if tiles is not None:
                    tiles[h0:h1, qb0:qb1, :wb] = err_t.float().cpu()

        if tiles is not None:
            self.tiles[layer_idx] = tiles

    def _accumulate_curves(
        self,
        layer: int,
        h0: int,
        h1: int,
        qb0: int,
        err_t: torch.Tensor,   # [H, Rb, Wb] float64
        mass_t: torch.Tensor,
        scores: torch.Tensor,  # [H, Rb, Wb] heuristic scores, -inf where causal-invalid
    ) -> None:
        heads, rows_b, _ = err_t.shape
        nvalid = torch.arange(qb0 + 1, qb0 + rows_b + 1, device=err_t.device)
        idx = (torch.ceil(self._grid.unsqueeze(0) * nvalid.unsqueeze(1)) - 1).clamp(min=0).long()
        idx = idx.unsqueeze(0).expand(heads, -1, -1)  # [H, Rb, G]

        def sample(cumsum: torch.Tensor) -> torch.Tensor:
            return cumsum.gather(-1, idx).sum(dim=1)  # sum over rows -> [H, G]

        acc = self._acc
        acc["curve_err_oracle"][layer, h0:h1] += sample(
            err_t.sort(dim=-1, descending=True).values.cumsum(dim=-1)
        )
        acc["curve_mass_oracle"][layer, h0:h1] += sample(
            mass_t.sort(dim=-1, descending=True).values.cumsum(dim=-1)
        )
        order = scores.argsort(dim=-1, descending=True)
        acc["curve_err_ranked"][layer, h0:h1] += sample(err_t.gather(-1, order).cumsum(dim=-1))

    def finish_doc(self) -> dict[str, "object"]:
        """Gather over tensor-parallel ranks, return numpy arrays (rank 0), reset."""
        if self._acc is None:
            raise RuntimeError("probe recorded nothing; was the probe attention active?")
        gathered = {}
        world = dist.get_world_size() if dist.is_initialized() else 1
        for name, tensor in self._acc.items():
            if world > 1:
                parts = [torch.empty_like(tensor) for _ in range(world)]
                dist.all_gather(parts, tensor)
                tensor = torch.cat(parts, dim=1)  # heads are sharded contiguously under TP
            gathered[name] = tensor.cpu().numpy()
        for name, tensor in self._acc.items():
            tensor.zero_()
        for layer, tiles in self.tiles.items():
            gathered[f"tiles_layer{layer}"] = tiles.numpy()
        self.tiles = {}
        return gathered


def probe_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    *,
    state: ProbeState,
    **kwargs,
):
    if query.shape[2] > 1:  # measure prefill only; decode steps pass straight through
        state.record(module.layer_idx, query, key, scaling if scaling is not None else query.shape[-1] ** -0.5)
    return state.baseline_fn(module, query, key, value, attention_mask, scaling=scaling, **kwargs)


def register_probe_attention(state: ProbeState) -> str:
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import flash_attention_mask

    name = "block_error_probe"
    if name not in AttentionInterface._global_mapping:
        AttentionInterface.register(name, partial(probe_attention_forward, state=state))
        AttentionMaskInterface.register(name, flash_attention_mask)
    return name
