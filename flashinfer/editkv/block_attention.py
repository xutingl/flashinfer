"""Block-level attention score computation for EditKV.

During prefill, captures block-level attention statistics that indicate
which blocks are most influenced by each key block. This information is
used to determine which blocks need recomputation when tokens are edited.

Two computation methods:
1. Full attention score computation (exact, O(N^2) per layer)
2. LSE-based approximation using FlashInfer's return_lse (approximate, O(N) extra)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from flashinfer.page import append_paged_kv_cache


@dataclass
class BlockAttentionScores:
    """Block-level attention map: for each key block, the top-k query blocks
    with highest aggregate attention.

    Attributes:
        block_indices: [num_layers, num_blocks, top_k] int32 — top-k block indices per key block.
        block_scores: [num_layers, num_blocks, top_k] float32 — attention scores.
        num_blocks: Total number of blocks.
        block_size: Tokens per block.
        top_k: Number of top attending blocks stored.
        device: Storage device ("cuda" or "cpu").
    """
    block_indices: torch.Tensor
    block_scores: torch.Tensor
    num_blocks: int
    block_size: int
    top_k: int
    device: str = "cuda"

    def get_affected_blocks(
        self,
        edit_block_indices: torch.LongTensor,
        union_across_layers: bool = True,
    ) -> torch.LongTensor:
        """Get blocks affected by edits to the given blocks.

        Args:
            edit_block_indices: 1-D tensor of edited block indices.
            union_across_layers: If True, union across all layers.

        Returns:
            Sorted 1-D tensor of unique affected block indices.
        """
        indices = self.block_indices
        if indices.device.type == "cpu":
            indices = indices.to(edit_block_indices.device)

        if union_across_layers:
            affected_parts = []
            for l in range(indices.shape[0]):
                affected_parts.append(indices[l, edit_block_indices.long()].flatten())
            affected = torch.cat(affected_parts)
        else:
            affected = indices[:, edit_block_indices.long()].flatten()

        affected = affected[affected > 0].unique()
        affected, _ = affected.sort()
        return affected.long()

    def to_cpu(self) -> "BlockAttentionScores":
        """Offload to CPU (pinned memory for fast transfer back)."""
        if self.device == "cpu":
            return self
        return BlockAttentionScores(
            block_indices=self.block_indices.to("cpu", non_blocking=True).pin_memory(),
            block_scores=self.block_scores.to("cpu", non_blocking=True).pin_memory(),
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            top_k=self.top_k,
            device="cpu",
        )

    def to_gpu(self, device: str = "cuda") -> "BlockAttentionScores":
        """Load back to GPU."""
        if self.device != "cpu":
            return self
        return BlockAttentionScores(
            block_indices=self.block_indices.to(device, non_blocking=True),
            block_scores=self.block_scores.to(device, non_blocking=True),
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            top_k=self.top_k,
            device=device,
        )

    def memory_bytes(self) -> int:
        return (
            self.block_indices.nelement() * self.block_indices.element_size()
            + self.block_scores.nelement() * self.block_scores.element_size()
        )


def compute_block_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    num_layers: int,
    layer_idx: int,
    block_size: int = 16,
    top_k: int = 8,
    aggregation: str = "max",
    existing_scores: Optional[BlockAttentionScores] = None,
    causal: bool = True,
) -> BlockAttentionScores:
    """Compute block-level attention scores for a single layer.

    This function computes the block-level attention map for one layer during
    or after prefill. Call it for each layer to build the full map.

    Args:
        q: Query tensor [seq_len, num_heads, head_dim] (after RoPE).
        k: Key tensor [seq_len, num_kv_heads, head_dim] (after RoPE).
        num_layers: Total number of layers (for allocating the output).
        layer_idx: Current layer index.
        block_size: Tokens per block.
        top_k: Number of top attending blocks to store per key block.
        aggregation: "max" or "mean" within block.
        existing_scores: If provided, update this scores object at layer_idx.
        causal: Whether to apply causal mask.

    Returns:
        BlockAttentionScores with this layer's data filled in.
    """
    device = q.device
    seq_len = q.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    num_blocks = (seq_len + block_size - 1) // block_size
    actual_top_k = min(top_k, max(num_blocks - 1, 1))

    # GQA: repeat KV heads to match Q heads
    num_kv_groups = num_heads // num_kv_heads
    if num_kv_groups > 1:
        k_expanded = k.repeat_interleave(num_kv_groups, dim=1)
    else:
        k_expanded = k

    # Compute attention scores: [num_heads, seq_len, seq_len]
    # q: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
    q_t = q.transpose(0, 1)
    k_t = k_expanded.transpose(0, 1)
    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim ** 0.5)

    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

    # Softmax + average over heads: [seq_len, seq_len]
    attn_weights = torch.softmax(attn_scores, dim=-1).mean(dim=0)

    # Aggregate to block level: [num_blocks, num_blocks]
    block_attn = torch.zeros(num_blocks, num_blocks, device=device)
    for bq in range(num_blocks):
        q_start = bq * block_size
        q_end = min(q_start + block_size, seq_len)
        for bk in range(bq):  # Causal: only earlier blocks
            k_start = bk * block_size
            k_end = min(k_start + block_size, seq_len)
            chunk = attn_weights[q_start:q_end, k_start:k_end]
            if aggregation == "max":
                block_attn[bq, bk] = chunk.max()
            else:
                block_attn[bq, bk] = chunk.mean()

    # Initialize or update the scores object
    if existing_scores is None:
        existing_scores = BlockAttentionScores(
            block_indices=torch.zeros(num_layers, num_blocks, actual_top_k, dtype=torch.int32, device=device),
            block_scores=torch.zeros(num_layers, num_blocks, actual_top_k, dtype=torch.float32, device=device),
            num_blocks=num_blocks,
            block_size=block_size,
            top_k=actual_top_k,
            device=str(device),
        )

    # Extract top-k per key block
    for bk in range(num_blocks):
        attending = block_attn[:, bk].clone()
        attending[:bk + 1] = float("-inf")
        valid_count = (attending != float("-inf")).sum().item()
        k_actual = min(actual_top_k, valid_count)
        if k_actual > 0:
            topk_vals, topk_idx = attending.topk(k_actual)
            existing_scores.block_indices[layer_idx, bk, :k_actual] = topk_idx.int()
            existing_scores.block_scores[layer_idx, bk, :k_actual] = topk_vals

    return existing_scores


def compute_block_attention_from_lse(
    lse: torch.Tensor,
    block_size: int = 16,
    top_k: int = 8,
    num_layers: int = 1,
    layer_idx: int = 0,
    existing_scores: Optional[BlockAttentionScores] = None,
) -> BlockAttentionScores:
    """Approximate block attention scores from log-sum-exp values.

    This is a fast approximation that uses FlashInfer's return_lse=True output.
    The LSE values indicate the "softness" of attention at each position — higher
    LSE means attention is more spread out, lower means more concentrated.

    This method is approximate but much faster than full attention computation.

    Args:
        lse: [seq_len, num_heads] log-sum-exp from FlashInfer prefill.
        block_size: Tokens per block.
        top_k: Top-k blocks to store.
        num_layers: Total layers.
        layer_idx: Current layer.
        existing_scores: Existing scores to update.

    Returns:
        BlockAttentionScores (approximate).
    """
    device = lse.device
    seq_len = lse.shape[0]
    num_blocks = (seq_len + block_size - 1) // block_size
    actual_top_k = min(top_k, max(num_blocks - 1, 1))

    # Average LSE across heads: [seq_len]
    lse_avg = lse.mean(dim=-1)

    # Block-level LSE: higher LSE in a block means it attends broadly
    # We use this as a proxy for "importance" — blocks with high LSE query blocks
    # are likely attending to many key blocks
    block_lse = torch.zeros(num_blocks, device=device)
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, seq_len)
        block_lse[b] = lse_avg[start:end].mean()

    if existing_scores is None:
        existing_scores = BlockAttentionScores(
            block_indices=torch.zeros(num_layers, num_blocks, actual_top_k, dtype=torch.int32, device=device),
            block_scores=torch.zeros(num_layers, num_blocks, actual_top_k, dtype=torch.float32, device=device),
            num_blocks=num_blocks,
            block_size=block_size,
            top_k=actual_top_k,
            device=str(device),
        )

    # For LSE-based approximation, each key block's "importance" is proportional
    # to how much query blocks after it have elevated LSE
    # This is a rough heuristic — the full attention computation is more accurate
    for bk in range(num_blocks):
        # Query blocks that come after this key block
        candidate_scores = block_lse.clone()
        candidate_scores[:bk + 1] = float("-inf")
        valid_count = (candidate_scores != float("-inf")).sum().item()
        k_actual = min(actual_top_k, valid_count)
        if k_actual > 0:
            topk_vals, topk_idx = candidate_scores.topk(k_actual)
            existing_scores.block_indices[layer_idx, bk, :k_actual] = topk_idx.int()
            existing_scores.block_scores[layer_idx, bk, :k_actual] = topk_vals

    return existing_scores
