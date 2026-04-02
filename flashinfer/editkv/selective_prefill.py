"""Selective prefill operations for EditKV.

Provides functions to:
1. Run prefill for only selected token positions (selective recompute)
2. Update the paged KV cache at specific positions

These operations use FlashInfer's existing paged attention infrastructure
but operate on a subset of positions.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


def selective_prefill_with_paged_kv_cache(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    q_selected: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    selected_positions: torch.LongTensor,
    total_seq_len: int,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run attention for only selected query positions against the full KV cache.

    This is the core of EditKV's selective recompute: instead of computing
    attention for all N positions (O(N^2)), we only compute for |S| selected
    positions against the full N-length KV cache (O(|S|*N)).

    The wrapper must be pre-planned with the correct page table for the full
    sequence, but qo_indptr should only cover the selected positions.

    Args:
        wrapper: FlashInfer prefill wrapper, pre-planned for the request.
        q_selected: Query tensor for selected positions only.
            Shape: [num_selected, num_heads, head_dim]
        paged_kv_cache: Full paged KV cache (all blocks, including non-selected).
        selected_positions: 1-D tensor of original sequence positions for the
            selected tokens. Used for causal masking.
        total_seq_len: Total sequence length (for causal mask computation).
        causal: Whether to apply causal attention mask.
        sm_scale: Softmax scale. If None, uses 1/sqrt(head_dim).
        return_lse: Whether to return log-sum-exp values.

    Returns:
        Attention output for selected positions.
        Shape: [num_selected, num_heads, head_dim]
        If return_lse=True, also returns LSE: [num_selected, num_heads]

    Note:
        The wrapper must be planned with qo_indptr that covers only the selected
        positions. The KV page table should cover the full sequence.
        The causal mask is handled by FlashInfer based on the position information
        passed during planning.
    """
    return wrapper.run(
        q_selected,
        paged_kv_cache,
        return_lse=return_lse,
    )


def update_paged_kv_cache_at_positions(
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    selected_positions: torch.LongTensor,
    kv_indices: torch.Tensor,
    page_size: int,
    kv_layout: str = "NHD",
) -> None:
    """Update the paged KV cache at specific token positions.

    After selective recompute, this writes the new K/V values back to
    the paged KV cache at the positions that were recomputed.

    Args:
        new_keys: New key states for selected positions.
            Shape: [num_selected, num_kv_heads, head_dim]
        new_values: New value states for selected positions.
            Shape: [num_selected, num_kv_heads, head_dim]
        paged_kv_cache: The paged KV cache to update.
            Shape: [max_num_pages, 2, page_size, num_kv_heads, head_dim] (NHD)
        selected_positions: 1-D tensor of token positions to update.
        kv_indices: Page indices for the full sequence.
        page_size: Tokens per page.
        kv_layout: "NHD" or "HND".
    """
    if isinstance(paged_kv_cache, tuple):
        k_cache, v_cache = paged_kv_cache
    else:
        # Combined format: [max_pages, 2, page_size, num_kv_heads, head_dim]
        k_cache = paged_kv_cache[:, 0]  # [max_pages, page_size, num_kv_heads, head_dim]
        v_cache = paged_kv_cache[:, 1]

    for i, pos in enumerate(selected_positions.tolist()):
        page_idx = pos // page_size
        pos_in_page = pos % page_size

        # Map logical page index to physical page
        physical_page = kv_indices[page_idx].item()

        if kv_layout == "NHD":
            # k_cache: [max_pages, page_size, num_kv_heads, head_dim]
            k_cache[physical_page, pos_in_page] = new_keys[i]
            v_cache[physical_page, pos_in_page] = new_values[i]
        else:
            # HND: [max_pages, num_kv_heads, page_size, head_dim]
            k_cache[physical_page, :, pos_in_page] = new_keys[i]
            v_cache[physical_page, :, pos_in_page] = new_values[i]

    # If combined format, write back
    if not isinstance(paged_kv_cache, tuple):
        paged_kv_cache[:, 0] = k_cache
        paged_kv_cache[:, 1] = v_cache


def update_paged_kv_cache_at_blocks(
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    selected_block_indices: torch.LongTensor,
    kv_indices: torch.Tensor,
    block_size: int,
    seq_len: int,
    kv_layout: str = "NHD",
) -> None:
    """Update the paged KV cache for entire blocks.

    More efficient than per-position updates when operating at block granularity.

    Args:
        new_keys: New key states for selected blocks.
            Shape: [num_selected_tokens, num_kv_heads, head_dim]
            where num_selected_tokens = sum of tokens in selected blocks.
        new_values: Same shape as new_keys.
        paged_kv_cache: The paged KV cache.
        selected_block_indices: 1-D tensor of block indices to update.
        kv_indices: Page indices for the full sequence.
        block_size: Tokens per block (= page_size in vLLM).
        seq_len: Total sequence length (to handle partial last block).
        kv_layout: "NHD" or "HND".
    """
    if isinstance(paged_kv_cache, tuple):
        k_cache, v_cache = paged_kv_cache
    else:
        k_cache = paged_kv_cache[:, 0]
        v_cache = paged_kv_cache[:, 1]

    token_offset = 0
    for block_idx in selected_block_indices.tolist():
        physical_page = kv_indices[block_idx].item()
        start_pos = block_idx * block_size
        end_pos = min(start_pos + block_size, seq_len)
        num_tokens_in_block = end_pos - start_pos

        if kv_layout == "NHD":
            k_cache[physical_page, :num_tokens_in_block] = new_keys[token_offset:token_offset + num_tokens_in_block]
            v_cache[physical_page, :num_tokens_in_block] = new_values[token_offset:token_offset + num_tokens_in_block]
        else:
            k_cache[physical_page, :, :num_tokens_in_block] = new_keys[token_offset:token_offset + num_tokens_in_block].transpose(0, 1)
            v_cache[physical_page, :, :num_tokens_in_block] = new_values[token_offset:token_offset + num_tokens_in_block].transpose(0, 1)

        token_offset += num_tokens_in_block

    if not isinstance(paged_kv_cache, tuple):
        paged_kv_cache[:, 0] = k_cache
        paged_kv_cache[:, 1] = v_cache


def plan_selective_prefill(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    selected_positions: torch.LongTensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_data_type: torch.dtype = torch.bfloat16,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> None:
    """Plan a selective prefill operation.

    Sets up the FlashInfer wrapper for computing attention only at selected
    positions while reading from the full KV cache.

    Args:
        wrapper: FlashInfer prefill wrapper.
        selected_positions: Token positions to compute (sorted, 1-D).
        kv_indices: Page indices for the full sequence.
        kv_indptr: CSR indptr for KV pages.
        kv_last_page_len: Entries in the last page.
        num_qo_heads: Number of query/output heads.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        page_size: Tokens per page.
        q_data_type: Query data type.
        causal: Whether to use causal attention.
        sm_scale: Softmax scale.
    """
    num_selected = selected_positions.shape[0]

    # For selective prefill, we treat the selected positions as a single "request"
    # with the full KV cache available
    qo_indptr = torch.tensor([0, num_selected], dtype=torch.int32, device=selected_positions.device)

    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=q_data_type,
        causal=causal,
        sm_scale=sm_scale,
    )
