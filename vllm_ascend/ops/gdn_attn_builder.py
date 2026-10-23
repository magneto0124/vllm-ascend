# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_final_chunk_indices,
    prepare_update_chunk_offsets,
)
from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time, tree_timer_enabled
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import selected_gdn_state_update_name
from vllm_ascend.worker.v2.spec_decode.tree.treewy import treewy_ancestor_mask

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18
logger = logging.getLogger(__name__)


def _tree_profiled_call(segment: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if not tree_timer_enabled():
        return fn(*args, **kwargs)
    with tree_time(segment):
        return fn(*args, **kwargs)


def _stable_argsort_for_npu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.int32)
    return torch.argsort(tensor, stable=True)


def _treat_single_token_prefills_with_state_as_decodes(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """Use decode metadata for one-token stateful prompt chunks.

    A final one-token prompt chunk can replay the same fixed graph as an
    ordinary decode. Once recurrent state exists, both paths must construct
    identical GDN metadata so the graph consumes the current state indices.
    First-token prefills remain on the prefill path because they have no state
    to update.
    """
    is_prefilling = common_attn_metadata.is_prefilling
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    if is_prefilling is None or seq_lens_cpu is None:
        return common_attn_metadata

    query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
    prefill_to_decode = is_prefilling & (query_lens_cpu == 1) & (seq_lens_cpu > 1)
    if not torch.any(prefill_to_decode).item():
        return common_attn_metadata

    is_prefilling = is_prefilling.clone()
    is_prefilling[prefill_to_decode] = False
    return common_attn_metadata.replace(is_prefilling=is_prefilling)


@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    num_decodes: int = 0
    cu_seqlens_kern: tuple[int, ...] | None = None
    keep_meta: torch.Tensor | None = None


@dataclass
class GDNCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    initial_state_mode: torch.Tensor | None


@dataclass
class GDNSpecCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor


@dataclass
class GDNPrefillMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    chunk: GDNChunkedPrefillMetadata


@dataclass
class GDNDecodeMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    actual_seq_lengths: torch.Tensor


@dataclass
class GDNSpecDecodeMetadata:
    spec_causal_conv1d: GDNSpecCausalConv1dMetadata
    actual_seq_lengths: torch.Tensor
    # Draft-tree only: per-token state page the recurrent kernel loads its
    # initial state from, see spec_decode/tree/state_rollback.py. ``None`` keeps
    # the linear-chain behaviour of the kernel.
    init_state_indices: torch.Tensor | None = None
    # Draft-tree only: conv tap rows (``[tokens, width]``) and the conv-state
    # columns (``[rows, width - 1]``) holding the committed history the next step
    # resumes from. ``None`` keeps the sliding-window custom operator.
    conv_window_rows: torch.Tensor | None = None
    conv_prefix_src: torch.Tensor | None = None
    # Draft-tree only: chunked verify of a tree wider than the recurrent kernel's
    # per-call width. ``None`` keeps the plain single-call path (see
    # spec_decode/tree/state_rollback.TreeVerifyPlan).
    verify_plan: "TreeVerifyPlan | None" = None
    # Tree layout has three independent widths: queries verified per request,
    # recurrent state pages, and short-conv node activations retained per page.
    # These are equal for ``snapshot`` but TreeWY uses a single recurrent page.
    gdn_state_update_method: str | None = None
    tree_verify_slots: int | None = None
    tree_recurrent_state_slots: int | None = None
    tree_conv_slots: int | None = None
    tree_parents: torch.Tensor | None = None
    tree_num_nodes: torch.Tensor | None = None
    tree_ancestors: torch.Tensor | None = None
    tree_row_widths: torch.Tensor | None = None
    tree_sequence_indices: torch.Tensor | None = None
    # Eager TreeWY stores each layer's recurrence scratch until sampling. In
    # graph mode these references point at the captured static outputs.
    treewy_contexts: dict[str, object] | None = None


def _tree_conv_width(kv_cache_spec: AttentionSpec, mamba_type, state_columns: int) -> int | None:
    """Short-conv kernel width of a GDN state spec (``None`` when not GDN).

    ``state_columns`` is the number of speculative state columns the conv-state
    shape carries: ``num_speculative_tokens`` while the spec still has the
    linear-chain layout, ``slots`` once ``attn_utils`` widened it to the
    draft-tree width.
    """
    shapes = getattr(kv_cache_spec, "shapes", None)
    if not shapes:
        return None
    # Lazy import: the tree spec-decode package imports this module.
    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import tree_conv_width

    return tree_conv_width(mamba_type, tuple(shapes[0]), state_columns)


def _pack_tree_rows_to_tokens(
    table: torch.Tensor,
    row_widths: torch.Tensor,
    slots: int,
) -> torch.Tensor:
    """Pack a ``[rows, slots]`` tree table into token-major order.

    The recurrent kernel reads ``ssm_state_indices`` and ``init_state_indices``
    by global token offset (``ssmStateIndicesGm_.GetValue(seq_i)`` in
    ``csrc/attention/recurrent_gated_delta_rule``), while the tree tables are
    indexed as ``[row, node]``. Every row normally verifies the full
    ``1 + budget = slots`` tokens, but the scheduler truncates the draft of a
    request near ``max_model_len`` and the tree search can run out of candidates,
    so the rows have to be laid out back to back: token ``p`` of the step takes
    node ``p - start(row)`` of its row. The result keeps the ``[rows, slots]``
    shape because the op receives ``reshape(-1)``, and only the slots past the
    packed tokens hold ``NULL_BLOCK_ID`` -- the kernel never reaches them.
    """
    rows = table.shape[0]
    # The usual tree decode verifies every slot. Its row-major table is already
    # token-major, so avoid launching the gather/scan sequence in this case.
    if (
        row_widths.device.type == "cpu"
        and row_widths.numel() == rows
        and bool(torch.all(row_widths == slots))
    ):
        return table
    widths = row_widths.to(device=table.device, dtype=torch.long).reshape(-1)
    if widths.numel() != rows:
        raise ValueError(f"got {widths.numel()} row widths for {rows} tree rows.")
    starts = torch.cumsum(widths, dim=0) - widths
    ends = starts + widths
    positions = torch.arange(rows * slots, device=table.device)
    token_rows = (positions.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1).clamp(max=rows - 1)
    token_cols = (positions - starts[token_rows]).clamp(max=slots - 1)
    packed = table.reshape(-1).index_select(0, token_rows * slots + token_cols)
    packed = packed.masked_fill(positions >= ends[-1], NULL_BLOCK_ID)
    return packed.view(rows, slots)


def _build_actual_seq_lengths(
    query_start_loc: torch.Tensor,
    num_sequences: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    actual_seq_lengths = (
        torch.empty_like(query_start_loc[: num_sequences + 1]) if out is None else out[: num_sequences + 1]
    )
    actual_seq_lengths[:1].copy_(query_start_loc[:1])
    torch.sub(
        query_start_loc[1 : num_sequences + 1],
        query_start_loc[:num_sequences],
        out=actual_seq_lengths[1:],
    )
    return actual_seq_lengths


def _compact_empty_segments(cu_seqlens_host, initial_state, device=None):
    """Drop zero-length segments so AscendC fwd_h/fwd_o indexing lines up.

    Returns ``(cu_seqlens_kern, initial_state_kern, keep_meta)``:
    cu_seqlens / initial_state with empty segments removed, plus a bool
    mask (None when nothing was removed).  The compacted ``final_state``
    must be scattered back via ``keep_meta`` (empty segments keep their
    initial state).

    When *device* is given, ``keep_meta`` is moved to that device so that
    callers can index NPU tensors without an extra host→device sync.
    """
    if cu_seqlens_host is None:
        return None, initial_state, None
    cu = torch.tensor(cu_seqlens_host, dtype=torch.int64)
    keep = (cu[1:] - cu[:-1]) > 0
    if bool(keep.all()):
        return cu_seqlens_host, initial_state, None
    # Compute compact cu_seqlens while keep is still on CPU (cu is CPU-only).
    cu_kern = torch.cat([cu[:1], cu[1:][keep]]).tolist()
    # Move keep to device only for indexing device-side tensors.
    if device is not None:
        keep = keep.to(device)
    st_kern = initial_state[keep] if initial_state is not None else None
    return cu_kern, st_kern, keep


def _build_non_spec_chunked_prefill_metadata(
    builder,
    cu_seqlens_cpu: torch.Tensor,
    device: torch.device,
) -> GDNChunkedPrefillMetadata:
    hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
    linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("num_heads") is not None:
        gdn_num_heads = linear_attn_config["num_heads"] // builder.vllm_config.parallel_config.tensor_parallel_size
    elif hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
        gdn_num_heads = (
            hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
        )
    else:
        gdn_num_heads = builder.vllm_config.model_config.get_num_attention_heads(builder.vllm_config.parallel_config)
    cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * _GDN_CHUNK_SIZE))
    cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()

    chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    chunk_offsets_chunk64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    update_chunk_offsets_chunk64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    final_chunk_indices_chunk64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    chunk_indices_large_block = prepare_chunk_indices(
        cu_seqlens_cpu,
        _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
    )
    block_indices_cumsum = prepare_chunk_indices(cu_seqlens_cpu, cumsum_chunk_size)

    cu_seqlens_host = tuple(cu_seqlens_cpu.to(torch.int64).reshape(-1).tolist())
    # Pre-compute compact cu_seqlens for AscendC kernels so each layer
    # can reuse them instead of calling _compact_empty_segments again.
    cu_seqlens_kern, _, keep_meta = _compact_empty_segments(cu_seqlens_host, None, device=device)
    if keep_meta is None:
        cu_seqlens_kern = None
    else:
        cu_seqlens_kern = tuple(cu_seqlens_kern)

    return GDNChunkedPrefillMetadata(
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_chunk64_host=tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist()),
        chunk_indices_chunk64=chunk_indices_chunk64.to(device=device, non_blocking=True),
        chunk_offsets_chunk64=chunk_offsets_chunk64.to(device=device, non_blocking=True),
        update_chunk_offsets_chunk64=update_chunk_offsets_chunk64.to(device=device, non_blocking=True),
        final_chunk_indices_chunk64=final_chunk_indices_chunk64.to(device=device, non_blocking=True),
        chunk_indices_large_block=chunk_indices_large_block.to(device=device, non_blocking=True),
        block_indices_cumsum=block_indices_cumsum.to(device=device, non_blocking=True),
        cu_seqlens_kern=cu_seqlens_kern,
        keep_meta=keep_meta,
    )


class AscendGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Draft-tree state layout, filled in by _resize_spec_buffers_for_tree;
        # ``None`` means the linear MTP chain (default).
        self.spec_tree_slots: int | None = None
        self.spec_init_state_indices: torch.Tensor | None = None
        self.spec_conv_width: int | None = None
        self.spec_conv_window_rows: torch.Tensor | None = None
        self.spec_conv_prefix_src: torch.Tensor | None = None
        # Stable chunk tensors of a verify wider than one recurrent-op call, see
        # _build_tree_verify_plan; ``None`` while the tree fits a single call.
        self.spec_verify_plan_buffers: "TreeVerifyPlanBuffers | None" = None
        self._resize_spec_buffers_for_tree(device)
        sequence_index_capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.decode_cudagraph_max_bs,
        )
        self._tree_state_table_graphs: dict[int, Any] = {}
        self._tree_state_table_graph_failures: set[int] = set()
        self.spec_tree_table_state_pages: torch.Tensor | None = None
        self.spec_tree_table_resume_slots: torch.Tensor | None = None
        self.spec_tree_table_prev_path: torch.Tensor | None = None
        self.spec_tree_table_prev_num_sampled: torch.Tensor | None = None
        if (
            self.use_full_cuda_graph
            and self.spec_tree_slots is not None
            and self.spec_conv_width is not None
            and selected_gdn_state_update_name() == "snapshot"
        ):
            self.spec_tree_table_state_pages = torch.empty(
                (sequence_index_capacity, self.spec_tree_slots), dtype=torch.int32, device=device
            )
            self.spec_tree_table_resume_slots = torch.empty(
                (sequence_index_capacity,), dtype=torch.int32, device=device
            )
            self.spec_tree_table_prev_path = torch.empty(
                (sequence_index_capacity, self.spec_tree_slots - 1), dtype=torch.int32, device=device
            )
            self.spec_tree_table_prev_num_sampled = torch.empty(
                (sequence_index_capacity,), dtype=torch.int32, device=device
            )

        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (sequence_index_capacity,), dtype=torch.bool, device=device
        )

        self.spec_sequence_masks_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        )

        self.spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.non_spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.spec_tree_parents: torch.Tensor | None = None
        self.spec_tree_num_nodes: torch.Tensor | None = None
        self.spec_tree_ancestors: torch.Tensor | None = None
        if self.spec_tree_slots is not None:
            self.spec_tree_parents = torch.empty(
                (sequence_index_capacity, self.spec_tree_slots - 1), dtype=torch.int32, device=device
            )
            self.spec_tree_num_nodes = torch.empty((sequence_index_capacity,), dtype=torch.int32, device=device)
            if selected_gdn_state_update_name() == "treewy":
                self.spec_tree_ancestors = torch.empty(
                    (sequence_index_capacity, self.spec_tree_slots, self.spec_tree_slots),
                    dtype=torch.bool,
                    device=device,
                )
        self.non_spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.spec_actual_seq_lengths: torch.Tensor = torch.empty(
            (sequence_index_capacity + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_actual_seq_lengths: torch.Tensor = torch.empty(
            (sequence_index_capacity + 1,),
            dtype=torch.int32,
            device=device,
        )

    def _resize_spec_buffers_for_tree(self, device: torch.device) -> None:
        """Follow the draft-tree state width in the spec-sized buffers.

        ``GDNAttentionMetadataBuilder.__init__`` sizes every spec-sized buffer
        with ``1 + num_speculative_tokens``, the number of queries of a linear
        draft chain, but a draft tree verifies ``1 + budget`` tokens per request
        (``tree_spec_config.budget >= num_speculative_tokens``) and caches one
        state per node. ``MambaSpec`` is widened to the same width by
        ``attn_utils.get_kv_cache_spec``. The tensors allocated below mirror the
        upstream constructor and have to stay in sync with it.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            tree_verify_slot_count,
            validate_tree_state_slots,
        )

        slots = tree_verify_slot_count(self.vllm_config)
        if slots is None:
            return
        mamba_type = getattr(self.kv_cache_spec, "mamba_type", None)
        if mamba_type is not None:
            validate_tree_state_slots(slots, mamba_type)
        self.spec_tree_slots = slots
        self.spec_init_state_indices = torch.empty(
            (self.decode_cudagraph_max_bs, slots), dtype=torch.int32, device=device
        )
        # ``attn_utils.get_kv_cache_spec`` widens the conv-state shape from the
        # linear-chain ``num_spec`` state columns to the tree's ``slots`` ones as
        # soon as the tree is wider than the chain, so the kernel width has to be
        # recovered with the column count the shape actually carries.
        state_columns = max(self.num_spec, slots)
        conv_width = _tree_conv_width(self.kv_cache_spec, mamba_type, state_columns)
        self.spec_conv_width = conv_width
        if conv_width is not None:
            # Row tables of the tree short conv, indexed by the spec-sized batch
            # and, for the windows, by its tokens; see _build_tree_conv_rows.
            self.spec_conv_window_rows = torch.empty(
                (self.decode_cudagraph_max_bs * slots, conv_width), dtype=torch.int32, device=device
            )
            self.spec_conv_prefix_src = torch.empty(
                (self.decode_cudagraph_max_bs, conv_width - 1), dtype=torch.int32, device=device
            )
        if slots == self.num_spec + 1:
            # The tree is exactly as wide as the draft chain, so the buffers the
            # upstream constructor allocated already carry its width. The verify
            # may still outgrow one op call, hence the chunk buffers first.
            self._allocate_tree_verify_plan_buffers(device, slots)
            return

        self.num_spec = slots - 1
        self.decode_cudagraph_max_bs = self.vllm_config.scheduler_config.max_num_seqs * slots
        max_capture_size = self.compilation_config.max_cudagraph_capture_size
        if max_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs, max_capture_size
            )

        max_bs = self.decode_cudagraph_max_bs
        self.spec_state_indices_tensor = torch.empty(
            (max_bs, slots), dtype=torch.int32, device=device
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (max_bs,), dtype=torch.int32, device=device
        )
        self.spec_token_indx = torch.empty(
            (max_bs * slots,), dtype=torch.int32, device=device
        )
        self.non_spec_token_indx = torch.empty(
            (max_bs * slots,), dtype=torch.int32, device=device
        )
        self.spec_query_start_loc = torch.empty(
            (max_bs + 1,), dtype=torch.int32, device=device
        )
        self.non_spec_query_start_loc = torch.empty(
            (max_bs + 1,), dtype=torch.int32, device=device
        )
        self.num_accepted_tokens = torch.empty(
            (max_bs,), dtype=torch.int32, device=device
        )
        if self.use_spec_decode and self.reorder_batch_threshold != 1:
            # The threshold counts the queries of one verification batch, and a
            # tree verification always runs the root plus every node.
            self.reorder_batch_threshold = slots

        self._allocate_tree_verify_plan_buffers(device, slots)

    def _allocate_tree_verify_plan_buffers(self, device: torch.device, slots: int) -> None:
        """Stable chunk tensors of a verify wider than one recurrent-op call.

        A tree wider than the recurrent kernel's per-call width is verified in
        chunks of that width (see :meth:`_build_tree_verify_plan`). A full graph
        replays the recorded chunks, so they have to be the same addresses with
        fresh values every step: they are allocated here, once, for the graph's
        row capacity, exactly like the state tables. An eager verify rebuilds its
        chunk tensors on every step and needs no buffers of its own.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            GDN_RECURRENT_MAX_VERIFY_TOKENS,
            allocate_tree_verify_plan_buffers,
        )

        if slots <= GDN_RECURRENT_MAX_VERIFY_TOKENS or not self.use_full_cuda_graph:
            return
        self.spec_verify_plan_buffers = allocate_tree_verify_plan_buffers(
            slots, self.decode_cudagraph_max_bs, device
        )

    def _build_tree_verify_plan(
        self,
        row_widths: torch.Tensor,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor | None,
        scratch_base: int,
    ) -> "TreeVerifyPlan | None":
        """Chunk plan of a verify step wider than one recurrent-op call.

        ``None`` -- the common case -- keeps the plain single-call path. A wider
        tree is cut into ``GDN_RECURRENT_MAX_VERIFY_TOKENS``-wide chunks of node
        columns which are chained through the state pages, so the op never
        verifies more tokens per request than its kernel buffers hold (above that
        the kernel returns *without updating the state*, i.e. silently stale).
        See ``spec_decode/tree/state_rollback.compute_tree_verify_plan``.

        A full graph replays the recorded chunks, so it needs the chunk tensors to
        be the same addresses with fresh values: the plan is copied into the
        stable buffers allocated by :meth:`_allocate_tree_verify_plan_buffers`.
        ``row_widths`` has one entry per op row, i.e. including the padded rows of
        a graph batch (they have length 0 and the kernel skips them).

        Both tables have to be in the packed token-major order of the step
        (``_pack_tree_rows_to_tokens``), where ``table.reshape(-1)[p]`` is token
        ``p`` of the step; the caller only builds a plan when that packing ran.
        """
        if init_state_indices is None or self.spec_tree_slots is None:
            return None
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_verify_plan,
        )

        plan = compute_tree_verify_plan(
            row_widths,
            state_indices,
            init_state_indices,
            scratch_base,
        )
        if plan is None:
            return None
        if self.use_full_cuda_graph:
            if self.spec_verify_plan_buffers is None:
                raise RuntimeError(
                    "a full CUDA graph replays the recorded verify chunks by address, "
                    "so a draft tree wider than one recurrent-op call needs the stable "
                    "chunk buffers of _allocate_tree_verify_plan_buffers."
                )
            plan = plan.copy_into(self.spec_verify_plan_buffers)
        return plan

    def _build_tree_init_state_indices(
        self,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
        query_lens_cpu: torch.Tensor,
        spec_sequence_masks_cpu: torch.Tensor,
        tree_parents: torch.Tensor | None,
        tree_num_nodes: torch.Tensor | None,
        spec_sequence_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Per-token state page each draft-tree node starts from.

        The result keeps the ``[rows, slots]`` shape but is packed token-major,
        because the recurrent kernel indexes the table by global token offset.
        Returns ``None`` for the linear MTP chain, where the recurrent kernel
        carries the state of the previous token in registers.
        """
        if self.spec_tree_slots is None or state_indices is None:
            return None
        if tree_parents is None or tree_num_nodes is None:
            raise RuntimeError(
                "draft-tree GDN state rollback needs the tree layout; "
                "tree_parents/tree_num_nodes were not passed to build()."
            )
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            selected_gdn_state_update_name,
        )

        if selected_gdn_state_update_name() == "treewy":
            # TreeWY starts every node from the committed root state and solves
            # the intra-tree recurrence with pseudo-values instead of parent
            # state pages.
            spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
            if bool(((spec_query_lens_cpu < 1) | (spec_query_lens_cpu > self.spec_tree_slots)).any()):
                raise ValueError(
                    "TreeWY GDN needs 1 <= verified tokens <= "
                    f"1 + budget = {self.spec_tree_slots} per request."
                )
            return None
        slots = self.spec_tree_slots
        # A row may verify fewer than ``1 + budget`` tokens: the scheduler
        # truncates the draft of a request near ``max_model_len`` and the tree
        # search can run out of candidates. The tables are packed token-major
        # below, so only the per-row bounds matter here.
        spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
        if bool(((spec_query_lens_cpu < 1) | (spec_query_lens_cpu > slots)).any()):
            raise ValueError(
                "draft-tree GDN state rollback needs 1 <= verified tokens <= "
                f"1 + budget = {slots} per speculative request, got "
                f"{spec_query_lens_cpu.tolist()}."
            )
        # Lazy import: the tree spec-decode package imports this module.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_init_state_indices,
        )

        init_state_indices = compute_tree_init_state_indices(
            state_indices,
            num_accepted_tokens,
            torch.index_select(tree_parents, 0, spec_sequence_indices),
            torch.index_select(tree_num_nodes, 0, spec_sequence_indices),
        )
        return _pack_tree_rows_to_tokens(init_state_indices, spec_query_lens_cpu, slots)

    def _select_spec_state_indices(
        self,
        block_table_tensor: torch.Tensor,
        spec_sequence_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select node pages, repeating the committed page for TreeWY rows."""
        if self.spec_tree_slots is None:
            return torch.index_select(
                block_table_tensor[:, : self.num_spec + 1],
                0,
                spec_sequence_indices,
            )

        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            selected_gdn_state_update_name,
        )

        if selected_gdn_state_update_name() == "treewy":
            root_pages = torch.index_select(block_table_tensor[:, :1], 0, spec_sequence_indices)
            # The same page owns recurrent state and conv state for each request;
            # only the conv state's internal columns vary by node.
            return root_pages.expand(-1, self.spec_tree_slots)
        return torch.index_select(
            block_table_tensor[:, : self.spec_tree_slots],
            0,
            spec_sequence_indices,
        )

    def _select_tree_layout(
        self,
        tree_parents: torch.Tensor | None,
        tree_num_nodes: torch.Tensor | None,
        spec_sequence_indices: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Gather tree topology into builder-owned buffers for graph stability."""
        if self.spec_tree_slots is None:
            return None, None
        if tree_parents is None or tree_num_nodes is None:
            raise RuntimeError(
                "draft-tree GDN metadata needs tree_parents and tree_num_nodes."
            )
        if self.spec_tree_parents is None or self.spec_tree_num_nodes is None:
            raise RuntimeError("draft-tree topology buffers were not allocated.")
        num_rows = spec_sequence_indices.numel()
        selected_parents = torch.index_select(tree_parents, 0, spec_sequence_indices)
        selected_num_nodes = torch.index_select(tree_num_nodes, 0, spec_sequence_indices)
        if selected_parents.shape[1] < self.spec_tree_slots - 1:
            raise ValueError(
                f"tree layout carries {selected_parents.shape[1]} parents for "
                f"{self.spec_tree_slots} verify slots."
            )
        self.spec_tree_parents[:num_rows].copy_(selected_parents[:, : self.spec_tree_slots - 1])
        self.spec_tree_num_nodes[:num_rows].copy_(selected_num_nodes)
        if self.spec_tree_ancestors is not None:
            # Topology changes every draft. Refresh a stable metadata buffer
            # once per batch, outside each GDN layer and its captured graph.
            self.spec_tree_ancestors[:num_rows].copy_(
                treewy_ancestor_mask(self.spec_tree_parents[:num_rows], self.spec_tree_num_nodes[:num_rows])
            )
        return self.spec_tree_parents[:num_rows], self.spec_tree_num_nodes[:num_rows]

    def _build_tree_conv_rows(
        self,
        num_accepted_tokens: torch.Tensor,
        tree_parents: torch.Tensor | None,
        tree_num_nodes: torch.Tensor | None,
        spec_sequence_indices: torch.Tensor,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
        row_widths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Conv tap table of every draft-tree node and the committed history.

        ``window_rows`` indexes ``cat([committed history, this step's
        activations])`` and ``prefix_src`` names the conv-state columns of the
        history the next step resumes from, both consumed by
        ``ops/gdn.tree_causal_conv1d``. ``(None, None)`` keeps the
        sliding-window custom operator of the linear MTP chain.
        """
        if self.spec_tree_slots is None or self.spec_conv_window_rows is None:
            return None, None
        if tree_parents is None or tree_num_nodes is None:
            raise RuntimeError(
                "draft-tree GDN state rollback needs the tree layout; "
                "tree_parents/tree_num_nodes were not passed to build()."
            )
        if prev_path_node_ids is None or prev_num_sampled is None:
            raise RuntimeError(
                "draft-tree GDN short conv needs the accepted path of the "
                "previous step; it was not passed to build()."
            )
        rows = num_accepted_tokens.shape[0]
        assert self.spec_conv_width is not None
        # Lazy import: the tree spec-decode package imports this module.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_conv_rows,
            compute_tree_prefix_source,
        )

        window_rows = compute_tree_conv_rows(
            torch.index_select(tree_parents, 0, spec_sequence_indices),
            torch.index_select(tree_num_nodes, 0, spec_sequence_indices),
            self.spec_conv_width,
            # The committed history of every request comes first, then this
            # step's activations, which are laid out request-major over the
            # tokens each request actually verifies.
            rows * (self.spec_conv_width - 1),
            row_widths=row_widths,
        )
        prefix_src = compute_tree_prefix_source(
            torch.index_select(prev_path_node_ids, 0, spec_sequence_indices),
            torch.index_select(prev_num_sampled, 0, spec_sequence_indices),
            self.spec_conv_width,
        )
        return window_rows, prefix_src

    def _copy_tree_state_table_inputs(
        self,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_sequence_indices: torch.Tensor,
        prev_path_node_ids: torch.Tensor,
        prev_num_sampled: torch.Tensor,
    ) -> int:
        """Refresh stable inputs used by the fixed-width state-table graph."""
        rows = spec_sequence_indices.numel()
        if (
            self.spec_tree_table_state_pages is None
            or self.spec_tree_table_resume_slots is None
            or self.spec_tree_table_prev_path is None
            or self.spec_tree_table_prev_num_sampled is None
        ):
            raise RuntimeError("snapshot tree state-table graph buffers were not allocated.")

        self.spec_tree_table_state_pages[:rows].copy_(state_indices)
        self.spec_tree_table_resume_slots[:rows].copy_(num_accepted_tokens)
        if prev_path_node_ids.shape[0] == rows:
            # The captured path is only used when every row in the fixed-width
            # batch is speculative, so the sequence indices are 0..rows-1.
            self.spec_tree_table_prev_path[:rows].copy_(prev_path_node_ids)
            self.spec_tree_table_prev_num_sampled[:rows].copy_(prev_num_sampled)
        else:
            self.spec_tree_table_prev_path[:rows].copy_(
                torch.index_select(prev_path_node_ids, 0, spec_sequence_indices)
            )
            self.spec_tree_table_prev_num_sampled[:rows].copy_(
                torch.index_select(prev_num_sampled, 0, spec_sequence_indices)
            )
        return rows

    def _compute_snapshot_tree_state_tables(self, rows: int) -> None:
        """Write fixed-width snapshot state tables into their stable outputs."""
        if (
            self.spec_tree_slots is None
            or self.spec_conv_width is None
            or self.spec_init_state_indices is None
            or self.spec_conv_window_rows is None
            or self.spec_conv_prefix_src is None
            or self.spec_tree_parents is None
            or self.spec_tree_num_nodes is None
            or self.spec_tree_table_state_pages is None
            or self.spec_tree_table_resume_slots is None
            or self.spec_tree_table_prev_path is None
            or self.spec_tree_table_prev_num_sampled is None
        ):
            raise RuntimeError("snapshot tree state-table graph is missing a required buffer.")

        # Lazy import: the tree spec-decode package imports this builder.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_conv_rows,
            compute_tree_init_state_indices,
            compute_tree_prefix_source,
        )

        slots = self.spec_tree_slots
        init_state_indices = compute_tree_init_state_indices(
            self.spec_tree_table_state_pages[:rows],
            self.spec_tree_table_resume_slots[:rows],
            self.spec_tree_parents[:rows],
            self.spec_tree_num_nodes[:rows],
        )
        self.spec_init_state_indices[:rows].copy_(init_state_indices)

        conv_window_rows = compute_tree_conv_rows(
            self.spec_tree_parents[:rows],
            self.spec_tree_num_nodes[:rows],
            self.spec_conv_width,
            rows * (self.spec_conv_width - 1),
        )
        self.spec_conv_window_rows[: rows * slots].copy_(conv_window_rows)

        conv_prefix_src = compute_tree_prefix_source(
            self.spec_tree_table_prev_path[:rows],
            self.spec_tree_table_prev_num_sampled[:rows],
            self.spec_conv_width,
        )
        self.spec_conv_prefix_src[:rows].copy_(conv_prefix_src)

    def _get_or_capture_snapshot_tree_state_table_graph(
        self,
        rows: int,
        *,
        capture: bool,
    ) -> Any | None:
        """Find a captured fixed-width table graph or capture it during warmup."""
        graph = self._tree_state_table_graphs.get(rows)
        if graph is not None:
            return graph
        if (
            not capture
            or rows in self._tree_state_table_graph_failures
            or self.spec_tree_slots is None
            or self.spec_conv_width is None
            or not hasattr(torch, "npu")
            or not hasattr(torch.npu, "NPUGraph")
            or not hasattr(torch.npu, "is_current_stream_capturing")
            or self.spec_tree_parents is None
            or self.spec_tree_parents.device.type != "npu"
        ):
            return None

        # This helper is called while outer model graphs are being prepared.
        # Never attempt nested NPUGraph capture if a backend calls it from an
        # active capture stream.
        if torch.npu.is_current_stream_capturing():
            return None

        try:
            from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
            from vllm.platforms import current_platform

            validate_cudagraph_capturing_enabled()
            graph_pool = current_platform.get_global_graph_pool()
            with torch.inference_mode():
                self._compute_snapshot_tree_state_tables(rows)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=graph_pool):
                    self._compute_snapshot_tree_state_tables(rows)
            self._tree_state_table_graphs[rows] = graph
            logger.info("Captured GDN snapshot state-table ACLGraph rows=%s", rows)
            return graph
        except Exception as exc:
            self._tree_state_table_graph_failures.add(rows)
            logger.warning(
                "GDN snapshot state-table ACLGraph capture failed for rows=%s; "
                "using eager table construction. %s",
                rows,
                exc,
            )
            return None

    def _run_snapshot_tree_state_table_graph(
        self,
        state_indices: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_sequence_indices: torch.Tensor,
        prev_path_node_ids: torch.Tensor,
        prev_num_sampled: torch.Tensor,
        *,
        capture: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Refresh graph inputs and replay the fixed-width snapshot table graph."""
        rows = spec_sequence_indices.numel()
        # Uncaptured sizes and capture failures must not pay extra device
        # copies on top of the eager fallback at every decode step.
        if rows not in self._tree_state_table_graphs and (
            not capture or rows in self._tree_state_table_graph_failures
        ):
            return None
        if (
            self.spec_tree_slots is None
            or self.spec_init_state_indices is None
            or self.spec_conv_window_rows is None
            or self.spec_conv_prefix_src is None
            or self.spec_tree_parents is None
            or self.spec_tree_num_nodes is None
        ):
            return None
        self._copy_tree_state_table_inputs(
            state_indices,
            num_accepted_tokens,
            spec_sequence_indices,
            prev_path_node_ids,
            prev_num_sampled,
        )
        graph = self._get_or_capture_snapshot_tree_state_table_graph(rows, capture=capture)
        if graph is None:
            return None
        graph.replay()
        slots = self.spec_tree_slots
        return (
            self.spec_init_state_indices[:rows],
            self.spec_conv_window_rows[: rows * slots],
            self.spec_conv_prefix_src[:rows],
        )

    def _count_tree_shape_stats(
        self,
        row_widths: torch.Tensor,
        spec_sequence_masks_cpu: torch.Tensor,
        spec_sequence_indices: torch.Tensor,
        tree_num_nodes: torch.Tensor | None,
        tree_parents: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor,
        init_state_indices: torch.Tensor | None,
        state_indices: torch.Tensor,
    ) -> None:
        """Record what the draft tree looked like for the exit report.

        Only active with ``tree_spec_config.enable_timer``, because the counters
        synchronize the device. They answer two questions that the acceptance
        rate cannot: did this step verify a tree at all (``tree.branch_rows``),
        and did the layout still match the tokens the scheduler put in the batch
        (``tree.unverified_extra_nodes``)?  The speculator proposes ``budget``
        nodes per request, while the scheduler can hand over fewer draft tokens
        near ``max_model_len``; the extra nodes then have no verified token, and
        a tree walk that still visits them accepts tokens the target never
        checked.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.timer import (
            tree_stat,
            tree_timer_enabled,
        )

        if not tree_timer_enabled() or tree_num_nodes is None or init_state_indices is None:
            return
        assert tree_parents is not None
        assert self.spec_tree_slots is not None
        tree_nodes = torch.index_select(tree_num_nodes, 0, spec_sequence_indices).to(torch.long).cpu()
        parents = torch.index_select(tree_parents, 0, spec_sequence_indices).to(torch.long).cpu()
        scheduled_drafts = num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].to(torch.long)
        # A row that follows the chain layout (``node t`` continues ``t - 1``)
        # never loads another node's state, so it cannot exercise the tree state
        # path at all.
        chain = torch.arange(parents.shape[1], dtype=torch.long)
        branch_rows = ((parents != chain).any(dim=1) & (tree_nodes > 0)).sum()
        extra_nodes = (tree_nodes - scheduled_drafts).clamp(min=0)
        tree_stat("tree.rows", row_widths.numel())
        tree_stat("tree.width", int(row_widths.sum().item()))
        tree_stat("tree.short_rows", int((row_widths < self.spec_tree_slots).sum().item()))
        tree_stat("tree.branch_rows", int(branch_rows.item()))
        tree_stat("tree.nodes", int(tree_nodes.sum().item()))
        tree_stat("tree.drafts", int(scheduled_drafts.sum().item()))
        tree_stat("tree.unverified_extra_nodes", int(extra_nodes.sum().item()))
        tree_stat(
            "tree.init_ne_own_nodes",
            int((init_state_indices[:, 1:] != state_indices[:, 1:]).sum().item()),
        )

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        super()._init_reorder_batch_threshold(
            reorder_batch_threshold,
            supports_spec_as_decode,
            True,
        )
        if self.reorder_batch_threshold != 1:  # type: ignore
            speculative_config = self.vllm_config.speculative_config
            method = getattr(speculative_config, "method", None)
            num_spec = getattr(speculative_config, "num_speculative_tokens", None)
            if num_spec is not None and method in ("dflash", "dspark"):
                # The target-model verification query always contains the
                # base token plus N speculative tokens. DSpark's
                # sample_from_anchor only makes the draft-model forward use N
                # queries; it must not change target batch reordering.
                self.reorder_batch_threshold = 1 + num_spec

    def _copy_sequence_indices_to_device(
        self,
        spec_sequence_masks_cpu: torch.Tensor,
        num_spec_decodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = spec_sequence_masks_cpu.numel()
        num_non_spec_decodes = num_reqs - num_spec_decodes

        spec_indices_cpu = self.spec_sequence_indices_cpu[:num_spec_decodes]
        spec_indices_cpu.copy_(
            torch.nonzero(spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        spec_indices = self.spec_sequence_indices[:num_spec_decodes]
        spec_indices.copy_(spec_indices_cpu, non_blocking=True)

        non_spec_indices_cpu = self.non_spec_sequence_indices_cpu[:num_non_spec_decodes]
        non_spec_indices_cpu.copy_(
            torch.nonzero(~spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        non_spec_indices = self.non_spec_sequence_indices[:num_non_spec_decodes]
        non_spec_indices.copy_(non_spec_indices_cpu, non_blocking=True)

        return spec_indices, non_spec_indices

    def _pad_non_spec_decode_graph_inputs(
        self,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        num_decode_tokens: int,
        graph_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh fixed buffers consumed by a non-spec decode graph."""
        assert num_decode_tokens <= graph_batch_size

        padded_state_indices = self.non_spec_state_indices_tensor[:graph_batch_size]
        padded_state_indices[num_decode_tokens:].fill_(NULL_BLOCK_ID)
        padded_state_indices[:num_decode_tokens].copy_(
            state_indices[:num_decode_tokens],
            non_blocking=True,
        )

        padded_query_start_loc = self.non_spec_query_start_loc[: graph_batch_size + 1]
        padded_query_start_loc[: num_decode_tokens + 1].copy_(
            query_start_loc[: num_decode_tokens + 1],
            non_blocking=True,
        )
        query_padding = padded_query_start_loc[num_decode_tokens + 1 :]
        if query_padding.numel() > 0:
            query_padding.copy_(
                padded_query_start_loc[num_decode_tokens].expand_as(query_padding),
                non_blocking=True,
            )

        return padded_state_indices, padded_query_start_loc

    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        """Make a captured speculative branch a no-op for this replay."""
        self.spec_state_indices_tensor[:graph_batch_size].fill_(PAD_SLOT_ID)
        self.spec_query_start_loc[: graph_batch_size + 1].zero_()
        self.num_accepted_tokens[:graph_batch_size].zero_()
        self.spec_actual_seq_lengths[: graph_batch_size + 1].zero_()
        if self.spec_init_state_indices is not None:
            self.spec_init_state_indices[:graph_batch_size].fill_(NULL_BLOCK_ID)
        if self.spec_verify_plan_buffers is not None:
            # Zero the lengths of every recorded verify chunk. The stale gather
            # indices of the previous step stay in bounds and the op skips each
            # zero-length row, so the captured chunks become no-op calls.
            self.spec_verify_plan_buffers.zero_lengths()

    def _attach_non_spec_prefill_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        chunk_metadata: GDNChunkedPrefillMetadata | None,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_prefill_metadata = None
        if attn_metadata.num_prefills <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.non_spec_query_start_loc for Ascend GDN non-spec prefill path.")
        if attn_metadata.prefill_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.prefill_query_start_loc for Ascend GDN non-spec prefill path.")
        if chunk_metadata is None:
            raise RuntimeError("Expected chunk metadata for Ascend GDN non-spec prefill path.")

        initial_state_mode = attn_metadata.has_initial_state
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN prefill conv1d path.")
        prefill_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        pcp_size = getattr(self.vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        if pcp_rank > 0 and attn_metadata.num_prefills > 0:
            prefill_seq_offset = max(0, prefill_num_rows - attn_metadata.num_prefills)
            initial_state_mode = initial_state_mode.clone()
            initial_state_mode[prefill_seq_offset:] = True
        attn_metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:prefill_num_rows],
                initial_state_mode=initial_state_mode,
            ),
            chunk=chunk_metadata,
        )
        return attn_metadata

    def _attach_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        init_state_indices: torch.Tensor | None = None,
        conv_window_rows: torch.Tensor | None = None,
        conv_prefix_src: torch.Tensor | None = None,
        tree_parents: torch.Tensor | None = None,
        tree_num_nodes: torch.Tensor | None = None,
        tree_sequence_indices: torch.Tensor | None = None,
    ) -> GDNAttentionMetadata:
        attn_metadata.spec_decode_metadata = None
        if attn_metadata.spec_sequence_masks is None:
            return attn_metadata

        if attn_metadata.spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.spec_query_start_loc for Ascend GDN speculative path.")
        if attn_metadata.spec_state_indices_tensor is None:
            raise RuntimeError("Expected spec_state_indices_tensor for Ascend GDN speculative conv1d path.")
        if attn_metadata.num_accepted_tokens is None:
            raise RuntimeError("Expected num_accepted_tokens for Ascend GDN speculative conv1d path.")

        num_sequences = attn_metadata.num_spec_decodes
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            num_sequences = attn_metadata.spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.spec_actual_seq_lengths
        spec_num_rows = attn_metadata.spec_query_start_loc.size(0) - 1
        actual_seq_lengths = _build_actual_seq_lengths(
            attn_metadata.spec_query_start_loc,
            num_sequences,
            actual_seq_lengths_buffer,
        )
        # A wider-than-one-call tree is verified in chunks; the rows are the op's
        # rows, i.e. the padded graph batch under a full graph (a padded row has
        # query length 0 and the kernel skips it). The plan indexes the state
        # tables by packed token offset, so it is only built when the tree packing
        # ran -- ``build`` gates that on the tree conv rows.
        verify_plan = None
        state_update_method = None
        recurrent_state_slots = None
        conv_slots = None
        if conv_window_rows is not None:
            verify_plan = self._build_tree_verify_plan(
                actual_seq_lengths[1:],
                attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                None if init_state_indices is None else init_state_indices[:spec_num_rows],
                attn_metadata.num_actual_tokens,
            )
            from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
                selected_gdn_state_update_name,
                tree_conv_slot_count,
                tree_recurrent_state_slot_count,
            )

            state_update_method = selected_gdn_state_update_name()
            recurrent_state_slots = tree_recurrent_state_slot_count(self.vllm_config)
            conv_slots = tree_conv_slot_count(self.vllm_config)

        attn_metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
            spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
                query_start_loc=attn_metadata.spec_query_start_loc,
                cache_indices=attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                num_accepted_tokens=attn_metadata.num_accepted_tokens[:spec_num_rows],
            ),
            actual_seq_lengths=actual_seq_lengths,
            init_state_indices=init_state_indices,
            conv_window_rows=conv_window_rows,
            conv_prefix_src=conv_prefix_src,
            verify_plan=verify_plan,
            gdn_state_update_method=state_update_method,
            tree_verify_slots=self.spec_tree_slots if conv_window_rows is not None else None,
            tree_recurrent_state_slots=recurrent_state_slots,
            tree_conv_slots=conv_slots,
            tree_parents=tree_parents,
            tree_num_nodes=tree_num_nodes,
            tree_ancestors=(
                self.spec_tree_ancestors[:tree_parents.shape[0]]
                if state_update_method == "treewy"
                and tree_parents is not None
                and self.spec_tree_ancestors is not None
                else None
            ),
            tree_row_widths=actual_seq_lengths[1:] if conv_window_rows is not None else None,
            tree_sequence_indices=tree_sequence_indices,
            treewy_contexts={} if state_update_method == "treewy" else None,
        )
        return attn_metadata

    def _attach_non_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_decode_metadata = None
        if attn_metadata.num_decodes <= 0 and attn_metadata.num_prefills <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected non-spec query_start_loc for Ascend GDN non-spec decode path.")
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN decode conv1d path.")

        num_sequences = attn_metadata.num_decodes
        non_spec_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_spec_decodes == 0:
            num_sequences = attn_metadata.non_spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.non_spec_actual_seq_lengths

        attn_metadata.non_spec_decode_metadata = GDNDecodeMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:non_spec_num_rows],
                initial_state_mode=None,
            ),
            actual_seq_lengths=_build_actual_seq_lengths(
                attn_metadata.non_spec_query_start_loc,
                num_sequences,
                actual_seq_lengths_buffer,
            ),
        )
        return attn_metadata

    def _fold_spec_sized_prefill_chunks_into_spec(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_sequence_masks_cpu: torch.Tensor,
        num_accepted_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance stateful spec-width prompt chunks through live spec inputs."""
        if self.spec_tree_slots is not None and selected_gdn_state_update_name() == "treewy":
            # TreeWY defers state writes until speculative acceptance. A prompt
            # chunk must commit every input token even when its sampler emits
            # only one token (final chunk), or no tree path exists at all.
            # Keep it on the native prefill path, which commits during forward.
            return spec_sequence_masks_cpu, num_accepted_tokens
        is_prefilling = common_attn_metadata.is_prefilling
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if is_prefilling is None or seq_lens_cpu is None or num_accepted_tokens is None:
            return spec_sequence_masks_cpu, num_accepted_tokens

        num_reqs = min(
            spec_sequence_masks_cpu.numel(),
            is_prefilling.numel(),
            seq_lens_cpu.numel(),
        )
        is_prefilling = is_prefilling[:num_reqs]
        seq_lens_cpu = seq_lens_cpu[:num_reqs]
        query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)[:num_reqs]
        fold = (
            is_prefilling
            & ~spec_sequence_masks_cpu
            & (query_lens_cpu == self.num_spec + 1)
            & (seq_lens_cpu > query_lens_cpu)
        )
        fold_indices = fold.nonzero(as_tuple=True)[0]
        if fold_indices.numel() == 0:
            return spec_sequence_masks_cpu, num_accepted_tokens

        spec_sequence_masks_cpu = spec_sequence_masks_cpu.clone()
        spec_sequence_masks_cpu[fold_indices] = True
        num_accepted_tokens = num_accepted_tokens.clone()
        num_accepted_tokens[fold_indices.to(num_accepted_tokens.device)] = self.num_spec + 1
        return spec_sequence_masks_cpu, num_accepted_tokens

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        tree_parents: torch.Tensor | None = None,
        tree_num_nodes: torch.Tensor | None = None,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
        _capture_tree_state_table_graph: bool = False,
    ) -> GDNAttentionMetadata:
        m = _treat_single_token_prefills_with_state_as_decodes(common_attn_metadata)

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_sequence_indices: torch.Tensor | None = None
        non_spec_sequence_indices: torch.Tensor | None = None
        spec_tree_parents: torch.Tensor | None = None
        spec_tree_num_nodes: torch.Tensor | None = None
        spec_tree_sequence_indices: torch.Tensor | None = None
        non_spec_conv1d_cache_indices: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_reqs = num_decode_draft_tokens_cpu.numel()
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_reqs]
            runtime_draft_tokens = num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            if runtime_draft_tokens.sum().item() > 0:
                torch.ge(
                    num_decode_draft_tokens_cpu,
                    0,
                    out=spec_sequence_masks_cpu,
                )
            else:
                # Dynamic speculative decoding can be enabled while this batch
                # carries no draft tokens. Treat it as ordinary decode unless a
                # stateful spec-width prompt chunk must use the spec branch.
                spec_sequence_masks_cpu.zero_()
            # DCP must retain prefill metadata for prompt chunks, even when
            # their width matches speculative decode. Keep the legacy fold
            # when decode context parallelism is disabled.
            if self.vllm_config.parallel_config.decode_context_parallel_size == 1:
                spec_sequence_masks_cpu, num_accepted_tokens = self._fold_spec_sized_prefill_chunks_into_spec(
                    m,
                    spec_sequence_masks_cpu,
                    num_accepted_tokens,
                )
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.spec_sequence_masks[:num_reqs]
                spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)
                spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                    spec_sequence_masks_cpu,
                    num_spec_decodes,
                )
                if self.spec_tree_slots is not None:
                    spec_tree_parents, spec_tree_num_nodes = _tree_profiled_call(
                        "gdn_tree_layout_select",
                        self._select_tree_layout,
                        tree_parents,
                        tree_num_nodes,
                        spec_sequence_indices,
                    )
                    spec_tree_sequence_indices = spec_sequence_indices

        spec_init_state_indices: torch.Tensor | None = None
        spec_conv_window_rows: torch.Tensor | None = None
        spec_conv_prefix_src: torch.Tensor | None = None
        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
                m,
                decode_threshold=1,
                treat_short_extends_as_decodes=False,
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            non_spec_conv1d_cache_indices = block_table_tensor
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            assert spec_sequence_masks_cpu is not None
            assert spec_sequence_indices is not None
            assert non_spec_sequence_indices is not None

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            num_spec_decode_tokens = query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens

            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                spec_state_indices_tensor = self._select_spec_state_indices(
                    block_table_tensor,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = _stable_argsort_for_npu(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = self._select_spec_state_indices(
                    block_table_tensor,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, 0],
                    0,
                    non_spec_sequence_indices,
                )
                non_spec_conv1d_cache_indices = non_spec_state_indices_tensor
                spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    spec_sequence_indices,
                )
                non_spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    non_spec_sequence_indices,
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    spec_query_lens,
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    non_spec_query_lens,
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = torch.index_select(
                num_accepted_tokens,
                0,
                spec_sequence_indices,
            )
            # Tokens each speculative request verifies; a row may be shorter
            # than the tree width, see _build_tree_init_state_indices.
            spec_row_widths = query_lens_cpu[spec_sequence_masks_cpu]
            full_width_snapshot_batch = (
                self.use_full_cuda_graph
                and self.spec_tree_slots is not None
                and self.spec_conv_width is not None
                and selected_gdn_state_update_name() == "snapshot"
                and num_prefills == 0
                and num_decodes == 0
                and num_spec_decodes == m.num_reqs
                and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
                and spec_row_widths.numel() == num_spec_decodes
                and bool(torch.all(spec_row_widths == self.spec_tree_slots))
            )
            state_table_graph_outputs = None
            if (
                full_width_snapshot_batch
                and (_capture_tree_state_table_graph or num_spec_decodes in self._tree_state_table_graphs)
                and spec_state_indices_tensor is not None
                and num_accepted_tokens is not None
                and spec_sequence_indices is not None
                and prev_path_node_ids is not None
                and prev_num_sampled is not None
            ):
                state_table_graph_outputs = _tree_profiled_call(
                    "gdn_state_tables_graph",
                    self._run_snapshot_tree_state_table_graph,
                    spec_state_indices_tensor,
                    num_accepted_tokens,
                    spec_sequence_indices,
                    prev_path_node_ids,
                    prev_num_sampled,
                    capture=_capture_tree_state_table_graph,
                )
            if state_table_graph_outputs is not None:
                (
                    spec_init_state_indices,
                    spec_conv_window_rows,
                    spec_conv_prefix_src,
                ) = state_table_graph_outputs
            else:
                spec_init_state_indices = _tree_profiled_call(
                    "gdn_recurrent_state_table",
                    self._build_tree_init_state_indices,
                    spec_state_indices_tensor,
                    num_accepted_tokens,
                    query_lens_cpu,
                    spec_sequence_masks_cpu,
                    tree_parents,
                    tree_num_nodes,
                    spec_sequence_indices,
                )
                spec_conv_window_rows, spec_conv_prefix_src = _tree_profiled_call(
                    "gdn_conv_state_tables",
                    self._build_tree_conv_rows,
                    num_accepted_tokens,
                    tree_parents,
                    tree_num_nodes,
                    spec_sequence_indices,
                    prev_path_node_ids,
                    prev_num_sampled,
                    row_widths=spec_row_widths,
                )
            if spec_conv_window_rows is not None and self.spec_conv_window_rows is not None:
                # Same packing for the state pages: the recurrent op consumes
                # them by global token offset, like the init pages and the conv
                # windows. The conv path keeps reading the table per row, taking
                # each request's running page at its token start
                # (ops/gdn.tree_causal_conv1d).
                if not full_width_snapshot_batch:
                    spec_state_indices_tensor = _tree_profiled_call(
                        "gdn_state_page_pack",
                        _pack_tree_rows_to_tokens,
                        spec_state_indices_tensor,
                        spec_row_widths,
                        spec_state_indices_tensor.shape[1],
                    )
                _tree_profiled_call(
                    "gdn_tree_shape_stats",
                    self._count_tree_shape_stats,
                    spec_row_widths,
                    spec_sequence_masks_cpu,
                    spec_sequence_indices,
                    tree_num_nodes,
                    tree_parents,
                    num_decode_draft_tokens_cpu,
                    spec_init_state_indices,
                    spec_state_indices_tensor,
                )

        # A FULL graph retains captured speculative conv/recurrent tasks. Clear
        # their stable inputs on every no-spec replay so an idle or prefill
        # batch cannot mutate state belonging to the preceding request.
        if self.use_full_cuda_graph and self.use_spec_decode and num_spec_decodes == 0:
            self._reset_spec_decode_graph_inputs(m.num_reqs)

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_query_start_loc_cpu: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        non_spec_chunked_prefill_metadata: GDNChunkedPrefillMetadata | None = None
        if num_prefills > 0:
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            assert prefill_query_start_loc_cpu is not None
            non_spec_chunked_prefill_metadata = _build_non_spec_chunked_prefill_metadata(
                self,
                prefill_query_start_loc_cpu,
                query_start_loc.device,
            )
            # Preserve upstream GDNAttentionMetadata fields for callers that
            # still use the chunk_gated_delta_rule API directly.
            chunk_indices = non_spec_chunked_prefill_metadata.chunk_indices_chunk64
            chunk_offsets = non_spec_chunked_prefill_metadata.chunk_offsets_chunk64

        if num_prefills > 0:
            (
                has_initial_state,
                nums_dict,
                batch_ptr,
                token_chunk_offset_ptr,
            ) = self._build_prefill_has_initial_state_and_causal_conv1d_meta(
                common_attn_metadata=m,
                context_lens_tensor=context_lens_tensor,
                num_prefills=num_prefills,
                spec_sequence_masks_cpu=spec_sequence_masks_cpu,
                non_spec_sequence_indices=non_spec_sequence_indices,
                non_spec_query_start_loc_cpu=non_spec_query_start_loc_cpu,
                query_start_loc=query_start_loc,
            )
            assert has_initial_state is not None
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            # Spec decode has multiple tokens per request. Keep the metadata
            # passed to conv1d/recurrent kernels at request granularity; padding
            # it to the token count makes the conv1d update kernel treat every
            # token as an independent decode sequence.
            spec_batch_size = m.num_reqs

            self.spec_state_indices_tensor[spec_batch_size:].fill_(NULL_BLOCK_ID)
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor,
                non_blocking=True,
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:spec_batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes],
                non_blocking=True,
            )
            spec_sequence_masks = self.spec_sequence_masks[:spec_batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx,
                non_blocking=True,
            )
            non_spec_token_indx = self.non_spec_token_indx[: non_spec_token_indx.size(0)]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx,
                non_blocking=True,
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc,
                non_blocking=True,
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore
            spec_query_start_loc = self.spec_query_start_loc[: spec_batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens,
                non_blocking=True,
            )
            num_accepted_tokens = self.num_accepted_tokens[:spec_batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

            if spec_init_state_indices is not None:
                # Padded rows replay with a zero query length, so their page ids
                # are never dereferenced; keep them consistent with the state
                # table padding.
                assert self.spec_init_state_indices is not None
                if self.spec_init_state_indices.data_ptr() != spec_init_state_indices.data_ptr():
                    self.spec_init_state_indices[:num_spec_decodes].copy_(
                        spec_init_state_indices,
                        non_blocking=True,
                    )
                spec_init_state_indices = self.spec_init_state_indices[:spec_batch_size]
                spec_init_state_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            if spec_conv_window_rows is not None:
                # The captured graph reads the tables by address, so they have
                # to live in stable buffers; rows of padded requests are never
                # gathered (the layer slices the window table by the real token
                # count and skips the rows that own no state page).
                assert spec_conv_prefix_src is not None
                assert self.spec_conv_window_rows is not None and self.spec_conv_prefix_src is not None
                if self.spec_conv_window_rows.data_ptr() != spec_conv_window_rows.data_ptr():
                    self.spec_conv_window_rows[: spec_conv_window_rows.size(0)].copy_(
                        spec_conv_window_rows,
                        non_blocking=True,
                    )
                if self.spec_conv_prefix_src.data_ptr() != spec_conv_prefix_src.data_ptr():
                    self.spec_conv_prefix_src[: spec_conv_prefix_src.size(0)].copy_(
                        spec_conv_prefix_src,
                        non_blocking=True,
                    )
                spec_conv_window_rows = self.spec_conv_window_rows[: spec_batch_size * self.spec_tree_slots]
                spec_conv_prefix_src = self.spec_conv_prefix_src[:spec_batch_size]

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            graph_batch_size = m.num_reqs
            (
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
            ) = self._pad_non_spec_decode_graph_inputs(
                non_spec_state_indices_tensor,
                non_spec_query_start_loc,
                num_decode_tokens=num_decode_tokens,
                graph_batch_size=graph_batch_size,
            )
            non_spec_conv1d_cache_indices = non_spec_state_indices_tensor

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        attn_metadata = self._attach_non_spec_prefill_metadata(
            attn_metadata,
            non_spec_chunked_prefill_metadata,
            non_spec_conv1d_cache_indices,
        )
        attn_metadata = self._attach_spec_decode_metadata(
            attn_metadata,
            spec_init_state_indices,
            spec_conv_window_rows,
            spec_conv_prefix_src,
            spec_tree_parents,
            spec_tree_num_nodes,
            spec_tree_sequence_indices,
        )
        return self._attach_non_spec_decode_metadata(
            attn_metadata,
            non_spec_conv1d_cache_indices,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> GDNAttentionMetadata:
        """Prepare tree-aware model metadata and capture fixed-width read tables.

        A full graph replays the kernels recorded at capture, so the recurrent op
        has to be recorded *with* its ``init_state_indices`` input; a chain
        layout keeps the recorded kernel on its in-register carry and the
        captured page ids are never used because capture outputs are dropped.
        Metadata preparation happens before model capture. Eligible snapshot
        batches capture a separate table graph here, then replay it during
        build() before the model graph consumes the stable table outputs.
        """
        if self.spec_tree_slots is None:
            return super().build_for_cudagraph_capture(common_attn_metadata)

        m = common_attn_metadata
        device = m.query_start_loc.device
        # tree_num_nodes == 0 selects the linear-chain parents in
        # state_rollback.compute_tree_init_state_indices.
        tree_parents = torch.zeros(
            (m.num_reqs, self.spec_tree_slots - 1),
            dtype=torch.int32,
            device=device,
        )
        tree_num_nodes = torch.zeros((m.num_reqs,), dtype=torch.int32, device=device)
        # A capture batch has no accepted path: ``num_sampled == 0`` keeps the
        # committed history where the chain op left it (identity source). The
        # values only have to be shaped right, every replay rebuilds them.
        prev_path_node_ids = torch.zeros(
            (m.num_reqs, self.spec_tree_slots - 1),
            dtype=torch.int32,
            device=device,
        )
        prev_num_sampled = torch.zeros((m.num_reqs,), dtype=torch.int32, device=device)
        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        return self.build(
            0,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            tree_parents=tree_parents,
            tree_num_nodes=tree_num_nodes,
            prev_path_node_ids=prev_path_node_ids,
            prev_num_sampled=prev_num_sampled,
            _capture_tree_state_table_graph=True,
        )

    def _build_prefill_has_initial_state_and_causal_conv1d_meta(
        self,
        *,
        common_attn_metadata: CommonAttentionMetadata,
        context_lens_tensor: torch.Tensor,
        num_prefills: int,
        spec_sequence_masks_cpu: torch.Tensor | None,
        non_spec_sequence_indices: torch.Tensor | None,
        non_spec_query_start_loc_cpu: torch.Tensor | None,
        query_start_loc: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        dict[int, dict[str, object]] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        del num_prefills
        has_initial_state = context_lens_tensor > 0
        if spec_sequence_masks_cpu is not None:
            assert non_spec_sequence_indices is not None
            has_initial_state = torch.index_select(
                has_initial_state,
                0,
                non_spec_sequence_indices,
            )
            assert non_spec_query_start_loc_cpu is not None
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            non_spec_query_start_loc_cpu,
            device=query_start_loc.device,
        )
        return (
            has_initial_state,
            nums_dict,
            batch_ptr,
            token_chunk_offset_ptr,
        )


class AscendGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        return AscendGDNAttentionMetadataBuilder
