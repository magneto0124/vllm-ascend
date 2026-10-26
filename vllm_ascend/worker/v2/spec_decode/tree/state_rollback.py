"""GDN state rollback for draft-tree speculative decoding.

The target verifies ``1 + budget`` tokens per request (accepted root plus ``budget``
nodes) and acceptance walks one root-to-leaf path, so GDN's one-state-per-page layout
needs an explicit strategy picked by ``tree_spec_config.gdn_state_update``: ``snapshot``
keeps a state per verified node and resumes from the last accepted one; ``treewy``
(:mod:`treewy`) keeps the committed page and rebuilds the accepted state from
pseudo-values.

Verify width and short-conv capacity (``1 + budget``) are what
``attn_utils.get_kv_cache_spec`` widens ``MambaSpec`` to; recurrent capacity is sized
separately (``1 + budget`` for snapshots, one page for TreeWY). Only the Qwen GDN
backend is wired up -- :func:`validate_tree_state_slots` rejects a tree elsewhere
instead of serving stale states -- and TreeWY still runs as PyTorch tensors on Ascend.
"""

from dataclasses import dataclass, replace

import torch

# MAX_MTP in csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h.
# At or above it the kernel returns without touching state (silently stale), so wider
# trees are verified in chunks -- see :func:`compute_tree_verify_plan`.
GDN_RECURRENT_MAX_VERIFY_TOKENS = 16


@dataclass(frozen=True)
class GdnStateUpdateMethod:
    """Capabilities of one ``tree_spec_config.gdn_state_update`` method.

    ``state_init``: a node loads its parent page; ``conv``: the short-conv half follows the tree.
    """

    state_init: bool
    conv: bool
    description: str
    recurrent_state_slots: int | None = None
    treewy_solver: bool = False
    implemented: bool = True


# Read on the GDN path only, so the setting stays inert for every other state backend.
# A new method must define its recurrent layout, verify/commit path and short-conv history.
GDN_STATE_UPDATE_METHODS: dict[str, GdnStateUpdateMethod] = {
    "snapshot": GdnStateUpdateMethod(
        state_init=True,
        conv=True,
        description=(
            "one recurrent state snapshot per tree node: node j loads its parent's "
            "slot and writes its own, and the short-conv page keeps this step's node "
            "activations with the history re-anchored on the node the sampler accepted"
        ),
    ),
    "treewy": GdnStateUpdateMethod(
        state_init=False,
        conv=True,
        description=(
            "one recurrent state page per request, with TreeWY pseudo-values retained "
            "until the accepted path is known"
        ),
        recurrent_state_slots=1,
        treewy_solver=True,
    ),
}
DEFAULT_GDN_STATE_UPDATE = "snapshot"


def selected_gdn_state_update_name() -> str:
    """Name of the draft-tree GDN state update method from ``tree_spec_config``.

    Resolved here, not in ``TreeSpecConfig``, so a non-GDN model never sees an unknown value.
    """
    from vllm_ascend.ascend_config import get_ascend_config

    name = getattr(
        get_ascend_config().tree_spec_config,
        "gdn_state_update",
        DEFAULT_GDN_STATE_UPDATE,
    )
    if name not in GDN_STATE_UPDATE_METHODS:
        raise ValueError(
            "tree_spec_config.gdn_state_update must be one of "
            f"{tuple(GDN_STATE_UPDATE_METHODS)}, got {name!r}"
        )
    return name


def selected_gdn_state_update_method() -> GdnStateUpdateMethod:
    """The draft-tree GDN state update method ``tree_spec_config`` selects."""
    return GDN_STATE_UPDATE_METHODS[selected_gdn_state_update_name()]


def tree_verify_slot_count(vllm_config) -> int | None:
    """Tokens the target verifies per request under a draft tree, or ``None`` when idle.

    ``1 + budget`` for both methods, independent of the recurrent and conv-cache layouts.
    """
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.worker.v2.spec_decode import dflash_tree_spec_enabled

    if not getattr(vllm_config, "use_v2_model_runner", False):
        # Only the v2 runner hosts the tree; a leftover config must not change the linear MTP layout.
        return None
    # Shared gate of the whole tree path (historical name; it also covers the dspark
    # host), which keeps the state width aligned with the runner's verify width.
    if not dflash_tree_spec_enabled(vllm_config):
        return None
    budget = get_ascend_config().tree_spec_config.budget
    if budget is None:
        return None
    if getattr(getattr(vllm_config, "cache_config", None), "mamba_cache_mode", None) == "align":
        # The align pre-copy shifts the short-conv window by num_accepted_tokens - 1; a
        # tree cursor is a node id with its own root-to-node window, so the shifted
        # window would silently corrupt the conv history.
        raise NotImplementedError(
            "tree_spec_config is not supported with mamba_cache_mode='align' "
            "yet; the align pre-copy shift of the short-conv state has no "
            "draft-tree equivalent. Disable mamba_cache_mode='align' or "
            "tree_spec_config."
        )
    return 1 + budget


def tree_conv_slot_count(vllm_config) -> int | None:
    """Number of node activations the short-conv cache retains per request."""
    return tree_verify_slot_count(vllm_config)


def tree_recurrent_state_slot_count(vllm_config) -> int | None:
    """Recurrent-state slots the selected GDN method needs.

    ``snapshot`` stores one page per verified node; TreeWY keeps one committed page.
    Separate from :func:`tree_verify_slot_count`, which stays the query width.
    """
    verify_slots = tree_verify_slot_count(vllm_config)
    if verify_slots is None:
        return None
    method = selected_gdn_state_update_method()
    return method.recurrent_state_slots or verify_slots


def tree_state_slot_count(vllm_config) -> int | None:
    """Backward-compatible name for :func:`tree_recurrent_state_slot_count`."""
    return tree_recurrent_state_slot_count(vllm_config)


def validate_tree_state_slots(slots: int, mamba_type) -> None:
    """Reject draft trees that the current state path cannot serve yet.

    Wider trees are fine: :func:`compute_tree_verify_plan` chunks them through the state pages.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    if slots < 1:
        raise ValueError(f"a draft tree verifies at least one token per request, got slots={slots}.")
    if mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
        raise NotImplementedError(
            f"tree_spec_config is enabled but mamba_type={mamba_type} does not "
            "support draft-tree state rollback yet; only GDN_ATTN (Qwen GDN) is "
            "wired up. Disable tree_spec_config for this model."
        )
    # Only with a GDN backend does gdn_state_update mean anything: unknown or
    # half-carrying methods are refused here, never for a model that does not use GDN.
    method_name = selected_gdn_state_update_name()
    method = GDN_STATE_UPDATE_METHODS[method_name]
    if not method.implemented:
        raise NotImplementedError(
            f"tree_spec_config.gdn_state_update={method_name!r} has a reserved cache "
            "layout, but its verification and accepted-state commit path is not "
            "wired up yet."
        )
    if not method.state_init and not method.treewy_solver:
        raise NotImplementedError(
            "the configured tree_spec_config.gdn_state_update does not carry the "
            "recurrent state across a draft tree yet: the recurrent kernel would "
            "read each node's own slot instead of its parent slot through "
            "init_state_indices. Disable tree_spec_config or pick another "
            "gdn_state_update."
        )
    if not method.conv:
        raise NotImplementedError(
            "the configured tree_spec_config.gdn_state_update does not carry the "
            "short-conv half across a draft tree yet: causal_conv1d would convolve "
            "each node over the previous token in the batch instead of its own "
            "root-to-node chain, so non-chain nodes would verify wrong. Disable "
            "tree_spec_config or pick another gdn_state_update."
        )


def tree_conv_width(
    mamba_type,
    conv_state_shape: tuple[int, ...],
    num_spec: int,
) -> int | None:
    """Short-conv kernel width behind a GDN conv-state shape.

    The state is ``(width - 1 + num_spec) x conv_dim`` with ``conv_dim`` far larger, so
    the smaller side is the length. ``None`` for layers without a short conv.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    if mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
        return None
    if len(conv_state_shape) != 2:
        return None
    width = min(conv_state_shape) - num_spec + 1
    if width < 2:
        raise ValueError(
            f"cannot recover the short-conv width from the state shape "
            f"{tuple(conv_state_shape)} with num_speculative_tokens={num_spec}."
        )
    return width


def compute_tree_resume_column(
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
) -> torch.Tensor:
    """State slot the next step must resume from, as an int32 tensor.

    ``num_sampled`` counts accepted tokens including the trailing resample, so the last
    accepted node is ``path_node_ids[:, num_sampled - 2]``; ``num_sampled == 1`` (no
    accepted draft) keeps the root slot, matching ``num_accepted_tokens = 1``.
    """
    sampled = num_sampled.to(torch.long)
    last_index = (sampled - 2).clamp(min=0).unsqueeze(1)
    last_node = path_node_ids.to(torch.long).gather(1, last_index).squeeze(1)
    last_node = torch.where(sampled >= 2, last_node, torch.zeros_like(last_node))
    return (last_node + 1).to(torch.int32)


def tree_row_mask(
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> torch.Tensor:
    """Rows of the batch whose step ran the tree op, in batch order, as a bool.

    ``num_sampled`` alone cannot separate a root-only tree step from a row with no draft;
    the samplers set ``num_rejected = num_logits - num_sampled``, so the sum is the verified count.
    """
    return (num_sampled + num_rejected) > 1


def compute_tree_init_state_indices(
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
) -> torch.Tensor:
    """Per-token state page the recurrent kernel loads its initial state from.

    ``[R, slots]`` int32 ids in ``state_indices`` order, packed token-major by
    ``gdn_attn_builder._pack_tree_rows_to_tokens``. A row folded in from a prompt chunk
    points at each token's own page, keeping the kernel's in-register linear carry.
    """
    rows, slots = state_indices.shape
    if slots - 1 > tree_parents.shape[1]:
        raise ValueError(
            f"state table has {slots} slots but the tree layout only carries "
            f"{tree_parents.shape[1]} parents; tree_spec_config.budget must "
            "match the state slot count."
        )
    parents = tree_parents[:rows, : slots - 1].to(torch.long)
    # Chain rows: node ``t`` (1-based) continues the token before it.
    chain_parents = torch.arange(1, slots, device=state_indices.device, dtype=torch.long) - 1
    parents = torch.where(tree_num_nodes[:rows, None] > 0, parents, chain_parents[None, :])
    num_accepted = num_accepted_tokens[:rows].to(torch.long)
    accepted_slot = (num_accepted - 1).clamp(min=0).unsqueeze(1)
    root_init = torch.gather(state_indices, 1, accepted_slot)
    node_init = torch.gather(state_indices, 1, parents)
    init_state_indices = torch.cat((root_init, node_init), dim=1)
    return init_state_indices.to(torch.int32)


def tree_node_parents_depths(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
    slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node parent ids and depths of the draft tree, ``[R, slots]`` int64 each.

    The root's parent is ``-1``; folded prompt rows are chained (``t`` follows ``t - 1``)
    so they keep the batch-order window of the linear MTP chain.
    """
    rows = tree_parents.shape[0]
    device = tree_parents.device
    parents = torch.full((rows, slots), -1, dtype=torch.long, device=device)
    parents[:, 1:] = tree_parents[:rows, : slots - 1].to(torch.long)
    chain_parents = torch.arange(slots, dtype=torch.long, device=device) - 1
    parents = torch.where(tree_num_nodes[:rows, None] > 0, parents, chain_parents.expand(rows, slots))
    depths = torch.zeros((rows, slots), dtype=torch.long, device=device)
    for node in range(1, slots):
        # Builders guarantee ``parent_id < node_id``, so one forward sweep resolves depths.
        parent_of_node = parents[:, node].clamp(min=0).unsqueeze(1)
        depths[:, node] = depths.gather(1, parent_of_node).squeeze(1) + 1
    return parents, depths


def _tree_conv_tap_rows(
    chain: torch.Tensor,
    depths: torch.Tensor,
    own_rows: torch.Tensor,
    request_ids: torch.Tensor,
    node_row_table: torch.Tensor,
    history_base: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Rows of the ``width`` conv taps of every target, oldest tap first.

    Rows index ``cat([committed, activations])``; a tap before the root resolves to the
    committed history, keeping a shallow node's window identical to the linear chain.
    """
    taps = []
    for tap in range(width - 1):
        # Tap ``width - 2`` is the parent, ``width - 3`` the grandparent, ...
        ancestor = node_row_table[request_ids, chain[:, width - 2 - tap].clamp(min=0)]
        committed = history_base[request_ids] + (depths + tap).clamp(max=width - 2)
        taps.append(torch.where(depths >= (width - 1 - tap), ancestor, committed))
    taps.append(own_rows)
    return torch.stack(taps, dim=1)


def compute_tree_conv_rows(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
    width: int,
    node_row_offset: int,
    row_widths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token conv window of every node of a draft tree.

    Returns ``[R * slots, width]`` taps (oldest first) for ``tree_causal_conv1d``, indexing
    ``cat([committed history, this step's activations])``; only the first ``row_widths.sum()``
    rows are read. Node ``t`` convolves over its own root-to-node chain, never its siblings --
    which is exactly what the linear-chain conv kernel gets wrong.
    """
    rows = tree_parents.shape[0]
    slots = tree_parents.shape[1] + 1
    device = tree_parents.device
    full_width = row_widths is None
    if row_widths is not None:
        # Validated before the device transfer: a device-side ``bool(...)`` would sync.
        row_widths = row_widths.reshape(-1)
        if row_widths.numel() != rows:
            raise ValueError(f"row_widths must hold {rows} widths, got {row_widths.numel()}.")
        if bool(((row_widths < 1) | (row_widths > slots)).any()):
            raise ValueError(
                f"every draft-tree row verifies between 1 and {slots} tokens, got {row_widths.tolist()}."
            )
        full_width = row_widths.device.type == "cpu" and bool(torch.all(row_widths == slots))
        if not full_width:
            row_widths = row_widths.to(device=device, dtype=torch.long)
    parents, depths = tree_node_parents_depths(tree_parents, tree_num_nodes, slots)
    # Walk-up ancestors: column ``k`` is the ``k + 1``-th ancestor; ``clamp`` keeps the
    # gather in range where an ancestor is missing (the depth test discards those).
    chain = [parents]
    for _ in range(width - 2):
        chain.append(torch.gather(parents, 1, chain[-1].clamp(min=0)))
    chain = torch.stack(chain, dim=2)

    # Token-major packing: the ``p``-th token is node ``p - start(r)`` of row ``r``. The
    # tables keep their fixed ``R * slots`` shape, so cost is width-independent.
    positions = torch.arange(rows * slots, device=device)
    if full_width:
        # Fixed-width decode: token order already matches row-major tree order, so two
        # integer divisions suffice instead of the scan and row search used for short rows.
        token_rows = positions // slots
        token_cols = positions % slots
        node_row_table = node_row_offset + torch.arange(rows, device=device).unsqueeze(1) * slots
        node_row_table = node_row_table + torch.arange(slots, device=device).unsqueeze(0)
    else:
        starts = torch.cumsum(row_widths, dim=0) - row_widths
        ends = starts + row_widths
        token_rows = (positions.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1).clamp(max=rows - 1)
        token_cols = (positions - starts[token_rows]).clamp(max=slots - 1)
        node_row_table = node_row_offset + starts.unsqueeze(1) + torch.arange(slots, device=device).unsqueeze(0)

    window_rows = _tree_conv_tap_rows(
        chain=chain[token_rows, token_cols],
        depths=depths[token_rows, token_cols],
        own_rows=node_row_offset + positions,
        request_ids=token_rows,
        node_row_table=node_row_table,
        history_base=torch.arange(rows, device=device) * (width - 1),
        width=width,
    )
    return window_rows.to(torch.int32)


def compute_tree_prefix_source(
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Committed-history columns the next step reads its conv window from.

    The window ends at the last accepted node (depth ``num_sampled - 1``), so a tap at depth
    ``d`` resolves to node column ``width - 1 + id`` (``d >= 1``), the root's column
    ``width - 1`` (``d == 0``) or committed column ``width - 1 + d`` (``d < 0``). Rows the
    sampler skipped map the committed columns onto themselves.
    """
    rows = num_sampled.shape[0]
    device = num_sampled.device
    # Depth of the last accepted node; ``-1`` makes a skipped row's taps land on the
    # committed columns themselves.
    last_depth = (num_sampled.to(torch.long) - 1).clamp(min=-1).unsqueeze(1)
    depth = last_depth - (width - 2) + torch.arange(width - 1, device=device).unsqueeze(0)
    # Path entry ``depth - 1`` is exactly the node of that depth; lower entries are
    # masked out below, so the clamped index only has to stay in range.
    node_ids = path_node_ids[:rows].to(torch.long).gather(1, (depth - 1).clamp(min=0))
    return torch.where(depth >= 1, width - 1 + node_ids, width - 1 + depth).to(torch.int32)


@dataclass(frozen=True)
class TreeVerifyChunk:
    """One ``npu_recurrent_gated_delta_rule`` call of a wide draft-tree verify.

    Chunk ``c`` owns node columns ``[c * width, (c + 1) * width)`` of every row, always a
    prefix, so node ``j`` loads the parent page an earlier chunk already wrote. The token dim
    is a padded ``rows * width`` grid, keeping shapes fixed for full CUDA graphs, with
    ``cu_seqlens`` bounding what the kernel reads. ``-1`` pages make the kernel return without
    updating state, so a malformed cell is a no-op rather than a silent rollback; padding
    cells scatter to ``scratch_base``.
    """

    token_indices: torch.Tensor
    state_indices: torch.Tensor
    init_state_indices: torch.Tensor
    out_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    width: int


@dataclass(frozen=True)
class TreeVerifyPlan:
    """Chunked recurrent verify of one draft-tree step, built by :func:`compute_tree_verify_plan`
    and consumed by ``vllm_ascend.ops.gdn`` and ``AscendGDNAttentionMetadataBuilder``.
    """

    rows: int
    slots: int
    width: int
    chunks: tuple[TreeVerifyChunk, ...]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def chunk_tokens(self) -> int:
        """Token dim of every chunk: ``rows * width``."""
        return self.rows * self.width

    def copy_into(self, buffers: "TreeVerifyPlanBuffers") -> "TreeVerifyPlan":
        """Write this plan into stable buffers and return the matching views.

        A full CUDA graph reads its inputs by address, so the recorded chunk tensors are
        refreshed in place per replay (plan rebuilt per step, buffers live in the builder).
        """
        size = self.chunk_tokens
        chunks = []
        for index, chunk in enumerate(self.chunks):
            token_indices = buffers.token_indices[index][:size]
            token_indices.copy_(chunk.token_indices)
            state_indices = buffers.state_indices[index][:size]
            state_indices.copy_(chunk.state_indices)
            init_state_indices = buffers.init_state_indices[index][:size]
            init_state_indices.copy_(chunk.init_state_indices)
            out_indices = buffers.out_indices[index][:size]
            out_indices.copy_(chunk.out_indices)
            cu_seqlens = buffers.cu_seqlens[index][: self.rows + 1]
            cu_seqlens.copy_(chunk.cu_seqlens)
            chunks.append(
                replace(
                    chunk,
                    token_indices=token_indices,
                    state_indices=state_indices,
                    init_state_indices=init_state_indices,
                    out_indices=out_indices,
                    cu_seqlens=cu_seqlens,
                )
            )
        return TreeVerifyPlan(rows=self.rows, slots=self.slots, width=self.width, chunks=tuple(chunks))


@dataclass
class TreeVerifyPlanBuffers:
    """Stable per-chunk tensors of a :class:`TreeVerifyPlan` (see ``copy_into``).

    One row per chunk, sized for the graph's row capacity so a replay takes the live count.
    """

    token_indices: torch.Tensor
    state_indices: torch.Tensor
    init_state_indices: torch.Tensor
    out_indices: torch.Tensor
    cu_seqlens: torch.Tensor

    def zero_lengths(self) -> None:
        """Turn every recorded chunk into a no-op call.

        A replay with no draft tree (idle, prefill) must not rerun stale indices; zero row
        lengths make the kernel skip every chunk.
        """
        self.cu_seqlens.zero_()


def tree_verify_chunk_count(
    slots: int,
    width: int = GDN_RECURRENT_MAX_VERIFY_TOKENS,
) -> int:
    """Number of recurrent-op calls a ``1 + budget = slots`` verify needs."""
    if width < 1:
        raise ValueError(f"the verify chunk width must be positive, got {width}.")
    if slots < 1:
        raise ValueError(f"a verify step holds at least one token per request, got slots={slots}.")
    return -(-slots // width)


def allocate_tree_verify_plan_buffers(
    slots: int,
    rows: int,
    device: torch.device,
    width: int = GDN_RECURRENT_MAX_VERIFY_TOKENS,
) -> TreeVerifyPlanBuffers:
    """Allocate the stable chunk tensors of a ``rows``-row, ``slots``-wide verify."""
    chunks = tree_verify_chunk_count(slots, width)
    tokens = rows * width
    return TreeVerifyPlanBuffers(
        token_indices=torch.empty((chunks, tokens), dtype=torch.long, device=device),
        state_indices=torch.empty((chunks, tokens), dtype=torch.int32, device=device),
        init_state_indices=torch.empty((chunks, tokens), dtype=torch.int32, device=device),
        out_indices=torch.empty((chunks, tokens), dtype=torch.long, device=device),
        cu_seqlens=torch.empty((chunks, rows + 1), dtype=torch.int32, device=device),
    )


def compute_tree_verify_plan(
    row_widths: torch.Tensor,
    state_indices: torch.Tensor,
    init_state_indices: torch.Tensor,
    scratch_base: int,
    width: int = GDN_RECURRENT_MAX_VERIFY_TOKENS,
) -> TreeVerifyPlan | None:
    """Cut a draft-tree verify into recurrent-op calls of at most ``width`` tokens.

    Returns ``None`` when the whole tree fits one call. Widths are trusted, not re-validated:
    the builder checks them on CPU data (a ``bool()`` here would sync the device per layer per
    step) and passes 0 for padded graph rows.
    """
    if state_indices.dim() != 2 or init_state_indices.shape != state_indices.shape:
        raise ValueError(
            f"the state tables must be a pair of [rows, slots] tensors, got "
            f"{tuple(state_indices.shape)} and {tuple(init_state_indices.shape)}."
        )
    rows, slots = state_indices.shape
    if slots <= width:
        return None
    if scratch_base < 0:
        raise ValueError(f"the output scratch base must be non-negative, got {scratch_base}.")
    device = state_indices.device
    widths = torch.as_tensor(row_widths, device=device, dtype=torch.long).reshape(-1)
    if widths.numel() != rows:
        raise ValueError(f"got {widths.numel()} row widths for {rows} state rows.")

    token_starts = torch.cumsum(widths, dim=0) - widths
    chunk_starts = torch.arange(0, slots, width, device=device, dtype=torch.long)
    cells = torch.arange(width, device=device, dtype=torch.long)
    # Node columns of every row inside every chunk: [chunks, rows, width].
    row_ends = torch.minimum(chunk_starts[:, None] + width, widths[None, :])
    row_lengths = (row_ends - chunk_starts[:, None]).clamp(min=0)
    owned = cells[None, None, :] < row_lengths[:, :, None]
    tokens = token_starts[None, :, None] + chunk_starts[:, None, None] + cells[None, None, :]
    # Padding cells are only gathered, never read, so index 0 keeps the gather in bounds.
    tokens = torch.where(owned, tokens, torch.zeros_like(tokens))

    flat_state = state_indices.reshape(-1)
    flat_init = init_state_indices.reshape(-1)
    minus_one = torch.tensor(-1, dtype=torch.int32, device=device)
    scratch = torch.arange(rows * width, device=device, dtype=torch.long) + scratch_base
    chunks = []
    for index in range(chunk_starts.numel()):
        chunk_tokens = tokens[index].reshape(-1)
        chunk_owned = owned[index].reshape(-1)
        chunks.append(
            TreeVerifyChunk(
                token_indices=chunk_tokens,
                state_indices=torch.where(
                    chunk_owned,
                    flat_state.index_select(0, chunk_tokens),
                    minus_one,
                ),
                init_state_indices=torch.where(
                    chunk_owned,
                    flat_init.index_select(0, chunk_tokens),
                    minus_one,
                ),
                out_indices=torch.where(chunk_owned, chunk_tokens, scratch),
                cu_seqlens=torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32, device=device),
                        row_lengths[index].to(torch.int32),
                    )
                ),
                width=width,
            )
        )
    return TreeVerifyPlan(rows=rows, slots=slots, width=width, chunks=tuple(chunks))

