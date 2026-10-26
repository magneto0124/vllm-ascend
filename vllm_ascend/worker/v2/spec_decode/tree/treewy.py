# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TreeWY GDN verification and accepted-state reconstruction.

Per node ``t`` with parent ``p(t)``: ``S_t = exp(g_t) S_{p(t)} + u_t k_t^T`` and
``u_t = beta_t (v_t - exp(g_t) S_{p(t)} k_t)``. Only ``u_t`` is stored per node; the
accepted state is rebuilt from those pseudo-values along the accepted path. The reference
functions are slow, synchronizing test oracles; the tensor-only ones are the current
Ascend execution path until a fused operator lands.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TreeWYReferenceResult:
    """Intermediate values needed to verify outputs and reconstruct a state."""

    output: torch.Tensor
    pseudo_values: torch.Tensor
    log_decay: torch.Tensor


@dataclass(frozen=True)
class TreeWYVerificationResult:
    """Packed outputs and intermediates retained until sampler acceptance."""

    output: torch.Tensor
    pseudo_values: torch.Tensor
    log_decay: torch.Tensor
    ancestors: torch.Tensor | None = None


@dataclass(frozen=True)
class TreeWYLayerContext:
    """Per-layer tensors kept from verification until the sampler commits a path."""

    key: torch.Tensor
    pseudo_values: torch.Tensor
    log_decay: torch.Tensor
    tree_parents: torch.Tensor
    tree_num_nodes: torch.Tensor
    row_widths: torch.Tensor
    sequence_indices: torch.Tensor
    root_state_indices: torch.Tensor
    initial_state: torch.Tensor
    state: torch.Tensor
    ancestors: torch.Tensor | None = None


def _row_parents(
    parents: list[list[int]],
    tree_num_nodes: list[int] | None,
    row: int,
    node_count: int,
) -> list[int]:
    """Return virtual-root parent ids, with ``-1`` denoting committed state."""
    result = [-1]
    is_chain = tree_num_nodes is not None and tree_num_nodes[row] == 0
    for node in range(1, node_count):
        parent = node - 1 if is_chain else parents[row][node - 1]
        if parent < 0 or parent >= node:
            raise ValueError(
                f"node {node} in row {row} must have an earlier parent, got {parent}"
            )
        result.append(parent)
    return result


def treewy_reference_verify(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    tree_parents: torch.Tensor,
    row_widths: torch.Tensor,
    scale: float,
    tree_num_nodes: torch.Tensor | None = None,
) -> TreeWYReferenceResult:
    """Run the GDN recurrence in TreeWY form as a test-only oracle.

    Same input layouts as :func:`treewy_verify_torch`; CPU tensors, synchronizing by
    design. A row with ``tree_num_nodes == 0`` is a linear chain.
    """
    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError("query and key must have matching [tokens, heads, dim] shapes")
    if value.ndim != 3 or g.ndim != 2 or beta.ndim not in (2, 3):
        raise ValueError("value, g, and beta must use [tokens, heads, dim] layouts")
    if tree_parents.ndim != 2 or row_widths.ndim != 1:
        raise ValueError("tree_parents and row_widths must have [rows, slots - 1] and [rows] layouts")
    rows, parent_columns = tree_parents.shape
    slots = parent_columns + 1
    if row_widths.numel() != rows or initial_state.ndim != 4 or initial_state.shape[0] != rows:
        raise ValueError("row_widths, tree_parents, and initial_state row counts must match")
    if value.shape[0] != query.shape[0] or g.shape[0] != query.shape[0]:
        raise ValueError("query, value, and g must have the same token count")
    if value.shape[1] != initial_state.shape[1] or value.shape[2] != initial_state.shape[2]:
        raise ValueError("value and initial_state head dimensions must match")
    if initial_state.shape[3] != key.shape[2] or g.shape[1] != value.shape[1]:
        raise ValueError("key/value/state feature dimensions do not match")
    if beta.shape[:2] != value.shape[:2]:
        raise ValueError("beta must align with value tokens and heads")
    if beta.ndim == 3 and beta.shape[2] != value.shape[2]:
        raise ValueError("per-value beta must have the same value dimension as value")
    value_heads = value.shape[1]
    key_heads = key.shape[1]
    if key_heads == 0 or value_heads % key_heads:
        raise ValueError("value_heads must be divisible by key_heads")
    if tree_num_nodes is not None and (tree_num_nodes.ndim != 1 or tree_num_nodes.numel() != rows):
        raise ValueError("tree_num_nodes must have one entry per row")

    widths = [int(width) for width in row_widths.to(device="cpu").tolist()]
    if any(width < 0 or width > slots for width in widths):
        raise ValueError(f"row widths must be between 0 and {slots}, got {widths}")
    parents_cpu = tree_parents.to(device="cpu", dtype=torch.long).tolist()
    tree_counts = None
    if tree_num_nodes is not None:
        tree_counts = tree_num_nodes.to(device="cpu", dtype=torch.long).tolist()

    output = torch.zeros_like(value)
    pseudo_values = torch.zeros_like(value, dtype=torch.float32)
    log_decay = torch.zeros(
        (rows, slots, value_heads), dtype=torch.float32, device=value.device
    )
    token_starts: list[int] = []
    token_start = 0
    for width in widths:
        token_starts.append(token_start)
        token_start += width
    if token_start != query.shape[0]:
        raise ValueError(
            f"row widths sum to {token_start}, but the packed token count is {query.shape[0]}"
        )

    query_group_size = value_heads // key_heads
    for row, node_count in enumerate(widths):
        parents = _row_parents(parents_cpu, tree_counts, row, node_count)
        start = token_starts[row]
        row_log_decay: list[torch.Tensor] = []
        for node in range(node_count):
            token = start + node
            current_log_decay = g[token].float()
            parent = parents[node]
            if parent >= 0:
                current_log_decay = current_log_decay + row_log_decay[parent]
            row_log_decay.append(current_log_decay)
            log_decay[row, node] = current_log_decay

        for node in range(node_count):
            token = start + node
            node_log_decay = row_log_decay[node]
            strict_ancestors = _ancestor_chain(parents, node)
            ancestors = _ancestor_chain(parents, node, include_self=True)
            for value_head in range(value_heads):
                key_head = value_head // query_group_size
                key_t = key[token, key_head].float()
                query_t = query[token, key_head].float() * scale
                state0 = initial_state[row, value_head].float()
                decay_t = torch.exp(node_log_decay[value_head])
                predicted_value = decay_t * torch.mv(state0, key_t)
                for ancestor in strict_ancestors:
                    ancestor_token = start + ancestor
                    ratio = torch.exp(node_log_decay[value_head] - row_log_decay[ancestor][value_head])
                    key_i = key[ancestor_token, key_head].float()
                    predicted_value = predicted_value + (
                        ratio
                        * torch.dot(key_t, key_i)
                        * pseudo_values[ancestor_token, value_head]
                    )

                beta_t = beta[token, value_head].float()
                current_pseudo = beta_t * (value[token, value_head].float() - predicted_value)
                pseudo_values[token, value_head] = current_pseudo

                output_t = decay_t * torch.mv(state0, query_t)
                for ancestor in ancestors:
                    ancestor_token = start + ancestor
                    ratio = torch.exp(node_log_decay[value_head] - row_log_decay[ancestor][value_head])
                    key_i = key[ancestor_token, key_head].float()
                    output_t = output_t + (
                        ratio
                        * torch.dot(key_i, query_t)
                        * pseudo_values[ancestor_token, value_head]
                    )
                output[token, value_head] = output_t.to(output.dtype)

    return TreeWYReferenceResult(output, pseudo_values, log_decay)


def _ancestor_chain(
    parents: list[int],
    node: int,
    include_self: bool = False,
) -> list[int]:
    """List node ids from root to ``node`` (or its strict ancestors)."""
    current = node if include_self else parents[node]
    chain = []
    while current >= 0:
        chain.append(current)
        current = parents[current]
    chain.reverse()
    return chain


def treewy_reference_commit(
    initial_state: torch.Tensor,
    key: torch.Tensor,
    pseudo_values: torch.Tensor,
    log_decay: torch.Tensor,
    tree_parents: torch.Tensor,
    row_widths: torch.Tensor,
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    tree_num_nodes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct each accepted state from only its pseudo-values and keys.

    ``num_sampled`` includes the resample, so the accepted draft node is
    ``path_node_ids[num_sampled - 2]`` (the root if none); chain rows commit their last node.
    """
    if tree_parents.ndim != 2:
        raise ValueError("tree_parents must have [rows, slots - 1] shape")
    rows, parent_columns = tree_parents.shape
    slots = parent_columns + 1
    if initial_state.ndim != 4 or initial_state.shape[0] != rows:
        raise ValueError("initial_state must have [rows, heads, value_dim, key_dim] shape")
    if key.ndim != 3 or row_widths.ndim != 1 or path_node_ids.ndim != 2 or num_sampled.ndim != 1:
        raise ValueError("key, row_widths, path_node_ids, and num_sampled have invalid ranks")
    if (
        row_widths.numel() != rows
        or path_node_ids.shape[0] != rows
        or num_sampled.numel() != rows
    ):
        raise ValueError("commit inputs must have the same row count")
    if pseudo_values.shape != (key.shape[0], initial_state.shape[1], initial_state.shape[2]):
        raise ValueError("pseudo_values must align with the packed key/value token layout")
    if key.shape[-1] != initial_state.shape[-1]:
        raise ValueError("key and initial_state must have the same key dimension")
    if log_decay.shape != (rows, slots, initial_state.shape[1]):
        raise ValueError("log_decay must have [rows, slots, value_heads] shape")
    if key.shape[1] == 0 or initial_state.shape[1] % key.shape[1]:
        raise ValueError("value_heads must be divisible by key_heads")

    widths = [int(width) for width in row_widths.to(device="cpu").tolist()]
    if any(width < 0 or width > slots for width in widths):
        raise ValueError(f"row widths must be between 0 and {slots}, got {widths}")
    if sum(widths) != key.shape[0]:
        raise ValueError(
            f"row widths sum to {sum(widths)}, but the packed token count is {key.shape[0]}"
        )
    parents_cpu = tree_parents.to(device="cpu", dtype=torch.long).tolist()
    tree_counts = None
    if tree_num_nodes is not None:
        if tree_num_nodes.ndim != 1 or tree_num_nodes.numel() != rows:
            raise ValueError("tree_num_nodes must have one entry per row")
        tree_counts = tree_num_nodes.to(device="cpu", dtype=torch.long).tolist()
    sampled = [int(count) for count in num_sampled.to(device="cpu").tolist()]
    paths = path_node_ids.to(device="cpu", dtype=torch.long).tolist()
    if any(count < 0 or count > width for count, width in zip(sampled, widths)):
        raise ValueError(f"num_sampled must be between zero and each row width, got {sampled}")

    final_state = initial_state.float().clone()
    start = 0
    for row, width in enumerate(widths):
        count = sampled[row]
        if width == 0:
            start += width
            continue
        parents = _row_parents(parents_cpu, tree_counts, row, width)
        is_chain = tree_counts is not None and tree_counts[row] == 0
        if is_chain:
            # A folded prompt chunk has no sampler output (count == 0), but
            # every token in its linear verify row still advances recurrent
            # state. Tree rows with count == 0, by contrast, have no accepted
            # path and retain the committed root state.
            accepted_node = width - 1 if count == 0 else min(count - 1, width - 1)
        elif count == 0:
            start += width
            continue
        elif count == 1:
            accepted_node = 0
        else:
            accepted_draft_count = count - 1
            accepted_path = paths[row][:accepted_draft_count]
            if len(accepted_path) != accepted_draft_count or any(node < 0 for node in accepted_path):
                raise ValueError(f"accepted path is missing its final node in row {row}")
            for index, node in enumerate(accepted_path):
                if node >= width:
                    raise ValueError(
                        f"accepted path node {node} is outside row {row}'s width {width}"
                    )
                expected_parent = 0 if index == 0 else accepted_path[index - 1]
                if parents[node] != expected_parent:
                    raise ValueError(
                        f"accepted path in row {row} is not a root-to-leaf path: "
                        f"node {node} has parent {parents[node]}, expected {expected_parent}"
                    )
            accepted_node = accepted_path[-1]
        if accepted_node < 0 or accepted_node >= width:
            raise ValueError(
                f"accepted node {accepted_node} is outside row {row}'s width {width}"
            )

        accepted_log_decay = log_decay[row, accepted_node].float()
        for value_head in range(initial_state.shape[1]):
            state = torch.exp(accepted_log_decay[value_head]) * initial_state[row, value_head].float()
            for ancestor in _ancestor_chain(parents, accepted_node, include_self=True):
                ratio = torch.exp(
                    accepted_log_decay[value_head] - log_decay[row, ancestor, value_head].float()
                )
                ancestor_token = start + ancestor
                key_head = value_head // (initial_state.shape[1] // key.shape[1])
                key_i = key[ancestor_token, key_head].float()
                pseudo_i = pseudo_values[ancestor_token, value_head].float()
                state = state + ratio * pseudo_i.unsqueeze(-1) * key_i.unsqueeze(0)
            final_state[row, value_head] = state
        start += width

    return final_state.to(initial_state.dtype)


def _tree_token_indices(
    row_widths: torch.Tensor,
    slots: int,
    token_count: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Build a shared gather map; a single padded row needs no gather."""
    rows = row_widths.shape[0]
    widths = row_widths.to(dtype=torch.long)
    node_ids = torch.arange(slots, device=widths.device)
    valid_nodes = node_ids[None, :] < widths[:, None]
    # Device-side mask: ACL graph replay can shorten a row without changing the captured
    # shape; multiple rows still need packing (one width shifts all later token offsets).
    if rows == 1 and token_count == slots:
        return None, valid_nodes
    starts = torch.cumsum(widths, dim=0) - widths
    token_indices = starts[:, None] + node_ids[None, :]
    return token_indices.clamp(min=0, max=max(0, token_count - 1)), valid_nodes


def _packed_tokens_to_tree_grid(
    tensor: torch.Tensor,
    token_indices: torch.Tensor | None,
    valid_nodes: torch.Tensor,
    rows: int,
    slots: int,
) -> torch.Tensor:
    """View full rows directly or gather ragged packed tokens using shared maps."""
    if tensor.shape[0] == 0:
        return tensor.new_zeros((rows, slots, *tensor.shape[1:]))
    if token_indices is None:
        grid = tensor.reshape(rows, slots, *tensor.shape[1:])
    else:
        grid = tensor.index_select(0, token_indices.reshape(-1)).reshape(rows, slots, *tensor.shape[1:])
    mask = valid_nodes.reshape(rows, slots, *([1] * (tensor.ndim - 1)))
    return torch.where(mask, grid, 0)


def _tree_parent_and_ancestor_tensors(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor | None,
    slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build inclusive ancestor masks with logarithmic-depth pointer jumping."""
    rows = tree_parents.shape[0]
    device = tree_parents.device
    parents = torch.full((rows, slots), -1, dtype=torch.long, device=device)
    node_ids = torch.arange(slots, dtype=torch.long, device=device)
    if slots > 1:
        parents[:, 1:] = tree_parents[:, : slots - 1].to(torch.long)
        if tree_num_nodes is not None:
            chain_parents = node_ids[None, :] - 1
            parents = torch.where(
                tree_num_nodes.to(device=device).reshape(rows, 1) == 0,
                chain_parents.expand(rows, slots),
                parents,
            )

    ancestors = torch.eye(slots, dtype=torch.bool, device=device).expand(rows, slots, slots)
    jump = parents
    distance = 1
    while distance < slots:
        indices = jump.clamp(min=0)
        inherited = ancestors.gather(1, indices[:, :, None].expand(rows, slots, slots))
        ancestors = ancestors | (inherited & (jump >= 0)[:, :, None])
        jump = jump.gather(1, indices)
        distance *= 2
    return parents, ancestors


def treewy_ancestor_mask(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prepare topology once per batch for all GDN layers sharing metadata."""
    return _tree_parent_and_ancestor_tensors(
        tree_parents, tree_num_nodes, tree_parents.shape[1] + 1
    )[1]


def _tree_grid_to_packed_indices(
    row_widths: torch.Tensor,
    slots: int,
    token_count: int,
) -> torch.Tensor | None:
    """Build a shared inverse map, or signal that a full grid is already packed."""
    widths = row_widths.to(dtype=torch.long)
    if widths.shape[0] == 1 and token_count == slots:
        return None
    starts = torch.cumsum(widths, dim=0) - widths
    ends = starts + widths
    positions = torch.arange(token_count, device=widths.device)
    row_ids = (positions[:, None] >= ends[None, :]).sum(dim=1).clamp(max=widths.shape[0] - 1)
    node_ids = positions - starts[row_ids]
    return row_ids * slots + node_ids


def treewy_verify_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    tree_parents: torch.Tensor,
    row_widths: torch.Tensor,
    scale: float,
    tree_num_nodes: torch.Tensor | None = None,
    *,
    ancestors: torch.Tensor | None = None,
) -> TreeWYVerificationResult:
    """Tensor-only TreeWY verify path suitable for device execution.

    Loop bounds depend only on tensor shapes: no host reads, no per-node state snapshots.
    ``ancestors`` may be prepared once by the metadata builder (refresh before graph replay).
    """
    rows, parent_columns = tree_parents.shape
    slots = parent_columns + 1
    widths = row_widths.to(device=query.device)
    token_indices, valid_nodes = _tree_token_indices(widths, slots, query.shape[0])
    query_grid = _packed_tokens_to_tree_grid(query, token_indices, valid_nodes, rows, slots)
    key_grid = _packed_tokens_to_tree_grid(key, token_indices, valid_nodes, rows, slots)
    value_grid = _packed_tokens_to_tree_grid(value, token_indices, valid_nodes, rows, slots)
    g_grid = _packed_tokens_to_tree_grid(g, token_indices, valid_nodes, rows, slots)
    beta_grid = _packed_tokens_to_tree_grid(beta, token_indices, valid_nodes, rows, slots)

    value_heads = value.shape[1]
    key_heads = key.shape[1]
    group_size = value_heads // key_heads
    value_to_key_heads = torch.arange(value_heads, device=key.device) // group_size
    query_heads = query_grid.index_select(2, value_to_key_heads).permute(0, 2, 1, 3).float() * scale
    key_heads_grid = key_grid.index_select(2, value_to_key_heads).permute(0, 2, 1, 3).float()
    if ancestors is None:
        _, ancestors = _tree_parent_and_ancestor_tensors(tree_parents, tree_num_nodes, slots)
    log_decay = torch.bmm(ancestors.float(), g_grid.float())
    log_decay_heads = log_decay.permute(0, 2, 1)
    decay = log_decay_heads.exp().unsqueeze(-1)
    # Non-ancestor differences may be positive: clamp BEFORE exp (masked to zero later).
    decay_ratio = (log_decay_heads.unsqueeze(-1) - log_decay_heads.unsqueeze(-2)).clamp(max=0.0).exp()
    inclusive_weights = decay_ratio * ancestors[:, None]
    strict_weights = inclusive_weights * ~torch.eye(slots, dtype=torch.bool, device=key.device)
    key_transposed = key_heads_grid.transpose(-1, -2)
    dependencies = torch.matmul(key_heads_grid, key_transposed) * strict_weights
    state_transposed = initial_state.float().transpose(-1, -2)
    residual = value_grid.permute(0, 2, 1, 3).float() - decay * torch.matmul(key_heads_grid, state_transposed)

    if beta.ndim == 2:
        beta_heads = beta_grid.permute(0, 2, 1).float().unsqueeze(-1)
        lower = beta_heads * dependencies
        # Direct solve as in the reference; forming the inverse adds rounding and many
        # small matmuls. unitriangular=True ignores the stored zero diagonal.
        pseudo_heads = torch.linalg.solve_triangular(lower, beta_heads * residual, upper=False, unitriangular=True)
    else:
        # Per-value beta gives a system per channel: substitute rather than allocate [R,H,V,N,N].
        beta_heads = beta_grid.permute(0, 2, 1, 3).float()
        pseudo_rows = []
        for node in range(slots):
            residual_node = residual[:, :, node]
            if node > 0:
                previous = torch.stack(pseudo_rows, dim=2)
                correction = torch.matmul(dependencies[:, :, node : node + 1, :node], previous).squeeze(2)
                residual_node = residual_node - correction
            pseudo_rows.append(beta_heads[:, :, node] * residual_node)
        pseudo_heads = torch.stack(pseudo_rows, dim=2)

    pseudo_heads = pseudo_heads * valid_nodes[:, None, :, None]
    output_weights = torch.matmul(query_heads, key_transposed) * inclusive_weights
    output_heads = decay * torch.matmul(query_heads, state_transposed) + torch.matmul(output_weights, pseudo_heads)
    output_heads = output_heads * valid_nodes[:, None, :, None]
    output_grid = output_heads.permute(0, 2, 1, 3).to(value.dtype)
    pseudo_grid = pseudo_heads.permute(0, 2, 1, 3)
    packed_indices = _tree_grid_to_packed_indices(widths, slots, query.shape[0])
    if packed_indices is None:
        output = output_grid.reshape(query.shape[0], *output_grid.shape[2:])
        pseudo_values = pseudo_grid.reshape(key.shape[0], *pseudo_grid.shape[2:])
    else:
        output = output_grid.flatten(0, 1).index_select(0, packed_indices)
        pseudo_values = pseudo_grid.flatten(0, 1).index_select(0, packed_indices)
    return TreeWYVerificationResult(output, pseudo_values, log_decay, ancestors)


def treewy_commit_torch(
    initial_state: torch.Tensor,
    key: torch.Tensor,
    pseudo_values: torch.Tensor,
    log_decay: torch.Tensor,
    tree_parents: torch.Tensor,
    row_widths: torch.Tensor,
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    tree_num_nodes: torch.Tensor | None = None,
    *,
    ancestors: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct the sampled state using device tensors only.

    ``initial_state`` is the committed state at the start of verification; rows with no
    sampled token or no verified node keep it unchanged.
    """
    rows, parent_columns = tree_parents.shape
    slots = parent_columns + 1
    token_indices, valid_nodes = _tree_token_indices(row_widths.to(device=key.device), slots, key.shape[0])
    key_grid = _packed_tokens_to_tree_grid(key, token_indices, valid_nodes, rows, slots)
    pseudo_grid = _packed_tokens_to_tree_grid(pseudo_values, token_indices, valid_nodes, rows, slots)
    value_heads = initial_state.shape[1]
    key_heads = key.shape[1]
    group_size = value_heads // key_heads
    value_to_key_heads = torch.arange(value_heads, device=key.device) // group_size
    key_grid = key_grid.index_select(2, value_to_key_heads).float()
    if ancestors is None:
        _, ancestors = _tree_parent_and_ancestor_tensors(tree_parents, tree_num_nodes, slots)

    sampled = num_sampled.to(device=initial_state.device, dtype=torch.long)
    if path_node_ids.shape[1] > 0:
        path_index = (sampled - 2).clamp(min=0, max=path_node_ids.shape[1] - 1)
        last_path_node = path_node_ids.to(torch.long).gather(1, path_index[:, None]).squeeze(1)
    else:
        last_path_node = torch.zeros_like(sampled)
    chain_rows = torch.zeros_like(sampled, dtype=torch.bool)
    if tree_num_nodes is not None:
        chain_rows = tree_num_nodes.to(device=sampled.device) == 0
    accepted_nodes = torch.where(sampled >= 2, last_path_node, torch.zeros_like(sampled))
    chain_node = torch.where(sampled > 0, sampled - 1, row_widths.to(sampled.device) - 1)
    accepted_nodes = torch.where(chain_rows, chain_node, accepted_nodes)
    accepted_nodes = accepted_nodes.clamp(min=0, max=slots - 1)

    accepted_ancestors = ancestors.gather(
        1, accepted_nodes[:, None, None].expand(rows, 1, slots)
    ).squeeze(1)
    accepted_log_decay = log_decay.gather(
        1, accepted_nodes[:, None, None].expand(rows, 1, value_heads)
    ).squeeze(1)
    state = accepted_log_decay.exp()[:, :, None, None] * initial_state.float()
    # Apply all accepted rank-one updates in one batched matmul, not one [R,H,V,K] rewrite per node.
    ratio = (accepted_log_decay[:, None] - log_decay.float()).clamp(max=0.0).exp()
    coefficient = ratio * (accepted_ancestors & valid_nodes)[:, :, None]
    weighted_pseudo = (coefficient[:, :, :, None] * pseudo_grid.float()).permute(0, 2, 3, 1)
    state = state + torch.matmul(weighted_pseudo, key_grid.permute(0, 2, 1, 3))

    active_rows = ((sampled > 0) | chain_rows) & (row_widths.to(device=sampled.device) > 0)
    state = torch.where(active_rows[:, None, None, None], state, initial_state.float())
    return state.to(initial_state.dtype)


def treewy_commit_contexts_torch(
    contexts: dict[str, TreeWYLayerContext],
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
) -> None:
    """Commit the accepted recurrent state for each GDN layer in place."""
    for context in contexts.values():
        row_indices = context.sequence_indices.to(torch.long)
        request_count = min(path_node_ids.shape[0], num_sampled.shape[0])
        if request_count == 0:
            continue
        valid_rows = (row_indices >= 0) & (row_indices < request_count)
        safe_row_indices = row_indices.clamp(min=0, max=request_count - 1)
        sampled = num_sampled.index_select(0, safe_row_indices)
        sampled = torch.where(valid_rows, sampled, torch.zeros_like(sampled))
        accepted_path = path_node_ids.index_select(0, safe_row_indices)
        accepted_path = accepted_path.masked_fill(~valid_rows[:, None], -1)
        accepted_state = treewy_commit_torch(
            context.initial_state,
            context.key,
            context.pseudo_values,
            context.log_decay,
            context.tree_parents,
            context.row_widths,
            accepted_path,
            sampled,
            context.tree_num_nodes,
            ancestors=context.ancestors,
        )
        chain_rows = context.tree_num_nodes == 0
        # FULL graphs may carry padded rows with a full verify width, though the sampler
        # only returns real requests; a padded row can map to a reused state page, so its
        # synthetic chain result must never overwrite a real request's committed state.
        active = valid_rows & ((sampled > 0) | chain_rows) & (context.row_widths > 0)
        pages = context.root_state_indices.to(torch.long)[active]
        context.state.index_copy_(0, pages, accepted_state[active])
