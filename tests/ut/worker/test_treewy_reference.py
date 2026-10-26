# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_ascend.worker.v2.spec_decode.tree.treewy as treewy_module

from vllm_ascend.worker.v2.spec_decode.tree.treewy import (
    TreeWYLayerContext,
    treewy_commit_contexts_torch,
    treewy_commit_torch,
    treewy_reference_commit,
    treewy_reference_verify,
    treewy_verify_torch,
)


def _reference_inputs():
    torch.manual_seed(17)
    rows = 2
    slots = 5
    key_heads = 2
    value_heads = 4
    key_dim = 3
    value_dim = 3
    token_count = rows * slots
    query = torch.randn(token_count, key_heads, key_dim)
    key = torch.randn_like(query)
    value = torch.randn(token_count, value_heads, value_dim)
    g = -torch.rand(token_count, value_heads) * 0.2
    beta = torch.rand(token_count, value_heads)
    initial_state = torch.randn(rows, value_heads, value_dim, key_dim)
    # Row 0 branches at nodes 1 and 2; row 1 is the linear-chain case.
    parents = torch.tensor([[0, 0, 1, 0], [0, 1, 2, 3]], dtype=torch.int32)
    row_widths = torch.tensor([slots, slots], dtype=torch.int32)
    tree_num_nodes = torch.tensor([slots - 1, 0], dtype=torch.int32)
    return query, key, value, g, beta, initial_state, parents, row_widths, tree_num_nodes


def _run_token_recurrence(
    query, key, value, g, beta, initial_state, parents, row_widths, tree_num_nodes, scale
):
    rows = initial_state.shape[0]
    slots = parents.shape[1] + 1
    value_heads = value.shape[1]
    key_heads = key.shape[1]
    group_size = value_heads // key_heads
    outputs = torch.empty_like(value)
    pseudo_values = torch.empty_like(value)
    states = torch.empty(
        (rows, slots, value_heads, value.shape[-1], key.shape[-1]),
        dtype=initial_state.dtype,
    )
    starts = torch.cumsum(
        torch.cat((row_widths.new_zeros(1), row_widths[:-1])), dim=0
    ).tolist()

    for row in range(rows):
        for node in range(int(row_widths[row])):
            token = starts[row] + node
            if node == 0:
                previous_state = initial_state[row]
            else:
                parent = (
                    node - 1
                    if tree_num_nodes[row] == 0
                    else int(parents[row, node - 1])
                )
                previous_state = states[row, parent]
            key_by_value_head = key[token].repeat_interleave(group_size, dim=0)
            query_by_value_head = query[token].repeat_interleave(group_size, dim=0) * scale
            decayed_state = previous_state * g[token].exp()[:, None, None]
            prediction = torch.einsum("hvd,hd->hv", decayed_state, key_by_value_head)
            beta_t = beta[token, :, None] if beta.ndim == 2 else beta[token]
            pseudo = beta_t * (value[token] - prediction)
            state = decayed_state + pseudo.unsqueeze(-1) * key_by_value_head.unsqueeze(-2)
            output = torch.einsum("hvd,hd->hv", state, query_by_value_head)
            pseudo_values[token] = pseudo
            states[row, node] = state
            outputs[token] = output

    return outputs, pseudo_values, states


def test_treewy_reference_matches_token_recurrence_on_tree_and_chain_rows():
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    rows = initial_state.shape[0]
    scale = key.shape[-1] ** -0.5

    result = treewy_reference_verify(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        scale,
        tree_num_nodes,
    )
    tensor_result = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        scale,
        tree_num_nodes,
    )
    expected_output, expected_pseudo, expected_states = _run_token_recurrence(
        query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes, scale
    )

    torch.testing.assert_close(result.output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result.pseudo_values, expected_pseudo, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_result.output, result.output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_result.pseudo_values, result.pseudo_values, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_result.log_decay, result.log_decay, rtol=1e-5, atol=1e-5)
    expected_decay = _expected_cumulative_decay(g, widths, parents, tree_num_nodes)
    torch.testing.assert_close(result.log_decay.exp(), expected_decay.exp())

    # Accepted path root -> node 1 -> node 3; the chain row commits its third node.
    path_node_ids = torch.tensor([[1, 3, -1, -1], [-1, -1, -1, -1]])
    num_sampled = torch.tensor([3, 4], dtype=torch.int32)
    committed = treewy_reference_commit(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )
    tensor_committed = treewy_commit_torch(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )

    torch.testing.assert_close(committed[0], expected_states[0, 3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(committed[1], expected_states[1, 3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_committed, committed, rtol=1e-5, atol=1e-5)

    state_pages = torch.zeros(
        (initial_state.shape[0], *initial_state.shape[1:]), dtype=initial_state.dtype
    )
    state_pages.copy_(initial_state)
    context = TreeWYLayerContext(
        key=key,
        pseudo_values=tensor_result.pseudo_values,
        log_decay=tensor_result.log_decay,
        tree_parents=parents,
        tree_num_nodes=tree_num_nodes,
        row_widths=widths,
        sequence_indices=torch.arange(rows),
        root_state_indices=torch.arange(rows),
        initial_state=initial_state,
        state=state_pages,
    )
    treewy_commit_contexts_torch({"layer": context}, path_node_ids, num_sampled)
    torch.testing.assert_close(state_pages, committed, rtol=1e-5, atol=1e-5)


def test_treewy_verify_masks_large_non_ancestor_decay_ratios():
    """A strongly decayed sibling must not poison another branch's WY solve."""
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    g[1].fill_(-100.0)
    scale = key.shape[-1] ** -0.5

    result = treewy_verify_torch(query, key, value, g, beta, initial_state, parents, widths, scale, tree_num_nodes)
    expected_output, expected_pseudo, _ = _run_token_recurrence(
        query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes, scale
    )

    assert torch.isfinite(result.output).all()
    assert torch.isfinite(result.pseudo_values).all()
    torch.testing.assert_close(result.output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result.pseudo_values, expected_pseudo, rtol=1e-5, atol=1e-5)


def test_treewy_verify_reuses_prepared_ancestor_mask_for_layers(monkeypatch):
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    prepared = treewy_module.treewy_ancestor_mask(parents, tree_num_nodes)
    calls = 0
    original = treewy_module._tree_parent_and_ancestor_tensors

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(treewy_module, "_tree_parent_and_ancestor_tensors", counted)
    first = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
        ancestors=prepared,
    )
    second = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
        ancestors=prepared,
    )

    assert calls == 0
    assert first.ancestors is second.ancestors is prepared
    torch.testing.assert_close(first.output, second.output, rtol=0, atol=0)
    torch.testing.assert_close(first.pseudo_values, second.pseudo_values, rtol=0, atol=0)


@pytest.mark.parametrize("slots", [1, 3, 17, 33, 65])
@pytest.mark.parametrize("topology", ["chain", "branch"])
@pytest.mark.parametrize("per_value_beta", [False, True])
def test_treewy_batched_solve_matches_recurrence(slots, topology, per_value_beta):
    """Direct solves handle non-power-of-two widths, mixed rows and grouped heads."""
    torch.manual_seed(slots)
    widths = torch.tensor([slots, max(1, slots - 2), 0], dtype=torch.int32)
    token_count = int(widths.sum())
    query = torch.nn.functional.normalize(torch.randn(token_count, 2, 5), dim=-1)
    key = torch.nn.functional.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(token_count, 4, 7)
    g = -torch.rand(token_count, 4) * 2.0
    beta = torch.rand_like(value) if per_value_beta else torch.rand(token_count, 4)
    initial_state = torch.randn(3, 4, 7, 5)
    parent_row = torch.arange(slots - 1, dtype=torch.int32)
    if topology == "branch":
        parent_row = parent_row // 2
    parents = parent_row.repeat(3, 1)
    tree_num_nodes = torch.tensor([slots - 1, 0, 0], dtype=torch.int32)
    scale = key.shape[-1] ** -0.5
    result = treewy_verify_torch(query, key, value, g, beta, initial_state, parents, widths, scale, tree_num_nodes)
    expected_output, expected_pseudo, expected_states = _run_token_recurrence(
        query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes, scale
    )
    torch.testing.assert_close(result.output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result.pseudo_values, expected_pseudo, rtol=1e-5, atol=1e-5)

    path = []
    node = slots - 1
    while node > 0:
        path.append(node)
        node = int(parent_row[node - 1])
    path.reverse()
    path_node_ids = torch.full((3, slots - 1), -1, dtype=torch.long)
    path_node_ids[0, : len(path)] = torch.tensor(path, dtype=torch.long)
    num_sampled = torch.tensor([len(path) + 1, 0, 0], dtype=torch.int32)
    committed = treewy_commit_torch(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
        ancestors=result.ancestors,
    )
    torch.testing.assert_close(committed[0], expected_states[0, slots - 1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(committed[1], expected_states[1, int(widths[1]) - 1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(committed[2], initial_state[2], rtol=0, atol=0)


@pytest.mark.parametrize(
    "gates",
    [(-1.0, -100.0, -1.0), (-5.0,) * 21],
    ids=["large_single_gate", "accumulated_moderate_gates"],
)
def test_treewy_commit_rejected_descendants_do_not_poison_next_step(gates):
    """Root-only acceptance must ignore strongly decayed unaccepted descendants."""
    slots = len(gates)
    query = torch.ones(slots, 1, 1)
    key = torch.ones_like(query)
    value = torch.ones_like(query)
    g = torch.tensor(gates, dtype=torch.float32).reshape(slots, 1)
    beta = torch.full((slots, 1), 0.5)
    initial_state = torch.ones(1, 1, 1, 1)
    parents = torch.arange(slots - 1, dtype=torch.int32).unsqueeze(0)
    widths = torch.tensor([slots], dtype=torch.int32)
    tree_num_nodes = torch.tensor([slots - 1], dtype=torch.int32)
    scale = 1.0
    reference = treewy_reference_verify(
        query, key, value, g, beta, initial_state, parents, widths, scale, tree_num_nodes
    )
    _, _, expected_states = _run_token_recurrence(
        query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes, scale
    )
    committed = treewy_commit_torch(
        initial_state,
        key,
        reference.pseudo_values,
        reference.log_decay,
        parents,
        widths,
        torch.full((1, slots - 1), -1, dtype=torch.long),
        torch.ones(1, dtype=torch.int32),
        tree_num_nodes,
    )
    expected_state = expected_states[:, 0]
    assert torch.isfinite(committed).all()
    torch.testing.assert_close(committed, expected_state, rtol=1e-5, atol=1e-5)

    # Even a subsequent small-gate step stays corrupted if commit wrote NaN.
    g.fill_(-0.1)
    result = treewy_verify_torch(query, key, value, g, beta, committed, parents, widths, scale, tree_num_nodes)
    expected_output, _, _ = _run_token_recurrence(
        query, key, value, g, beta, expected_state, parents, widths, tree_num_nodes, scale
    )
    assert torch.isfinite(result.output).all()
    torch.testing.assert_close(result.output, expected_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_treewy_batched_updates_match_recurrence_over_multiple_steps(dtype):
    """Changed trees and accepted branches must not accumulate state drift."""
    torch.manual_seed(29)
    rows, slots = 2, 17
    widths = torch.full((rows,), slots, dtype=torch.int32)
    tree_num_nodes = torch.full((rows,), slots - 1, dtype=torch.int32)
    actual_state = torch.randn(rows, 4, 11, 7)
    expected_state = actual_state.clone()
    for step in range(12):
        parents = torch.stack([torch.randint(node, (rows,)) for node in range(1, slots)], dim=1)
        query = torch.nn.functional.normalize(torch.randn(rows * slots, 2, 7), dim=-1).to(dtype)
        key = torch.nn.functional.normalize(torch.randn_like(query.float()), dim=-1).to(dtype)
        value = torch.randn(rows * slots, 4, 11).to(dtype)
        g = -torch.rand(rows * slots, 4) * 0.2
        beta = torch.rand(rows * slots, 4).to(dtype)
        scale = key.shape[-1] ** -0.5
        result = treewy_verify_torch(query, key, value, g, beta, actual_state, parents, widths, scale, tree_num_nodes)
        expected_output, _, expected_states = _run_token_recurrence(
            query.float(),
            key.float(),
            value.float(),
            g,
            beta.float(),
            expected_state,
            parents,
            widths,
            tree_num_nodes,
            scale,
        )
        tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(result.output.float(), expected_output, rtol=tolerance, atol=tolerance)
        path_node_ids = torch.full((rows, slots - 1), -1, dtype=torch.long)
        num_sampled = torch.ones(rows, dtype=torch.int32)
        for row in range(rows):
            leaf = (step + row * (slots - 1)) % slots
            node, path = leaf, []
            while node > 0:
                path.append(node)
                node = int(parents[row, node - 1])
            path.reverse()
            path_node_ids[row, : len(path)] = torch.tensor(path, dtype=torch.long)
            num_sampled[row] = len(path) + 1
            expected_state[row] = expected_states[row, leaf]
        actual_state = treewy_commit_torch(
            actual_state,
            key,
            result.pseudo_values,
            result.log_decay,
            parents,
            widths,
            path_node_ids,
            num_sampled,
            tree_num_nodes,
        )
        assert torch.isfinite(actual_state).all()
        torch.testing.assert_close(actual_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("slots", [17, 33, 65])
def test_treewy_solve_stays_stable_for_identical_keys(slots):
    """A full delta update with identical unit keys must return each new value."""
    # A finite Neumann series would be numerically unsafe here.
    torch.manual_seed(41)
    query = torch.ones(slots, 1, 1)
    key = torch.ones_like(query)
    value = torch.randn(slots, 1, 3)
    g = torch.zeros(slots, 1)
    beta = torch.ones(slots, 1)
    initial_state = torch.randn(1, 1, 3, 1)
    parents = torch.arange(slots - 1, dtype=torch.int32).unsqueeze(0)
    widths = torch.tensor([slots], dtype=torch.int32)
    tree_num_nodes = torch.tensor([slots - 1], dtype=torch.int32)
    result = treewy_verify_torch(query, key, value, g, beta, initial_state, parents, widths, 1.0, tree_num_nodes)
    torch.testing.assert_close(result.output, value, rtol=1e-5, atol=1e-5)
    committed = treewy_commit_torch(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        torch.arange(1, slots).unsqueeze(0),
        widths,
        tree_num_nodes,
    )
    torch.testing.assert_close(committed[0, :, :, 0], value[-1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("slots", [33, 65, 129])
def test_treewy_solve_stays_stable_for_nearly_collinear_keys(slots):
    """Small decay and almost-full updates should agree with token recurrence."""
    torch.manual_seed(47)
    key = torch.nn.functional.normalize(torch.ones(slots, 2, 16) + 0.01 * torch.randn(slots, 2, 16), dim=-1)
    query = key.clone()
    value = torch.randn(slots, 4, 7)
    g = -torch.rand(slots, 4) * 1e-4
    beta = 1.0 - torch.rand(slots, 4) * 0.01
    initial_state = torch.randn(1, 4, 7, 16)
    parents = torch.arange(slots - 1, dtype=torch.int32).unsqueeze(0)
    widths = torch.tensor([slots], dtype=torch.int32)
    tree_num_nodes = torch.tensor([slots - 1], dtype=torch.int32)
    result = treewy_verify_torch(query, key, value, g, beta, initial_state, parents, widths, 1.0, tree_num_nodes)
    expected_output, expected_pseudo, expected_states = _run_token_recurrence(
        query.double(),
        key.double(),
        value.double(),
        g.double(),
        beta.double(),
        initial_state.double(),
        parents,
        widths,
        tree_num_nodes,
        1.0,
    )
    # A long chain sums many nearly cancelling terms: allow FP32 reduction error
    # against a double-precision oracle.
    torch.testing.assert_close(result.output.double(), expected_output, rtol=1e-5, atol=3e-5)
    torch.testing.assert_close(result.pseudo_values.double(), expected_pseudo, rtol=1e-5, atol=1e-5)
    committed = treewy_commit_torch(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        torch.arange(1, slots).unsqueeze(0),
        widths,
        tree_num_nodes,
    )
    torch.testing.assert_close(committed.double(), expected_states[:, -1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_treewy_long_running_commits_match_fp64_recurrence(dtype):
    """Exercise persistent state with weak decay and changing accepted branches."""
    torch.manual_seed(53)
    rows, slots = 2, 17
    widths = torch.full((rows,), slots, dtype=torch.int32)
    tree_num_nodes = widths - 1
    state = torch.randn(rows, 4, 11, 16)
    expected_state = state.double()
    for step in range(64):
        parents = torch.stack([torch.randint(node, (rows,)) for node in range(1, slots)], dim=1)
        key = torch.nn.functional.normalize(torch.randn(rows * slots, 2, 16), dim=-1).to(dtype)
        query = torch.nn.functional.normalize(torch.randn_like(key.float()), dim=-1).to(dtype)
        value = torch.randn(rows * slots, 4, 11).to(dtype)
        g = -torch.rand(rows * slots, 4) * 1e-4
        beta = torch.rand(rows * slots, 4).to(dtype)
        result = treewy_verify_torch(query, key, value, g, beta, state, parents, widths, 0.25, tree_num_nodes)
        expected_output, expected_pseudo, expected_states = _run_token_recurrence(
            query.double(),
            key.double(),
            value.double(),
            g.double(),
            beta.double(),
            expected_state,
            parents,
            widths,
            tree_num_nodes,
            0.25,
        )
        tolerance = 5e-3 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(result.output.double(), expected_output, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(result.pseudo_values.double(), expected_pseudo, rtol=2e-5, atol=2e-5)
        paths = torch.full((rows, slots - 1), -1, dtype=torch.long)
        sampled = torch.ones(rows, dtype=torch.int32)
        for row in range(rows):
            leaf = (step * 7 + row) % slots
            path, node = [], leaf
            while node > 0:
                path.append(node)
                node = int(parents[row, node - 1])
            path.reverse()
            paths[row, : len(path)] = torch.tensor(path, dtype=torch.long)
            sampled[row] += len(path)
            expected_state[row] = expected_states[row, leaf]
        state = treewy_commit_torch(
            state,
            key,
            result.pseudo_values,
            result.log_decay,
            parents,
            widths,
            paths,
            sampled,
            tree_num_nodes,
            ancestors=result.ancestors,
        )
        torch.testing.assert_close(state.double(), expected_state, rtol=2e-5, atol=2e-5)


def test_treewy_packing_skips_empty_rows_and_preserves_strided_tokens():
    torch.manual_seed(59)
    widths = torch.tensor([0, 3, 0, 1, 0], dtype=torch.int32)
    parents = torch.tensor([[0, 0]] * 5, dtype=torch.int32)
    tree_num_nodes = torch.tensor([0, 2, 0, 0, 0], dtype=torch.int32)
    query = torch.randn(8, 2, 4)[::2]
    key = torch.nn.functional.normalize(torch.randn(8, 2, 4), dim=-1)[::2]
    value = torch.randn(8, 4, 7)[::2]
    g = -torch.rand(8, 4)[::2]
    beta = torch.rand(8, 4)[::2]
    initial_state = torch.randn(5, 4, 7, 4)
    result = treewy_verify_torch(query, key, value, g, beta, initial_state, parents, widths, 0.5, tree_num_nodes)
    expected_output, expected_pseudo, _ = _run_token_recurrence(
        query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes, 0.5
    )
    torch.testing.assert_close(result.output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result.pseudo_values, expected_pseudo, rtol=1e-5, atol=1e-5)


def test_treewy_single_padded_row_keeps_dynamic_width_mask():
    query, key, value, g, beta, initial_state, parents, _, tree_num_nodes = _reference_inputs()
    slots = parents.shape[1] + 1
    query, key, value, g, beta = (x[:slots] for x in (query, key, value, g, beta))
    parents, tree_num_nodes, initial_state = parents[:1], tree_num_nodes[:1], initial_state[:1]
    ancestors = treewy_module.treewy_ancestor_mask(parents, tree_num_nodes)
    width = torch.tensor([slots], dtype=torch.int32)
    full = treewy_verify_torch(
        query, key, value, g, beta, initial_state, parents, width, 0.5, tree_num_nodes, ancestors=ancestors
    )
    width.fill_(2)
    short = treewy_verify_torch(
        query, key, value, g, beta, initial_state, parents, width, 0.5, tree_num_nodes, ancestors=ancestors
    )
    expected_output, expected_pseudo, _ = _run_token_recurrence(
        query[:2], key[:2], value[:2], g[:2], beta[:2], initial_state, parents, width, tree_num_nodes, 0.5
    )
    torch.testing.assert_close(short.output[:2], expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(short.pseudo_values[:2], expected_pseudo, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(full.output[:2], short.output[:2], rtol=0, atol=0)
    assert torch.count_nonzero(short.output[2:]) == 0
    assert torch.count_nonzero(short.pseudo_values[2:]) == 0


def test_treewy_reference_commit_keeps_tree_root_and_commits_folded_prompt_chunk():
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    result = treewy_reference_verify(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
    )
    tensor_result = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
    )
    path_node_ids = torch.full((2, 4), -1, dtype=torch.int64)
    num_sampled = torch.tensor([1, 0], dtype=torch.int32)

    committed = treewy_reference_commit(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )
    tensor_committed = treewy_commit_torch(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )

    expected_output, _, expected_states = _run_token_recurrence(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        tree_num_nodes,
        key.shape[-1] ** -0.5,
    )
    del expected_output
    torch.testing.assert_close(committed[0], expected_states[0, 0], rtol=1e-5, atol=1e-5)
    # Row 1 has no tree nodes: a linear prompt chunk folded into the verify
    # batch, whose full recurrent state still has to be committed.
    torch.testing.assert_close(committed[1], expected_states[1, widths[1] - 1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_result.output, result.output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_committed, committed, rtol=1e-5, atol=1e-5)


def test_treewy_commit_keeps_zero_width_padded_rows_unchanged():
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    slots = parents.shape[1] + 1
    row_widths = torch.tensor([slots, 0], dtype=torch.int32)
    result = treewy_reference_verify(
        query[:slots],
        key[:slots],
        value[:slots],
        g[:slots],
        beta[:slots],
        initial_state,
        parents,
        row_widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
    )
    tensor_result = treewy_verify_torch(
        query[:slots],
        key[:slots],
        value[:slots],
        g[:slots],
        beta[:slots],
        initial_state,
        parents,
        row_widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
    )
    path_node_ids = torch.full((2, 4), -1, dtype=torch.int64)
    num_sampled = torch.tensor([1, 0], dtype=torch.int32)

    expected = treewy_reference_commit(
        initial_state,
        key[:slots],
        result.pseudo_values,
        result.log_decay,
        parents,
        row_widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )
    actual = treewy_commit_torch(
        initial_state,
        key[:slots],
        tensor_result.pseudo_values,
        tensor_result.log_decay,
        parents,
        row_widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )

    torch.testing.assert_close(expected[1], initial_state[1])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_treewy_commit_contexts_writes_folded_prompt_chunks():
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    result = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
    )
    state_pages = initial_state.clone()
    context = TreeWYLayerContext(
        key=key,
        pseudo_values=result.pseudo_values,
        log_decay=result.log_decay,
        tree_parents=parents,
        tree_num_nodes=tree_num_nodes,
        row_widths=widths,
        sequence_indices=torch.arange(initial_state.shape[0]),
        root_state_indices=torch.arange(initial_state.shape[0]),
        initial_state=initial_state,
        state=state_pages,
    )
    path_node_ids = torch.full((2, 4), -1, dtype=torch.int64)
    num_sampled = torch.tensor([1, 0], dtype=torch.int32)

    treewy_commit_contexts_torch({"layer": context}, path_node_ids, num_sampled)

    expected = treewy_reference_commit(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )
    torch.testing.assert_close(state_pages, expected, rtol=1e-5, atol=1e-5)


def test_treewy_commit_contexts_handles_full_graph_padding():
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    slots = parents.shape[1] + 1
    # The padded row aliases page 0, a safe page a real request also uses, so its
    # initial state must match the aliased page.
    padded_initial_state = torch.cat((initial_state, initial_state[:1]))
    padded_parents = torch.cat((parents, parents[:1]))
    padded_widths = torch.tensor([slots, slots, slots], dtype=torch.int32)
    padded_tree_num_nodes = torch.tensor([slots - 1, slots - 1, 0], dtype=torch.int32)
    padded_query = torch.cat((query, query[:slots]))
    padded_key = torch.cat((key, key[:slots]))
    padded_value = torch.cat((value, value[:slots]))
    padded_g = torch.cat((g, g[:slots]))
    padded_beta = torch.cat((beta, beta[:slots]))
    result = treewy_verify_torch(
        padded_query,
        padded_key,
        padded_value,
        padded_g,
        padded_beta,
        padded_initial_state,
        padded_parents,
        padded_widths,
        key.shape[-1] ** -0.5,
        padded_tree_num_nodes,
    )
    state_pages = initial_state.clone()
    context = TreeWYLayerContext(
        key=padded_key,
        pseudo_values=result.pseudo_values,
        log_decay=result.log_decay,
        tree_parents=padded_parents,
        tree_num_nodes=padded_tree_num_nodes,
        row_widths=padded_widths,
        sequence_indices=torch.arange(3),
        root_state_indices=torch.tensor([0, 1, 0]),
        initial_state=padded_initial_state,
        state=state_pages,
        ancestors=result.ancestors,
    )
    # Sampler outputs cover the two real requests; the context has a padded third.
    path_node_ids = torch.tensor([[1, -1, -1, -1], [1, 2, -1, -1]])
    num_sampled = torch.tensor([2, 3], dtype=torch.int32)

    treewy_commit_contexts_torch({"layer": context}, path_node_ids, num_sampled)

    expected_real_rows = treewy_commit_torch(
        padded_initial_state,
        padded_key,
        result.pseudo_values,
        result.log_decay,
        padded_parents,
        padded_widths,
        torch.cat((path_node_ids, torch.full((1, 4), -1, dtype=torch.long))),
        torch.tensor([2, 3, 0], dtype=torch.int32),
        padded_tree_num_nodes,
    )[:2]
    torch.testing.assert_close(state_pages, expected_real_rows, rtol=1e-5, atol=1e-5)


def _expected_cumulative_decay(g, row_widths, parents, tree_num_nodes):
    slots = parents.shape[1] + 1
    result = torch.zeros((row_widths.numel(), slots, g.shape[1]), dtype=torch.float32)
    starts = torch.cumsum(
        torch.cat((row_widths.new_zeros(1), row_widths[:-1])), dim=0
    ).tolist()
    for row in range(row_widths.numel()):
        for node in range(int(row_widths[row])):
            token = starts[row] + node
            if node == 0:
                result[row, node] = g[token]
            else:
                parent = (
                    node - 1
                    if tree_num_nodes[row] == 0
                    else int(parents[row, node - 1])
                )
                result[row, node] = result[row, parent] + g[token]
    return result


def test_treewy_reference_supports_truncated_rows_and_per_value_beta():
    inputs = _reference_inputs()
    query, key, value, g, _, initial_state, parents, _, _ = inputs
    row_widths = torch.tensor([4, 2], dtype=torch.int32)
    tree_num_nodes = torch.tensor([4, 0], dtype=torch.int32)
    slots = parents.shape[1] + 1
    query = torch.cat((query[:4], query[slots : slots + 2]))
    key = torch.cat((key[:4], key[slots : slots + 2]))
    value = torch.cat((value[:4], value[slots : slots + 2]))
    g = torch.cat((g[:4], g[slots : slots + 2]))
    beta = torch.rand_like(value)
    scale = key.shape[-1] ** -0.5

    result = treewy_reference_verify(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        row_widths,
        scale,
        tree_num_nodes,
    )
    tensor_result = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        row_widths,
        scale,
        tree_num_nodes,
    )
    expected_output, expected_pseudo, expected_states = _run_token_recurrence(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        row_widths,
        tree_num_nodes,
        scale,
    )
    torch.testing.assert_close(result.output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(result.pseudo_values, expected_pseudo, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_result.output, result.output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_result.pseudo_values, result.pseudo_values, rtol=1e-5, atol=1e-5)

    # The tree row is truncated before node 4 and accepts root -> 1 -> 3; row 1 is
    # a two-token chain packed right after it.
    path_node_ids = torch.tensor([[1, 3, -1, -1], [-1, -1, -1, -1]])
    num_sampled = torch.tensor([3, 2], dtype=torch.int32)
    committed = treewy_reference_commit(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        row_widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )
    tensor_committed = treewy_commit_torch(
        initial_state,
        key,
        result.pseudo_values,
        result.log_decay,
        parents,
        row_widths,
        path_node_ids,
        num_sampled,
        tree_num_nodes,
    )
    torch.testing.assert_close(committed[0], expected_states[0, 3], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(committed[1], expected_states[1, 1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(tensor_committed, committed, rtol=1e-5, atol=1e-5)


def test_treewy_reference_commit_rejects_non_ancestral_accepted_path():
    inputs = _reference_inputs()
    query, key, value, g, beta, initial_state, parents, widths, tree_num_nodes = inputs
    result = treewy_reference_verify(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        widths,
        key.shape[-1] ** -0.5,
        tree_num_nodes,
    )

    with pytest.raises(ValueError, match="not a root-to-leaf path"):
        treewy_reference_commit(
            initial_state,
            key,
            result.pseudo_values,
            result.log_decay,
            parents,
            widths,
            torch.tensor([[1, 2, -1, -1], [-1, -1, -1, -1]]),
            torch.tensor([3, 0], dtype=torch.int32),
            tree_num_nodes,
        )
