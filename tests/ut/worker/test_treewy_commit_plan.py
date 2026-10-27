# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TreeWY accepted-state commit: one shared plan, static shapes, page-safe write-back."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_ascend.worker.v2.spec_decode.tree.treewy as treewy_module
from vllm_ascend.worker.v2.spec_decode.tree.treewy import (
    TreeWYLayerContext,
    treewy_ancestor_mask,
    treewy_commit_contexts_torch,
    treewy_commit_torch,
    treewy_verify_torch,
)

_ROWS = 2
_SLOTS = 5
_KEY_HEADS = 2
_VALUE_HEADS = 4
_DIM = 3
_PATH_WIDTH = 4


def _topology():
    """Row 0 branches at nodes 1 and 2; row 1 is the linear-chain case."""
    parents = torch.tensor([[0, 0, 1, 0], [0, 1, 2, 3]], dtype=torch.int32)
    row_widths = torch.tensor([_SLOTS, _SLOTS], dtype=torch.int32)
    tree_num_nodes = torch.tensor([_SLOTS - 1, 0], dtype=torch.int32)
    return parents, row_widths, tree_num_nodes


def _layer(
    seed: int,
    initial_state: torch.Tensor,
    pages: torch.Tensor,
    topology=None,
    ancestors=None,
    sequence_indices=None,
):
    """Verify one layer and keep what the sampler commit needs.

    The metadata builder shares the tree tables, the row-to-request map and the ancestor mask
    across layers, so the multi-layer tests pass the same objects in.
    """
    torch.manual_seed(seed)
    parents, row_widths, tree_num_nodes = topology or _topology()
    if ancestors is None:
        ancestors = treewy_ancestor_mask(parents, tree_num_nodes)
    if sequence_indices is None:
        sequence_indices = torch.arange(_ROWS)
    tokens = _ROWS * _SLOTS
    query = torch.randn(tokens, _KEY_HEADS, _DIM)
    key = torch.randn_like(query)
    value = torch.randn(tokens, _VALUE_HEADS, _DIM)
    g = -torch.rand(tokens, _VALUE_HEADS) * 0.2
    beta = torch.rand(tokens, _VALUE_HEADS)
    result = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        row_widths,
        _DIM**-0.5,
        tree_num_nodes,
    )
    context = TreeWYLayerContext(
        key=key,
        pseudo_values=result.pseudo_values,
        log_decay=result.log_decay,
        tree_parents=parents,
        tree_num_nodes=tree_num_nodes,
        row_widths=row_widths,
        sequence_indices=sequence_indices,
        root_state_indices=torch.arange(_ROWS),
        initial_state=initial_state,
        state=pages,
        ancestors=ancestors,
    )
    return context, key, result


def _layers(count: int = 3):
    topology = _topology()
    ancestors = treewy_ancestor_mask(topology[0], topology[2])
    sequence_indices = torch.arange(_ROWS)
    contexts = {}
    expected = {}
    for index in range(count):
        torch.manual_seed(100 + index)
        initial_state = torch.randn(_ROWS, _VALUE_HEADS, _DIM, _DIM)
        pages = initial_state.clone()
        context, key, result = _layer(
            index, initial_state, pages, topology, ancestors, sequence_indices
        )
        contexts[f"layer{index}"] = context
        # The pages are written per layer, so each layer needs its own reference.
        expected[f"layer{index}"] = (
            treewy_commit_torch(
                initial_state,
                key,
                result.pseudo_values,
                result.log_decay,
                context.tree_parents,
                context.row_widths,
                _PATH,
                _NUM_SAMPLED,
                context.tree_num_nodes,
                ancestors=ancestors,
            ),
            pages,
        )
    return contexts, expected


_PATH = torch.tensor([[1, 3, -1, -1], [-1, -1, -1, -1]])
_NUM_SAMPLED = torch.tensor([3, 4], dtype=torch.int32)


def test_one_plan_serves_every_layer(monkeypatch):
    """The topology/accepted-path work is per batch, not per (of 48) GDN layers."""
    contexts, _ = _layers(count=3)
    builds = 0
    original = treewy_module.treewy_commit_plan

    def counting_plan(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(treewy_module, "treewy_commit_plan", counting_plan)

    treewy_commit_contexts_torch(contexts, _PATH, _NUM_SAMPLED)

    assert builds == 1, (
        "layers of one batch share the draft tree and the accepted path, so the commit plan "
        f"must be built once; it was built {builds} times for {len(contexts)} layers"
    )


def test_every_layer_writes_its_own_accepted_state():
    contexts, expected = _layers(count=3)

    treewy_commit_contexts_torch(contexts, _PATH, _NUM_SAMPLED)

    for name, (wanted, pages) in expected.items():
        torch.testing.assert_close(pages, wanted, rtol=1e-5, atol=1e-5)


def test_a_padded_chain_row_never_commits_into_an_aliased_page():
    """Row 1 is a padded graph row the builder marked linear; the chain rule would commit it."""
    parents, row_widths, _ = _topology()
    # Row 0 is a tree row that accepted nothing, row 1 a padded linear row.
    padded_num_nodes = torch.tensor([_SLOTS - 1, 0], dtype=torch.int32)
    torch.manual_seed(3)
    pages = torch.randn(2, _VALUE_HEADS, _DIM, _DIM)
    aliased = torch.tensor([0, 0])
    # Both rows address page 0, so the initial state of both is that page's content.
    initial_state = pages[aliased].clone()
    tokens = _ROWS * _SLOTS
    query = torch.randn(tokens, _KEY_HEADS, _DIM)
    key = torch.randn_like(query)
    value = torch.randn(tokens, _VALUE_HEADS, _DIM)
    g = -torch.rand(tokens, _VALUE_HEADS) * 0.2
    beta = torch.rand(tokens, _VALUE_HEADS)
    result = treewy_verify_torch(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        parents,
        row_widths,
        _DIM**-0.5,
        padded_num_nodes,
    )
    context = TreeWYLayerContext(
        key=key,
        pseudo_values=result.pseudo_values,
        log_decay=result.log_decay,
        tree_parents=parents,
        tree_num_nodes=padded_num_nodes,
        row_widths=row_widths,
        sequence_indices=torch.arange(_ROWS),
        root_state_indices=aliased,
        initial_state=initial_state,
        state=pages,
        ancestors=result.ancestors,
    )
    before = pages.clone()

    # One real request in the sampler output, so row 1 is a padded row.
    treewy_commit_contexts_torch(
        {"layer": context},
        torch.tensor([[1, -1, -1, -1]]),
        torch.tensor([0], dtype=torch.int32),
    )

    assert torch.equal(pages, before), (
        "no row accepted a path, so the shared page must keep the committed state; the padded "
        f"linear row committed {pages[0].flatten()[:3].tolist()} over "
        f"{before[0].flatten()[:3].tolist()}"
    )


def test_an_accepted_row_wins_its_page_whatever_the_row_order():
    """A skipped row aliasing a committed page must not overwrite it with the old content."""
    # Both rows are tree rows, so the second one is skipped rather than carried as a chain.
    parents = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=torch.int32)
    tree_num_nodes = torch.tensor([_SLOTS - 1, _SLOTS - 1], dtype=torch.int32)
    row_widths = torch.tensor([_SLOTS, _SLOTS], dtype=torch.int32)
    topology = (parents, row_widths, tree_num_nodes)
    torch.manual_seed(7)
    initial_state = torch.randn(2, _VALUE_HEADS, _DIM, _DIM)
    pages = initial_state.clone()
    context, key, result = _layer(7, initial_state, pages, topology=topology)
    # Both rows address page 0 and only row 0 commits; with a write order dependent scatter,
    # row 1's unchanged state would win half of the time.
    object.__setattr__(context, "root_state_indices", torch.tensor([0, 0]))
    object.__setattr__(context, "sequence_indices", torch.tensor([0, -1]))

    treewy_commit_contexts_torch({"layer": context}, _PATH, _NUM_SAMPLED)

    wanted = treewy_commit_torch(
        initial_state[:1],
        key[: _SLOTS],
        result.pseudo_values[: _SLOTS],
        result.log_decay[:1],
        parents[:1],
        row_widths[:1],
        _PATH[:1],
        _NUM_SAMPLED[:1],
        tree_num_nodes[:1],
        ancestors=result.ancestors[:1],
    )
    torch.testing.assert_close(pages[0], wanted[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pages[1], initial_state[1], rtol=1e-5, atol=1e-5)


class _GraphBlockerCounter(TorchDispatchMode):
    """Counts what makes an op un-capturable: host reads and data-dependent shapes."""

    def __init__(self):
        super().__init__()
        self.host_reads = 0
        self.dynamic_shapes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name()
        if name == "_local_scalar_dense":
            self.host_reads += 1
        if name in {"nonzero", "masked_select", "unique", "unique_consecutive"}:
            self.dynamic_shapes += 1
        if name.split(".")[0] == "index" and len(args) > 1 and isinstance(args[1], (list, tuple)):
            bool_indices = [
                index for index in args[1] if isinstance(index, torch.Tensor) and index.dtype == torch.bool
            ]
            if bool_indices:
                # Boolean advanced indexing: the result length is a device value.
                self.dynamic_shapes += 1
        return func(*args, **(kwargs or {}))


def test_commit_has_no_host_read_and_no_data_dependent_shape():
    contexts, _ = _layers(count=3)

    with _GraphBlockerCounter() as counter:
        treewy_commit_contexts_torch(contexts, _PATH, _NUM_SAMPLED)

    assert counter.host_reads == 0, (
        "a host read inside the commit forces a device sync and cannot be captured; the "
        f"commit made {counter.host_reads} of them"
    )
    assert counter.dynamic_shapes == 0, (
        "the commit must write every page through a static-shape scatter; "
        f"{counter.dynamic_shapes} data-dependent shapes were produced"
    )


@pytest.fixture
def fake_npu(monkeypatch):
    """Enough of ``torch.npu`` for the model state's capture gate."""

    class _Graph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            NPUGraph=_Graph,
            is_current_stream_capturing=lambda: False,
        ),
        raising=False,
    )
    return _Graph


def _model_state():
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridModelState,
    )

    state = object.__new__(AscendMambaHybridModelState)
    state._treewy_commit_token = None
    state._treewy_commit_graphs = {}
    state._treewy_commit_graph_failures = set()
    state._treewy_commit_buffers = {}
    return state


def _captured_entry(graph, contexts, plan, path_buffer, sampled_buffer):
    first = next(iter(contexts.values()))
    fingerprint = (len(contexts), id(first.key), id(first.pseudo_values), id(first.initial_state))
    return (graph, fingerprint, plan, path_buffer, sampled_buffer)


def test_commit_graph_needs_a_captured_layer_set(fake_npu):
    contexts, _ = _layers(count=2)
    state = _model_state()

    # No token: the verification ran eagerly, so a replay would read addresses nothing refreshes.
    assert not state._run_treewy_commit_graph(contexts, _PATH, _NUM_SAMPLED)

    # A token for another layer set (or a recaptured one) must not be reused either.
    state._treewy_commit_token = ({}, 1)
    assert not state._run_treewy_commit_graph(contexts, _PATH, _NUM_SAMPLED)

    # The sampler output must be a device tensor; an int cannot be refreshed into a buffer.
    state._treewy_commit_token = (contexts, 1)
    assert not state._run_treewy_commit_graph(contexts, _PATH, 1)


def test_commit_graph_replays_a_captured_layer_set(fake_npu):
    contexts, _ = _layers(count=2)
    state = _model_state()
    graph = fake_npu()
    plan = treewy_module.treewy_commit_plan(
        next(iter(contexts.values())).tree_parents,
        next(iter(contexts.values())).tree_num_nodes,
        next(iter(contexts.values())).row_widths,
        _NUM_SAMPLED,
        _PATH,
        _VALUE_HEADS,
        _KEY_HEADS,
        _ROWS * _SLOTS,
        ancestors=next(iter(contexts.values())).ancestors,
    )
    path_buffer, sampled_buffer = state._treewy_commit_inputs(_ROWS, _PATH_WIDTH, _PATH.device)
    entry = _captured_entry(graph, contexts, plan, path_buffer, sampled_buffer)
    state._treewy_commit_graphs[(id(contexts), 1, _ROWS, _PATH_WIDTH)] = entry
    state._treewy_commit_token = (contexts, 1)

    assert state._run_treewy_commit_graph(contexts, _PATH, _NUM_SAMPLED)
    assert graph.replays == 1

    # The sampler output is refreshed into the buffers the recorded graph reads.
    torch.testing.assert_close(path_buffer, _PATH)
    torch.testing.assert_close(sampled_buffer, _NUM_SAMPLED)


def test_commit_graph_is_dropped_when_an_eager_forward_replaced_the_layers(fake_npu):
    contexts, _ = _layers(count=2)
    state = _model_state()
    graph = fake_npu()
    first = next(iter(contexts.values()))
    plan = treewy_module.treewy_commit_plan(
        first.tree_parents,
        first.tree_num_nodes,
        first.row_widths,
        _NUM_SAMPLED,
        _PATH,
        _VALUE_HEADS,
        _KEY_HEADS,
        _ROWS * _SLOTS,
        ancestors=first.ancestors,
    )
    path_buffer, sampled_buffer = state._treewy_commit_inputs(_ROWS, _PATH_WIDTH, _PATH.device)
    key = (id(contexts), 1, _ROWS, _PATH_WIDTH)
    entry = _captured_entry(graph, contexts, plan, path_buffer, sampled_buffer)
    state._treewy_commit_graphs[key] = entry
    state._treewy_commit_token = (contexts, 1)

    # An eager forward rebuilt a layer context: its tensors no longer live at the recorded
    # addresses, so replaying the commit graph would read stale bytes.
    an_eager_context, _, _ = _layer(9, first.initial_state.clone(), first.state.clone())
    contexts["layer0"] = an_eager_context

    assert not state._run_treewy_commit_graph(contexts, _PATH, _NUM_SAMPLED)
    assert graph.replays == 0
    assert key not in state._treewy_commit_graphs
    assert key in state._treewy_commit_graph_failures


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_bookkeeping_writes_request_slots_without_touching_inactive_slots(index_dtype):
    state = _model_state()
    state.num_accepted_tokens_gpu = torch.full((5,), 9, dtype=torch.int32)
    prev_path = torch.full((5, 6), -7, dtype=torch.int32)
    prev_sampled = torch.full((5,), 9, dtype=torch.int32)
    idx = torch.tensor([3, 1], dtype=index_dtype)
    mask = torch.tensor([True, False])

    with _GraphBlockerCounter() as counter:
        state._commit_treewy_bookkeeping(idx, _PATH.to(torch.int32), _NUM_SAMPLED, mask, prev_path, prev_sampled)

    wanted_path = torch.full_like(prev_path, -7)
    wanted_path[idx, :_PATH_WIDTH] = _PATH.to(torch.int32)
    torch.testing.assert_close(prev_path, wanted_path)
    torch.testing.assert_close(prev_sampled, torch.tensor([9, 0, 9, 3, 9], dtype=torch.int32))
    torch.testing.assert_close(state.num_accepted_tokens_gpu, torch.tensor([9, 1, 9, 1, 9], dtype=torch.int32))
    assert counter.host_reads == 0
    assert counter.dynamic_shapes == 0


def test_fused_graph_replay_refreshes_raw_request_order_and_tree_mask(fake_npu):
    contexts, _ = _layers(count=2)
    state = _model_state()
    state._treewy_commit_token = (contexts, 1)
    prev_path = torch.full((5, 6), -7, dtype=torch.int32)
    prev_sampled = torch.full((5,), 9, dtype=torch.int32)
    state.num_accepted_tokens_gpu = torch.full((5,), 9, dtype=torch.int32)
    state._tree_buffers = lambda device: (prev_path, prev_sampled)
    idx = torch.tensor([3, 1])
    buffers = (
        torch.empty_like(idx), torch.empty_like(_PATH, dtype=torch.int32),
        torch.empty_like(_NUM_SAMPLED), torch.empty(_ROWS, dtype=torch.bool),
    )

    class ReplayBookkeeping(fake_npu):
        def replay(self):
            super().replay()
            state._commit_treewy_bookkeeping(*buffers, prev_path, prev_sampled)

    graph = ReplayBookkeeping()
    path_buffer, sampled_buffer = state._treewy_commit_inputs(_ROWS, _PATH_WIDTH, _PATH.device)
    key = (id(contexts), 1, _ROWS, _PATH_WIDTH, (
        _ROWS, idx.dtype, prev_path.data_ptr(), prev_sampled.data_ptr(), state.num_accepted_tokens_gpu.data_ptr(),
    ))
    state._treewy_commit_graphs[key] = (
        *_captured_entry(graph, contexts, None, path_buffer, sampled_buffer), buffers,
    )

    assert state._run_treewy_commit_graph(contexts, _PATH, _NUM_SAMPLED, bookkeeping=(idx, None))
    torch.testing.assert_close(prev_sampled[idx], _NUM_SAMPLED)

    # Same batch shape, different persistent request slots, path, count, and tree membership.
    idx = torch.tensor([0, 4])
    changed_path = _PATH.flip(0)
    changed_sampled = torch.tensor([1, 2], dtype=torch.int32)
    assert state._run_treewy_commit_graph(
        contexts, changed_path, changed_sampled, bookkeeping=(idx, torch.tensor([False, True]))
    )
    assert graph.replays == 2
    torch.testing.assert_close(prev_path[idx, :_PATH_WIDTH], changed_path.to(torch.int32))
    torch.testing.assert_close(prev_sampled, torch.tensor([0, 4, 9, 3, 2], dtype=torch.int32))
    torch.testing.assert_close(state.num_accepted_tokens_gpu, torch.tensor([1, 1, 9, 1, 1], dtype=torch.int32))
