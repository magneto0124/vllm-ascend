"""Short-conv history hand-off between a tree step and the step before it.

``gdn.tree_causal_conv1d`` re-anchors the committed history (conv-state columns
``0 .. width - 2``) before it convolves the step's tokens, and where the history
comes from depends on the operator that produced the *previous* step:

* a step that ran the tree op left every node's activation in the node columns
  ``width - 1 + node``, so the history has to be rebuilt from them (the "tree"
  hand-off);
* a step that ran the linear operator (``npu_causal_conv1d_custom`` ->
  ``WriteBackState``: prompt chunks, rows the scheduler gave no draft tokens, the
  first step after a prefill) already leaves the last ``width - 1`` activations of
  the committed path in columns ``0 .. width - 2``, so the history must be left
  alone (the "chain" hand-off). This is the invariant the op's own docstring
  states: the prefix columns hold the linear-chain convention at all times.

``compute_tree_prefix_source`` picks the hand-off from the staged sample count:
``0`` maps the committed columns onto themselves, while ``>= 1`` is read as "the
previous step's root is the last accepted node, one node per accepted token". A
tree step that accepted only its root reports ``1`` with an all ``-1`` path, which
is numerically identical to what a linear step reports, so the count the runtime
stages decides -- not the raw sampler output. ``AscendMambaHybridModelState``
stages ``0`` for every row whose step verified a single token (see
``tests/ut/worker/test_tree_conv_state_handoff.py`` for that decision and for the
regression guard on it).

The tests here pin what the op has to produce for either staged value. The
chain-shaped tree makes the contract exact -- when the parents form a chain the
tree step has to reproduce the plain linear convolution of the same tokens, tap
for tap -- so a wrong tap or a wrong tap order cannot cancel out.
"""

import pytest
import torch

from vllm_ascend.ops import gdn
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_conv_rows,
    compute_tree_prefix_source,
)

_WIDTH = 4  # conv kernel width: three committed taps plus the token itself
_DIM = 3
_BUDGET = 2
_SLOTS = 1 + _BUDGET  # root plus two draft nodes
_ROWS = 1
_BLOCKS = 4
_PAGES = torch.tensor([1], dtype=torch.int32)
_QUERY_START_LOC = torch.tensor([0, _SLOTS], dtype=torch.int32)
# A tree whose parents form a chain: node 1 follows the root, node 2 node 1.
_CHAIN_PARENTS = torch.tensor([[0, 1]], dtype=torch.long)
_CHAIN_NUM_NODES = torch.tensor([_BUDGET], dtype=torch.int32)
# The previous step of a row that verified no draft token and sampled its own
# bonus token. The samplers report the neutral count with an all ``-1`` path both
# for a tree step that accepted no draft node and for a linear step (a prompt
# chunk, a row without draft tokens) -- the two are numerically identical, which
# is why the model state resolves the collision before staging (see the module
# docstring).
_PREV_NO_DRAFT_PATH = torch.full((_ROWS, _BUDGET), -1, dtype=torch.int32)
# What the model state stages for a row that verified a single token: the linear
# operator committed it, so the committed columns are the history to keep.
_PREV_CHAIN_SAMPLED = torch.tensor([0], dtype=torch.int32)
_PREV_ONE_SAMPLED = torch.tensor([1], dtype=torch.int32)
# Weights with a distinct value per tap and dimension, so a wrong tap or a wrong
# tap order cannot cancel out.
_WEIGHT = torch.arange(1, _DIM * _WIDTH + 1, dtype=torch.float32).reshape(_DIM, _WIDTH) / 10
_BIAS = torch.tensor([0.0, 0.1, -0.2])


@pytest.fixture(autouse=True)
def dim_last(monkeypatch):
    """Dim-last conv state; the tap order is layout independent."""
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)


def _conv_state() -> torch.Tensor:
    # width - 1 committed columns plus one column per node, as the tree op writes.
    return torch.zeros((_BLOCKS, _WIDTH - 1 + _SLOTS, _DIM), dtype=torch.float32)


def _set_columns(state: torch.Tensor, values: torch.Tensor, start: int) -> None:
    state[_PAGES, start : start + values.shape[0], :] = values


def _get_columns(state: torch.Tensor, start: int, columns: int) -> torch.Tensor:
    return state[_PAGES, start : start + columns, :]


def _chain_reference(history: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Linear convolution of ``history + x``, taps oldest first.

    ``compute_tree_conv_rows`` documents the same convention ("rows of the taps
    (oldest first)"), and for a chain-shaped tree every node's window is exactly
    the linear window of the concatenated sequence.
    """
    sequence = torch.cat((history, x), dim=0)
    taps = sequence.unfold(0, _WIDTH, 1).transpose(1, 2)
    return torch.nn.functional.silu((taps * _WEIGHT.transpose(0, 1)).sum(dim=1) + _BIAS)


def _run_tree_step(conv_state: torch.Tensor, x: torch.Tensor, prefix_src: torch.Tensor) -> torch.Tensor:
    window_rows = compute_tree_conv_rows(
        _CHAIN_PARENTS,
        _CHAIN_NUM_NODES,
        _WIDTH,
        _ROWS * (_WIDTH - 1),
    )
    return gdn.tree_causal_conv1d(
        x,
        _WEIGHT,
        conv_state,
        _PAGES[:, None].expand(-1, _SLOTS).contiguous(),
        _QUERY_START_LOC,
        window_rows,
        prefix_src,
        _BIAS,
        True,
    )


def test_chain_shaped_tree_step_keeps_the_history_left_by_a_chain_step():
    """The previous step ran the linear operator, so columns 0..width-2 are it."""
    history = torch.arange((_WIDTH - 1) * _DIM, dtype=torch.float32).reshape(_WIDTH - 1, _DIM)
    # Node columns of an *older* tree step: the linear operator never writes them.
    stale_nodes = torch.full((_SLOTS, _DIM), -7.0)
    x = torch.arange(_SLOTS * _DIM, dtype=torch.float32).reshape(_SLOTS, _DIM) + 100.0
    conv_state = _conv_state()
    _set_columns(conv_state, history, 0)
    _set_columns(conv_state, stale_nodes, _WIDTH - 1)

    prefix_src = compute_tree_prefix_source(_PREV_NO_DRAFT_PATH, _PREV_CHAIN_SAMPLED, _WIDTH)

    out = _run_tree_step(conv_state, x, prefix_src)

    assert torch.allclose(out, _chain_reference(history, x), atol=1e-6), (
        "the previous step was produced by the linear operator, which left the window ending at its last "
        "token in columns 0..width-2; the tree step must keep it, but re-anchored the history out of a "
        f"stale node column instead (prefix_src={prefix_src.tolist()})"
    )
    # The step's own activations are still kept for the next step.
    assert torch.allclose(_get_columns(conv_state, _WIDTH - 1, _SLOTS), x)


def test_chain_shaped_tree_step_rebuilds_the_history_of_a_tree_step():
    """Control: the same tables are right when the previous step was a tree step."""
    history = torch.arange((_WIDTH - 1) * _DIM, dtype=torch.float32).reshape(_WIDTH - 1, _DIM)
    # The previous step accepted only its root, whose activation sits in node
    # column ``width - 1``; the two activations before it are the committed tail.
    previous_root = torch.full((1, _DIM), 77.0)
    x = torch.arange(_SLOTS * _DIM, dtype=torch.float32).reshape(_SLOTS, _DIM) + 100.0
    conv_state = _conv_state()
    _set_columns(conv_state, history, 0)
    _set_columns(conv_state, previous_root, _WIDTH - 1)

    prefix_src = compute_tree_prefix_source(_PREV_NO_DRAFT_PATH, _PREV_ONE_SAMPLED, _WIDTH)

    out = _run_tree_step(conv_state, x, prefix_src)

    # The window the next root continues from ends at the previous accepted node.
    expected_history = torch.cat((history[1:], previous_root), dim=0)
    assert torch.allclose(out, _chain_reference(expected_history, x), atol=1e-6), (
        f"a tree step that accepted the root only must rebuild the history from the node columns "
        f"(prefix_src={prefix_src.tolist()})"
    )


def test_the_staged_count_chooses_the_hand_off():
    """The same tables, both staged counts: identity vs. rebuild from the root."""
    assert compute_tree_prefix_source(_PREV_NO_DRAFT_PATH, _PREV_CHAIN_SAMPLED, _WIDTH).tolist() == [[0, 1, 2]], (
        "a linear step leaves the committed columns where the next root continues from, so the mapping "
        "has to be the identity"
    )
    assert compute_tree_prefix_source(_PREV_NO_DRAFT_PATH, _PREV_ONE_SAMPLED, _WIDTH).tolist() == [
        [1, 2, _WIDTH - 1]
    ], "a tree step that accepted the root only has its root in node column ``width - 1``"


def test_prefix_source_after_a_tree_step_rebuilds_from_the_accepted_node():
    """Control: the tree mapping itself must not change."""
    path = torch.tensor([[1, 2]], dtype=torch.int32)

    # The root only: the root's own node column becomes the newest committed one.
    assert compute_tree_prefix_source(path, _PREV_ONE_SAMPLED, _WIDTH).tolist() == [[1, 2, _WIDTH - 1]]
    # The whole chain: the accepted nodes' activations, oldest first.
    assert compute_tree_prefix_source(path, torch.tensor([3], dtype=torch.int32), _WIDTH).tolist() == [
        [_WIDTH - 1, _WIDTH, _WIDTH + 1]
    ]
