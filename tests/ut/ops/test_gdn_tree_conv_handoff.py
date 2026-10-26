"""Short-conv history hand-off: the staged sample count picks whether the tree op
keeps the committed columns or rebuilds them from the node columns.
"""

import pytest
import torch

from vllm_ascend.ops import gdn
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_conv_rows,
    compute_tree_prefix_source,
)

_WIDTH = 4  # three committed taps plus the token itself
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
# Root-only tree steps and linear steps report the same count; state resolves it.
_PREV_NO_DRAFT_PATH = torch.full((_ROWS, _BUDGET), -1, dtype=torch.int32)
# Verified a single token: the linear operator committed it, so keep the columns.
_PREV_CHAIN_SAMPLED = torch.tensor([0], dtype=torch.int32)
_PREV_ONE_SAMPLED = torch.tensor([1], dtype=torch.int32)
# A distinct value per tap and dimension, so a wrong tap cannot cancel out.
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
    """Linear convolution of ``history + x``, taps oldest first."""
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
    history = torch.arange((_WIDTH - 1) * _DIM, dtype=torch.float32).reshape(_WIDTH - 1, _DIM)
    # Root-only step: its activation sits in node column ``width - 1``.
    previous_root = torch.full((1, _DIM), 77.0)
    x = torch.arange(_SLOTS * _DIM, dtype=torch.float32).reshape(_SLOTS, _DIM) + 100.0
    conv_state = _conv_state()
    _set_columns(conv_state, history, 0)
    _set_columns(conv_state, previous_root, _WIDTH - 1)

    prefix_src = compute_tree_prefix_source(_PREV_NO_DRAFT_PATH, _PREV_ONE_SAMPLED, _WIDTH)

    out = _run_tree_step(conv_state, x, prefix_src)

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
    path = torch.tensor([[1, 2]], dtype=torch.int32)

    # The root only: the root's own node column becomes the newest committed one.
    assert compute_tree_prefix_source(path, _PREV_ONE_SAMPLED, _WIDTH).tolist() == [[1, 2, _WIDTH - 1]]
    # The whole chain: the accepted nodes' activations, oldest first.
    assert compute_tree_prefix_source(path, torch.tensor([3], dtype=torch.int32), _WIDTH).tolist() == [
        [_WIDTH - 1, _WIDTH, _WIDTH + 1]
    ]
