import pytest
import torch

from vllm_ascend.ops import gdn
from vllm_ascend.ops.gdn_attn_builder import _pack_tree_rows_to_tokens
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_conv_rows,
    compute_tree_prefix_source,
    compute_tree_resume_column,
)

_WIDTH = 4  # three committed taps
_DIM = 3
_BLOCKS = 4
_SLOTS = 3  # 1 + budget
_ROWS = 2
# Node 2 of row 0 branches off the root instead of node 1; row 1 is a chain.
_TREE_PARENTS = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
_TREE_NUM_NODES = torch.tensor([2, 2], dtype=torch.int32)
# Sampler path layout: entry ``k`` = depth ``k + 1`` node, ``-1`` = unused slot.
_TREE_PATH = torch.tensor([[-1, -1], [1, 2]], dtype=torch.int32)
_PAGES = torch.tensor([1, 2], dtype=torch.int32)
_QUERY_START_LOC = torch.tensor([0, _SLOTS, 2 * _SLOTS], dtype=torch.int32)


@pytest.fixture
def dim_first(request, monkeypatch):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: request.param)
    return request.param


def _conv_state(dim_first: bool) -> torch.Tensor:
    columns = _WIDTH - 1 + _SLOTS
    shape = (_BLOCKS, _DIM, columns) if dim_first else (_BLOCKS, columns, _DIM)
    return torch.zeros(shape, dtype=torch.float32)


def _set_columns(state: torch.Tensor, values: torch.Tensor, start: int, dim_first: bool) -> None:
    if dim_first:
        state[_PAGES, :, start : start + values.shape[1]] = values.transpose(1, 2)
    else:
        state[_PAGES, start : start + values.shape[1], :] = values


def _get_columns(state: torch.Tensor, start: int, columns: int, dim_first: bool) -> torch.Tensor:
    pages = state[_PAGES]
    window = pages[:, :, start : start + columns] if dim_first else pages[:, start : start + columns, :]
    return window.transpose(1, 2) if dim_first else window


def _tables(num_sampled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    window_rows = compute_tree_conv_rows(
        _TREE_PARENTS,
        _TREE_NUM_NODES,
        _WIDTH,
        _ROWS * (_WIDTH - 1),
    )
    prefix_src = compute_tree_prefix_source(
        _TREE_PATH,
        num_sampled,
        _WIDTH,
    )
    return window_rows, prefix_src


@pytest.mark.parametrize("dim_first", [False, True], indirect=True)
def test_tree_conv_convolves_each_node_over_its_ancestors(dim_first):
    # Row 0 accepted no draft node, row 1 the last node of its chain.
    window_rows, prefix_src = _tables(torch.tensor([1, 3], dtype=torch.int32))
    history = torch.arange(_ROWS * (_WIDTH - 1) * _DIM, dtype=torch.float32).reshape(
        _ROWS, _WIDTH - 1, _DIM
    )
    nodes = torch.arange(_ROWS * _SLOTS * _DIM, dtype=torch.float32).reshape(_ROWS, _SLOTS, _DIM) + 100
    conv_state = _conv_state(dim_first)
    _set_columns(conv_state, history, 0, dim_first)
    # The previous step's node activations, which the history is rebuilt from.
    _set_columns(conv_state, nodes - 50, _WIDTH - 1, dim_first)
    x = nodes.reshape(-1, _DIM)
    weight = torch.randn(_DIM, _WIDTH)
    bias = torch.randn(_DIM)

    out = gdn.tree_causal_conv1d(
        x,
        weight,
        conv_state,
        _PAGES[:, None].expand(-1, _SLOTS).contiguous(),
        _QUERY_START_LOC,
        window_rows,
        prefix_src,
        bias,
        True,
    )

    # Row 0 (root only): the activation before the root plus its own node column.
    expected_history = torch.cat(
        (history[0, 1:], (nodes - 50)[0, :1]), dim=0
    )
    assert torch.allclose(_get_columns(conv_state, 0, _WIDTH - 1, dim_first)[0], expected_history)
    expected_history = (nodes - 50)[1, : _WIDTH - 1]
    assert torch.allclose(_get_columns(conv_state, 0, _WIDTH - 1, dim_first)[1], expected_history)

    source = torch.cat(
        (_get_columns(conv_state, 0, _WIDTH - 1, dim_first).reshape(-1, _DIM), x), dim=0
    )
    taps = source.index_select(0, window_rows.reshape(-1).to(torch.long)).view(-1, _WIDTH, _DIM)
    expected = torch.nn.functional.silu((taps * weight.transpose(0, 1)).sum(dim=1) + bias)
    assert torch.allclose(out, expected, atol=1e-6)

    assert torch.allclose(_get_columns(conv_state, _WIDTH - 1, _SLOTS, dim_first), nodes)


def test_tree_conv_skips_zero_length_rows(monkeypatch):
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
    # Row 1 is a padded row: the sampler leaves it without a path.
    window_rows, prefix_src = _tables(torch.tensor([1, 0], dtype=torch.int32))
    conv_state = _conv_state(dim_first=False)
    nodes = torch.arange(_ROWS * _SLOTS * _DIM, dtype=torch.float32).reshape(_ROWS, _SLOTS, _DIM) + 1
    weight = torch.randn((_DIM, _WIDTH))
    # Full-graph padding replays with zero-length requests, which own no page.
    query_start_loc = torch.tensor([0, _SLOTS, _SLOTS], dtype=torch.int32)
    # Only the rows that own a page contribute tokens to the spec batch.
    x = nodes.reshape(-1, _DIM)[:_SLOTS]

    gdn.tree_causal_conv1d(
        x,
        weight,
        conv_state,
        torch.tensor([[1], [-1]], dtype=torch.int32).expand(-1, _SLOTS).contiguous(),
        query_start_loc,
        # The layer slices the window table by the real token count.
        window_rows[: x.shape[0]],
        prefix_src,
        None,
        True,
    )

    assert torch.count_nonzero(conv_state[0]) == 0
    assert torch.count_nonzero(conv_state[1]) > 0


def test_tree_conv_packs_a_short_row_token_major(monkeypatch):
    """A row may verify fewer tokens than 1 + budget (scheduler truncation)."""
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
    row_widths = torch.tensor([2, _SLOTS], dtype=torch.int32)
    window_rows = compute_tree_conv_rows(
        _TREE_PARENTS,
        _TREE_NUM_NODES,
        _WIDTH,
        _ROWS * (_WIDTH - 1),
        row_widths=row_widths,
    )
    prefix_src = compute_tree_prefix_source(
        _TREE_PATH,
        torch.tensor([1, 3], dtype=torch.int32),
        _WIDTH,
    )
    conv_state = _conv_state(dim_first=False)
    nodes = torch.arange(_ROWS * _SLOTS * _DIM, dtype=torch.float32).reshape(_ROWS, _SLOTS, _DIM) + 100
    # The spec batch carries row 0's two tokens followed by row 1's three.
    x = torch.cat((nodes[0, :2], nodes[1]), dim=0)
    query_start_loc = torch.tensor([0, 2, 2 + _SLOTS], dtype=torch.int32)
    # The op reads the page table by global token offset, so entries must be
    # packed token-major (padding unequal rows would misread row 1's page).
    cache_indices = _pack_tree_rows_to_tokens(
        _PAGES[:, None].expand(-1, _SLOTS).contiguous(),
        row_widths,
        _SLOTS,
    )

    out = gdn.tree_causal_conv1d(
        x,
        torch.randn((_DIM, _WIDTH)),
        conv_state,
        cache_indices,
        query_start_loc,
        window_rows[: x.shape[0]],
        prefix_src,
        None,
        True,
    )

    assert out.shape == x.shape
    assert torch.allclose(_get_columns(conv_state, _WIDTH - 1, 2, False)[0], nodes[0, :2])
    assert torch.allclose(_get_columns(conv_state, _WIDTH - 1, _SLOTS, False)[1], nodes[1])


@pytest.mark.parametrize("activation", [False, True])
@pytest.mark.parametrize("bias_value", [None, 0.00390625])
def test_tree_conv_accumulates_bf16_taps_in_fp32(monkeypatch, activation, bias_value):
    """A BF16 multiply rounds away information before the native FP32 sum."""
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
    state = torch.zeros((2, _WIDTH, 1), dtype=torch.bfloat16)
    state[1, : _WIDTH - 1, 0] = 100
    x = torch.tensor([[100]], dtype=torch.bfloat16)
    weight = torch.tensor([[1.0078125, -1, 1.0078125, -1]], dtype=torch.bfloat16)
    bias = None if bias_value is None else torch.tensor([bias_value], dtype=x.dtype)

    out = gdn.tree_causal_conv1d(
        x,
        weight,
        state,
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        bias,
        activation,
    )

    # BF16 products round 100 * 1.0078125 to 101; the native kernel uses FP32.
    assert out.dtype == x.dtype
    expected = torch.tensor([[1.5625 + (bias_value or 0)]], dtype=torch.float32)
    if activation:
        expected = torch.nn.functional.silu(expected)
    torch.testing.assert_close(out, expected.to(x.dtype), rtol=0, atol=0)


def test_tree_prefix_source_follows_the_accepted_node():
    """The history source ends on the node the next root continues from."""
    path_node_ids = torch.tensor([[-1, -1, -1], [1, 2, -1]], dtype=torch.int32)

    # A skipped row (folded prompt chunk) keeps the chain history where it is.
    assert compute_tree_prefix_source(path_node_ids, torch.tensor([0, 0]), _WIDTH).tolist() == [
        [0, 1, 2],
        [0, 1, 2],
    ]
    # Row 0 accepted the root only; row 1 accepted nodes 1 and 2.
    num_sampled = torch.tensor([1, 3], dtype=torch.int32)
    prefix_src = compute_tree_prefix_source(path_node_ids, num_sampled, _WIDTH)
    assert prefix_src.tolist() == [
        [1, 2, _WIDTH - 1 + 0],
        [_WIDTH - 1 + 0, _WIDTH - 1 + 1, _WIDTH - 1 + 2],
    ]
    # The last committed column is the resumed node's column; conv and rollback agree.
    assert (prefix_src[:, -1] - (_WIDTH - 2)).tolist() == compute_tree_resume_column(
        path_node_ids, num_sampled
    ).tolist()


def test_tree_conv_metadata_fields_default_to_chain():
    from vllm_ascend.ops.gdn_attn_builder import GDNSpecDecodeMetadata

    # None keeps the linear-chain operator and the single recurrent-op call.
    metadata = GDNSpecDecodeMetadata(spec_causal_conv1d=object(), actual_seq_lengths=object())

    assert metadata.init_state_indices is None
    assert metadata.conv_window_rows is None
    assert metadata.conv_prefix_src is None
    assert metadata.verify_plan is None
