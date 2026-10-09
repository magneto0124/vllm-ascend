from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.ascend_config import TreeSpecConfig
from vllm_ascend.worker.v2.attn_utils import _widen_mamba_spec_for_tree
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    DEFAULT_GDN_STATE_UPDATE,
    GDN_RECURRENT_MAX_VERIFY_TOKENS,
    GDN_STATE_UPDATE_METHODS,
    GdnStateUpdateMethod,
    compute_tree_conv_rows,
    compute_tree_init_state_indices,
    compute_tree_resume_column,
    compute_tree_verify_plan,
    tree_state_slot_count,
    tree_verify_chunk_count,
    validate_tree_state_slots,
)

_NUM_SPEC = 3
# Largest tree width the single-call verify path serves: 1 + budget == MAX_MTP
# (16). A wider tree is verified in chunks, see the verify-plan tests below.
_BUDGET = 15
_CONV_DIM = 8192
_CONV_WIDTH_MINUS_1 = 3  # width == 4
_METHOD = "snapshot"


@pytest.fixture
def tree_enabled(monkeypatch):
    """Force ``tree_spec_config`` on with a fixed budget."""

    def _enable(budget=_BUDGET, gdn_state_update=_METHOD):
        monkeypatch.setattr(
            "vllm_ascend.worker.v2.spec_decode.dflash_tree_spec_enabled",
            lambda vllm_config=None: True,
        )
        monkeypatch.setattr(
            "vllm_ascend.ascend_config.get_ascend_config",
            lambda: SimpleNamespace(
                tree_spec_config=SimpleNamespace(
                    budget=budget,
                    gdn_state_update=gdn_state_update,
                )
            ),
        )

    return _enable


def _config(**kwargs):
    # ``_widen_mamba_spec_for_tree`` reads the speculation width off the real
    # VllmConfig, so the stub has to carry a speculative config as well.
    return SimpleNamespace(
        **{
            "use_v2_model_runner": True,
            "speculative_config": SimpleNamespace(num_speculative_tokens=_NUM_SPEC),
            **kwargs,
        }
    )


def _spec(num_spec=_NUM_SPEC, mamba_type=MambaAttentionBackendEnum.GDN_ATTN):
    conv_state_len = _CONV_WIDTH_MINUS_1 + num_spec
    return MambaSpec(
        shapes=((_CONV_DIM, conv_state_len), (32, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        block_size=1024,
        mamba_type=mamba_type,
        num_speculative_blocks=num_spec,
        page_size_padded=2 * 1024 * 1024 + _CONV_DIM * conv_state_len * 2,
    )


def test_resume_column_uses_last_accepted_node():
    # num_sampled counts the accepted drafts plus the trailing resample, so the
    # last accepted node sits at index num_sampled - 2. path index num_sampled - 1
    # is already the -1 padding of the tree walk.
    path = torch.tensor([[7, 11, -1, -1], [4, -1, -1, -1], [-1, -1, -1, -1]])
    num_sampled = torch.tensor([3, 2, 1], dtype=torch.int32)

    assert compute_tree_resume_column(path, num_sampled).tolist() == [12, 5, 1]


def test_resume_column_falls_back_to_root_slot():
    # Chunked prefill samples nothing and the tree walk only pads the paths.
    path = torch.full((2, 4), -1, dtype=torch.long)
    num_sampled = torch.zeros((2,), dtype=torch.int32)

    column = compute_tree_resume_column(path, num_sampled)

    assert column.tolist() == [1, 1]
    assert column.dtype == torch.int32


def test_resume_column_covers_fully_accepted_path():
    path = torch.arange(1, 5).unsqueeze(0)
    num_sampled = torch.tensor([5], dtype=torch.int32)

    assert compute_tree_resume_column(path, num_sampled).tolist() == [5]


def test_slot_count_follows_budget_when_tree_enabled(tree_enabled):
    tree_enabled(budget=_BUDGET)

    assert tree_state_slot_count(_config()) == 1 + _BUDGET


def test_slot_count_ignores_tree_without_v2_runner(tree_enabled):
    tree_enabled()

    assert tree_state_slot_count(_config(use_v2_model_runner=False)) is None


def test_slot_count_ignores_disabled_tree(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash_tree_spec_enabled",
        lambda vllm_config=None: False,
    )

    assert tree_state_slot_count(_config()) is None


def test_validate_rejects_other_state_backends():
    with pytest.raises(NotImplementedError, match="only GDN_ATTN"):
        validate_tree_state_slots(4, MambaAttentionBackendEnum.MAMBA2)


def test_validate_accepts_a_tree_wider_than_one_kernel_call():
    # A tree wider than the recurrent kernel's per-call width is verified in
    # chunks (compute_tree_verify_plan), so it is not rejected at startup.
    validate_tree_state_slots(GDN_RECURRENT_MAX_VERIFY_TOKENS * 4, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_rejects_an_empty_verify():
    with pytest.raises(ValueError, match="at least one token"):
        validate_tree_state_slots(0, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_accepts_the_registered_snapshot_method(tree_enabled):
    """``snapshot`` is the method the GDN state path implements end to end."""
    tree_enabled()
    method = GDN_STATE_UPDATE_METHODS[_METHOD]
    assert method.state_init and method.conv

    validate_tree_state_slots(4, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_rejects_an_unknown_state_update(tree_enabled):
    tree_enabled(gdn_state_update="magic-snapshot")

    with pytest.raises(ValueError, match="gdn_state_update must be one of"):
        validate_tree_state_slots(4, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_ignores_the_state_update_for_other_backends(tree_enabled):
    # The key is GDN-only: another state backend fails on the backend and never
    # on the value -- even a nonsense one.
    tree_enabled(gdn_state_update="magic-snapshot")

    with pytest.raises(NotImplementedError, match="only GDN_ATTN"):
        validate_tree_state_slots(4, MambaAttentionBackendEnum.MAMBA2)


def test_validate_reports_a_method_without_parent_slot_state(monkeypatch, tree_enabled):
    monkeypatch.setitem(
        GDN_STATE_UPDATE_METHODS,
        "no-parent-state",
        GdnStateUpdateMethod(state_init=False, conv=True, description=""),
    )
    tree_enabled(gdn_state_update="no-parent-state")

    with pytest.raises(NotImplementedError, match="parent slot"):
        validate_tree_state_slots(4, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_reports_a_method_without_conv_history(monkeypatch, tree_enabled):
    monkeypatch.setitem(
        GDN_STATE_UPDATE_METHODS,
        "no-conv-history",
        GdnStateUpdateMethod(state_init=True, conv=False, description=""),
    )
    tree_enabled(gdn_state_update="no-conv-history")

    with pytest.raises(NotImplementedError, match="short-conv"):
        validate_tree_state_slots(4, MambaAttentionBackendEnum.GDN_ATTN)


def test_config_defers_the_state_update_to_the_gdn_path():
    """The key configures nothing by itself: a GDN model resolves it."""
    assert TreeSpecConfig().gdn_state_update == DEFAULT_GDN_STATE_UPDATE
    assert DEFAULT_GDN_STATE_UPDATE in GDN_STATE_UPDATE_METHODS

    # No membership check at parse time, so a model that does not use GDN never
    # trips over the value.
    config = TreeSpecConfig(
        enabled=True,
        method="prefix",
        budget=4,
        topk=2,
        gdn_state_update="magic-snapshot",
    )

    assert config.gdn_state_update == "magic-snapshot"


def test_slot_count_rejects_align_cache_mode(tree_enabled):
    # The align pre-copy shifts the conv window by the accepted cursor, which a
    # draft tree cannot express (see tree_state_slot_count).
    tree_enabled()
    config = _config(cache_config=SimpleNamespace(mamba_cache_mode="align"))

    with pytest.raises(NotImplementedError, match="mamba_cache_mode='align'"):
        tree_state_slot_count(config)


def test_slot_count_accepts_default_cache_mode(tree_enabled):
    tree_enabled()
    assert tree_state_slot_count(_config(cache_config=SimpleNamespace(mamba_cache_mode="none"))) == 1 + _BUDGET


def test_init_state_indices_follow_tree_parents():
    state_indices = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32
    )
    # Row 0 resumes from the root slot, row 1 from the slot of node 2.
    num_accepted = torch.tensor([1, 3], dtype=torch.int32)
    # Node 2 of row 0 branches off the root instead of node 1.
    tree_parents = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
    tree_num_nodes = torch.tensor([3, 3], dtype=torch.int32)

    init = compute_tree_init_state_indices(
        state_indices, num_accepted, tree_parents, tree_num_nodes
    )

    # Column 0 is the resume page, column j the page of node j's parent.
    assert init.tolist() == [[10, 10, 10, 11], [22, 20, 21, 22]]
    assert init.dtype == torch.int32


def test_init_state_indices_use_chain_parents_for_folded_rows():
    state_indices = torch.tensor([[10, 11, 12]], dtype=torch.int32)
    num_accepted = torch.tensor([1], dtype=torch.int32)
    # A prompt chunk folded into the spec batch carries no tree.
    tree_parents = torch.zeros((1, 2), dtype=torch.long)
    tree_num_nodes = torch.zeros((1,), dtype=torch.int32)

    init = compute_tree_init_state_indices(
        state_indices, num_accepted, tree_parents, tree_num_nodes
    )

    # Token t points at token t-1's page, so the kernel keeps its carry.
    assert init.tolist() == [[10, 10, 11]]


def test_init_state_indices_require_matching_budget():
    state_indices = torch.zeros((1, 4), dtype=torch.int32)
    num_accepted = torch.ones((1,), dtype=torch.int32)
    tree_parents = torch.zeros((1, 2), dtype=torch.long)
    tree_num_nodes = torch.ones((1,), dtype=torch.int32)

    with pytest.raises(ValueError, match="tree_spec_config.budget"):
        compute_tree_init_state_indices(
            state_indices, num_accepted, tree_parents, tree_num_nodes
        )


def test_widen_mamba_spec_adds_state_slots_and_conv_columns(tree_enabled, monkeypatch):
    # The fixture describes a dim-first conv state (the 310P / DS layout).
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.attn_utils.is_conv_state_dim_first",
        lambda: True,
    )
    spec = _spec()
    tree_enabled(budget=_BUDGET)

    widened = _widen_mamba_spec_for_tree(spec, _config())

    # One recurrent state page per tree node ...
    assert widened.num_speculative_blocks == _BUDGET
    # ... and the short-conv page keeps every node's activation, so it grows by
    # the columns the tree has beyond the linear chain.
    extra_columns = (_BUDGET + 1) - _NUM_SPEC
    assert widened.shapes[0] == (_CONV_DIM, spec.shapes[0][1] + extra_columns)
    assert widened.shapes[1] == spec.shapes[1]
    assert widened.page_size_padded == spec.page_size_padded + extra_columns * _CONV_DIM * 2
    assert widened.page_size_bytes == widened.page_size_padded


def test_widen_mamba_spec_keeps_chain_layout_without_tree(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash_tree_spec_enabled",
        lambda vllm_config=None: False,
    )
    spec = _spec()

    assert _widen_mamba_spec_for_tree(spec, _config()) is spec


def test_widen_mamba_spec_widens_the_conv_page_at_chain_width(tree_enabled, monkeypatch):
    # ``budget == num_speculative_tokens``: the recurrent state needs no extra
    # page, but the conv page still serves one node per verified token, i.e. one
    # column more than the linear draft chain.
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.attn_utils.is_conv_state_dim_first",
        lambda: True,
    )
    spec = _spec()
    tree_enabled(budget=_NUM_SPEC)

    widened = _widen_mamba_spec_for_tree(spec, _config())

    assert widened is not spec
    assert widened.num_speculative_blocks == spec.num_speculative_blocks
    assert widened.shapes[0] == (_CONV_DIM, spec.shapes[0][1] + 1)
    assert widened.shapes[1] == spec.shapes[1]
    assert widened.page_size_padded == spec.page_size_padded + _CONV_DIM * 2


def test_tree_path_buffers_keep_the_tree_width(tree_enabled):
    # The sampler hands over a path as wide as the draft chain while the tree
    # budget is wider. The buffers must keep the tree width, or the width check
    # in ``_tree_buffers`` re-creates (and zeroes) them on every step and the
    # short-conv re-anchor silently never happens.
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridModelState,
    )

    tree_enabled(budget=_BUDGET)
    state = SimpleNamespace(
        vllm_config=_config(),
        num_accepted_tokens_gpu=torch.zeros((2,), dtype=torch.int32),
        _tree_prev_path=None,
        _tree_prev_num_sampled=None,
    )

    path, num_sampled = AscendMambaHybridModelState._tree_buffers(state, torch.device("cpu"))
    stored_path, stored_num_sampled = AscendMambaHybridModelState._tree_buffers(state, torch.device("cpu"))

    assert path.shape == (2, _BUDGET)
    assert num_sampled.shape == (2,)
    # Same buffers on the next step, so what postprocess_state stored is still
    # there when prepare_attn reads it.
    assert stored_path is path
    assert stored_num_sampled is num_sampled


def _expected_window_rows(history_base, node_offset, request, slots, path_ids, depth, width):
    """Row ids of the ``width`` taps ending at ``path_ids[depth]``."""
    rows = []
    for tap in range(width):
        position = depth - (width - 1) + tap
        if position >= 0:
            rows.append(node_offset + request * slots + path_ids[position])
        else:
            # The committed activations of the request, oldest first.
            rows.append(history_base + width - 1 + position)
    return rows


def _paths(parents_row, slots):
    """Root-to-node node ids of every node of one request."""
    parents = [-1] + parents_row.tolist()
    paths = [[0]]
    for node in range(1, slots):
        paths.append(paths[parents[node]] + [node])
    return paths


def test_conv_window_rows_follow_node_ancestors():
    slots = 4
    width = 4
    # Node 2 of row 0 branches off the root instead of node 1; row 1 is a chain.
    tree_parents = torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long)
    tree_num_nodes = torch.tensor([3, 3], dtype=torch.int32)
    rows = tree_parents.shape[0]
    node_offset = rows * (width - 1)

    window_rows = compute_tree_conv_rows(tree_parents, tree_num_nodes, width, node_offset)

    expected_window = []
    for request in range(rows):
        paths = _paths(tree_parents[request], slots)
        history_base = request * (width - 1)
        for path_ids in paths:
            expected_window.append(
                _expected_window_rows(
                    history_base,
                    node_offset,
                    request,
                    slots,
                    path_ids,
                    len(path_ids) - 1,
                    width,
                )
            )

    assert window_rows.tolist() == expected_window
    assert window_rows.dtype == torch.int32
    # A node never sees its siblings: node 2 of row 0 branches off the root, so
    # its parent tap is the root and not node 1.
    assert window_rows[2].tolist() == expected_window[2]
    assert window_rows[2].tolist()[-2] == node_offset


def test_conv_window_rows_keep_batch_order_for_folded_rows():
    # A row folded in from a prompt chunk (tree_num_nodes == 0) is a linear
    # sequence, so its window must be the batch-order one.
    slots = 3
    width = 4
    tree_parents = torch.zeros((1, slots - 1), dtype=torch.long)
    tree_num_nodes = torch.zeros((1,), dtype=torch.int32)
    node_offset = width - 1

    window_rows = compute_tree_conv_rows(tree_parents, tree_num_nodes, width, node_offset)

    expected_window = [
        _expected_window_rows(0, node_offset, 0, slots, [0, 1, 2], depth, width)
        for depth in range(slots)
    ]

    assert window_rows.tolist() == expected_window


# ---------------------------------------------------------------------------
# Chunked verify of a tree wider than one recurrent-op call
# ---------------------------------------------------------------------------


def _tree_tables(widths, slots):
    """Packed state/parent tables of a chained tree, one request per row.

    Row ``r`` keeps node ``j``'s state in page ``r * slots + j`` and starts from
    the committed page ``rows * slots + r``; node ``j > 0`` continues node
    ``j - 1``, so every parent has a smaller node id, which is what the chunk
    chain relies on. Only the first ``sum(widths)`` entries carry a state: the
    tables are packed token-major, like the builder packs them.
    """
    rows = widths.numel()
    state = torch.zeros((rows, slots), dtype=torch.int32)
    parents = torch.zeros((rows, slots), dtype=torch.int32)
    flat_state = state.reshape(-1)
    flat_parents = parents.reshape(-1)
    starts = torch.cumsum(widths, dim=0) - widths
    for row in range(rows):
        for node in range(int(widths[row])):
            token = int(starts[row]) + node
            flat_state[token] = row * slots + node
            flat_parents[token] = rows * slots + row if node == 0 else row * slots + node - 1
    return state, parents


class _FakeRecurrentKernel:
    """State bookkeeping of the AscendC ``Process`` loop.

    Mirrors ``csrc/attention/recurrent_gated_delta_rule/op_kernel``: rows are
    walked in order over their ``cu_seqlens`` range, a row of length 0 is skipped,
    a row longer than ``MAX_MTP`` and a negative initial page abort the whole
    call, the first token of a row loads ``init_state_indices`` unconditionally
    and a later token only reloads when its parent page differs from the page the
    previous token of the same row wrote.
    """

    def __init__(self, num_pages, max_mtp=GDN_RECURRENT_MAX_VERIFY_TOKENS):
        self.pages = [0.0] * num_pages
        self.max_mtp = max_mtp

    def run(self, cu_seqlens, state_indices, init_state_indices, values):
        seq0 = int(cu_seqlens[0])
        for row in range(cu_seqlens.numel() - 1):
            seq_len = int(cu_seqlens[row + 1])
            if seq_len <= 0:
                continue
            if seq_len > self.max_mtp:
                return
            seq1 = seq0 + seq_len
            if int(init_state_indices[seq0]) < 0:
                return
            state = self.pages[int(init_state_indices[seq0])]
            for token in range(seq0, seq1):
                if token > seq0:
                    parent = int(init_state_indices[token])
                    if parent < 0:
                        return
                    if parent != int(state_indices[token - 1]):
                        state = self.pages[parent]
                state = state + float(values[token])
                self.pages[int(state_indices[token])] = state
            seq0 = seq1


def _run_chunked(plan, values, num_pages):
    kernel = _FakeRecurrentKernel(num_pages)
    for chunk in plan.chunks:
        kernel.run(
            chunk.cu_seqlens,
            chunk.state_indices,
            chunk.init_state_indices,
            values[chunk.token_indices],
        )
    return kernel.pages


def test_verify_plan_is_none_while_the_tree_fits_one_call():
    widths = torch.tensor([GDN_RECURRENT_MAX_VERIFY_TOKENS, 4], dtype=torch.int32)
    state, parents = _tree_tables(widths, GDN_RECURRENT_MAX_VERIFY_TOKENS)

    assert compute_tree_verify_plan(widths, state, parents, scratch_base=16) is None


def test_verify_chunk_count_follows_the_kernel_width():
    width = GDN_RECURRENT_MAX_VERIFY_TOKENS

    assert tree_verify_chunk_count(width) == 1
    assert tree_verify_chunk_count(width + 1) == 2
    assert tree_verify_chunk_count(2 * width) == 2
    assert tree_verify_chunk_count(2 * width + 1) == 3


def test_verify_plan_covers_every_token_exactly_once():
    width = GDN_RECURRENT_MAX_VERIFY_TOKENS
    slots = 2 * width
    widths = torch.tensor([3, slots, slots - 1, 0], dtype=torch.int32)
    state, parents = _tree_tables(widths, slots)
    scratch_base = 4096

    plan = compute_tree_verify_plan(widths, state, parents, scratch_base=scratch_base)

    assert plan is not None
    assert plan.width == width
    assert plan.num_chunks == 2
    assert plan.chunk_tokens == plan.rows * width
    starts = torch.cumsum(widths, dim=0) - widths
    # The packed tables are their own token -> page map (see _tree_tables).
    flat_state = state.reshape(-1)
    flat_parents = parents.reshape(-1)
    seen = []
    for index, chunk in enumerate(plan.chunks):
        assert chunk.token_indices.shape == (plan.chunk_tokens,)
        assert chunk.token_indices.dtype == torch.long
        assert chunk.out_indices.dtype == torch.long
        assert chunk.state_indices.dtype == torch.int32
        assert chunk.init_state_indices.dtype == torch.int32
        # The kernel's ``actual_seq_lengths`` convention: an unused base entry
        # followed by the per-row token count. A row of length 0 is skipped.
        row_lengths = chunk.cu_seqlens[1:]
        assert chunk.cu_seqlens.dtype == torch.int32
        assert bool((row_lengths <= width).all())
        valid = (torch.arange(width).unsqueeze(0) < row_lengths.unsqueeze(1)).reshape(-1)
        padding = ~valid
        # Padding cells are never processed: they carry no page in either table
        # and their (undefined) op output goes past every verified token.
        assert bool((chunk.state_indices[padding] == -1).all())
        assert bool((chunk.init_state_indices[padding] == -1).all())
        assert bool((chunk.out_indices[padding] >= scratch_base).all())
        # A verified cell reads its own parent page and writes its own page, both
        # taken from the packed table of the step, and its op output lands on the
        # token's own row of the output buffer.
        tokens = chunk.token_indices[valid]
        assert bool((chunk.state_indices[valid] == flat_state[tokens]).all())
        assert bool((chunk.init_state_indices[valid] == flat_parents[tokens]).all())
        assert bool((chunk.out_indices[valid] == tokens).all())
        # Chunk ``index`` owns the node columns [index * width, (index + 1) * width)
        # of every row, in order.
        token_grid = chunk.token_indices.reshape(plan.rows, width)
        mask_grid = valid.reshape(plan.rows, width)
        for row in range(plan.rows):
            low = index * width
            high = min(int(widths[row]), low + width)
            expected = list(range(int(starts[row]) + low, int(starts[row]) + high))
            assert token_grid[row][mask_grid[row]].tolist() == expected
            seen.extend(expected)

    assert sorted(seen) == list(range(int(widths.sum())))


def test_chunked_verify_matches_one_wide_call():
    width = GDN_RECURRENT_MAX_VERIFY_TOKENS
    slots = 2 * width
    # A row exactly one chunk wide, a full row, a short row, an empty (padded)
    # row and a row that spans both chunks.
    widths = torch.tensor([width, slots, 7, 0, slots - 3], dtype=torch.int32)
    state, parents = _tree_tables(widths, slots)
    tokens = int(widths.sum())
    values = torch.arange(1, tokens + 1, dtype=torch.float32)
    num_pages = int(widths.numel()) * slots + int(widths.numel()) + 8

    # Reference: the step the kernel would run on its own if it could hold every
    # node in one call (which is exactly what it must not be asked to do).
    reference = _FakeRecurrentKernel(num_pages, max_mtp=tokens)
    reference.run(
        torch.cat((torch.zeros(1, dtype=torch.int32), widths)),
        state.reshape(-1),
        parents.reshape(-1),
        values,
    )

    plan = compute_tree_verify_plan(widths, state, parents, scratch_base=tokens)
    assert plan is not None
    assert plan.num_chunks == 2

    assert _run_chunked(plan, values, num_pages) == reference.pages


def test_chunked_verify_chains_rows_through_their_parent_pages():
    # Every row spans two chunks, so the first node of a row in the second chunk
    # has to reload the state its parent page got in the first chunk.
    width = GDN_RECURRENT_MAX_VERIFY_TOKENS
    slots = 2 * width
    widths = torch.tensor([slots, slots], dtype=torch.int32)
    state, parents = _tree_tables(widths, slots)
    plan = compute_tree_verify_plan(widths, state, parents, scratch_base=2 * slots)

    assert plan is not None
    for chunk in plan.chunks:
        # The kernel accumulates the per-row lengths into token offsets itself.
        row_starts = chunk.cu_seqlens.cumsum(0)[:-1]
        for row in range(plan.rows):
            start = int(row_starts[row])
            end = start + int(chunk.cu_seqlens[row + 1])
            assert end > start
            # The row's first cell always carries a real parent page: the kernel
            # returns on a negative one.
            assert int(chunk.init_state_indices[start]) >= 0
            # A node that continues the previous token of the same row must not
            # reload -- the state the previous token wrote is the one it needs.
            for token in range(start + 1, end):
                assert int(chunk.init_state_indices[token]) == int(chunk.state_indices[token - 1])
