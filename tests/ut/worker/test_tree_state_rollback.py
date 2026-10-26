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
    tree_conv_slot_count,
    tree_recurrent_state_slot_count,
    tree_state_slot_count,
    tree_verify_slot_count,
    tree_verify_chunk_count,
    validate_tree_state_slots,
)

_NUM_SPEC = 3
# Widest tree the single-call verify path serves (1 + budget == MAX_MTP).
_BUDGET = 15
_CONV_DIM = 8192
_CONV_WIDTH_MINUS_1 = 3  # width == 4
_METHOD = "snapshot"


@pytest.fixture
def tree_enabled(monkeypatch):
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
    # The real VllmConfig is read for width, so the stub carries a spec config.
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
    # num_sampled = accepted drafts + trailing resample: last accepted node is -2.
    path = torch.tensor([[7, 11, -1, -1], [4, -1, -1, -1], [-1, -1, -1, -1]])
    num_sampled = torch.tensor([3, 2, 1], dtype=torch.int32)

    assert compute_tree_resume_column(path, num_sampled).tolist() == [12, 5, 1]


def test_resume_column_falls_back_to_root_slot():
    # Chunked prefill samples nothing; the tree walk only pads the paths.
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
    assert tree_verify_slot_count(_config()) == 1 + _BUDGET
    assert tree_conv_slot_count(_config()) == 1 + _BUDGET


def test_treewy_layout_separates_verify_conv_and_recurrent_widths(tree_enabled):
    tree_enabled(budget=_BUDGET, gdn_state_update="treewy")
    config = _config()

    assert tree_verify_slot_count(config) == 1 + _BUDGET
    assert tree_conv_slot_count(config) == 1 + _BUDGET
    assert tree_recurrent_state_slot_count(config) == 1
    assert tree_state_slot_count(config) == 1
    method = GDN_STATE_UPDATE_METHODS["treewy"]
    assert method.treewy_solver and method.conv and method.recurrent_state_slots == 1
    validate_tree_state_slots(1 + _BUDGET, MambaAttentionBackendEnum.GDN_ATTN)


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
    # Over-wide trees are verified in chunks, so startup must not reject them.
    validate_tree_state_slots(GDN_RECURRENT_MAX_VERIFY_TOKENS * 4, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_rejects_an_empty_verify():
    with pytest.raises(ValueError, match="at least one token"):
        validate_tree_state_slots(0, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_accepts_the_registered_snapshot_method(tree_enabled):
    tree_enabled()
    method = GDN_STATE_UPDATE_METHODS[_METHOD]
    assert method.state_init and method.conv

    validate_tree_state_slots(4, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_rejects_an_unknown_state_update(tree_enabled):
    tree_enabled(gdn_state_update="magic-snapshot")

    with pytest.raises(ValueError, match="gdn_state_update must be one of"):
        validate_tree_state_slots(4, MambaAttentionBackendEnum.GDN_ATTN)


def test_validate_ignores_the_state_update_for_other_backends(tree_enabled):
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
    assert TreeSpecConfig().gdn_state_update == DEFAULT_GDN_STATE_UPDATE
    assert DEFAULT_GDN_STATE_UPDATE in GDN_STATE_UPDATE_METHODS

    # No membership check at parse time, so non-GDN models never trip on it.
    config = TreeSpecConfig(
        enabled=True,
        method="prefix",
        budget=4,
        topk=2,
        gdn_state_update="magic-snapshot",
    )

    assert config.gdn_state_update == "magic-snapshot"


def test_slot_count_rejects_align_cache_mode(tree_enabled):
    # The align pre-copy shifts the conv window by the accepted cursor: a tree
    # cannot express that.
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

    assert widened.num_speculative_blocks == _BUDGET
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
    # budget == num_speculative_tokens: the conv page still needs one extra column.
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


def test_widen_mamba_spec_keeps_one_recurrent_page_for_treewy(tree_enabled, monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.attn_utils.is_conv_state_dim_first",
        lambda: True,
    )
    spec = _spec()
    tree_enabled(budget=_BUDGET, gdn_state_update="treewy")

    widened = _widen_mamba_spec_for_tree(spec, _config())

    assert widened.num_speculative_blocks == 0
    assert widened.shapes[0] == (_CONV_DIM, _CONV_WIDTH_MINUS_1 + 1 + _BUDGET)
    assert widened.shapes[1] == spec.shapes[1]


def test_tree_path_buffers_keep_the_tree_width(tree_enabled):
    # Sampler paths are chain-wide; buffers must keep the tree width or they are
    # re-created (and zeroed) every step.
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
    # Same buffers next step, so postprocess_state's data survives to prepare_attn.
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
    # Row 1 is a chain.
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
    # Node 2 of row 0 branches off the root, so its parent tap is not node 1.
    assert window_rows[2].tolist() == expected_window[2]
    assert window_rows[2].tolist()[-2] == node_offset


def test_conv_window_rows_keep_batch_order_for_folded_rows():
    # A folded prompt-chunk row (tree_num_nodes == 0) stays linear batch order.
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


def _tree_tables(widths, slots):
    """Packed chained-tree tables: node ``j`` continues ``j - 1``, token-major."""
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
    """State bookkeeping of the AscendC ``Process`` loop."""

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
    flat_state = state.reshape(-1)
    flat_parents = parents.reshape(-1)
    seen = []
    for index, chunk in enumerate(plan.chunks):
        assert chunk.token_indices.shape == (plan.chunk_tokens,)
        assert chunk.token_indices.dtype == torch.long
        assert chunk.out_indices.dtype == torch.long
        assert chunk.state_indices.dtype == torch.int32
        assert chunk.init_state_indices.dtype == torch.int32
        # cu_seqlens: an unused base entry followed by the per-row token count.
        row_lengths = chunk.cu_seqlens[1:]
        assert chunk.cu_seqlens.dtype == torch.int32
        assert bool((row_lengths <= width).all())
        valid = (torch.arange(width).unsqueeze(0) < row_lengths.unsqueeze(1)).reshape(-1)
        padding = ~valid
        # Padding cells are never processed: no page, and their op output is parked
        # past every verified token.
        assert bool((chunk.state_indices[padding] == -1).all())
        assert bool((chunk.init_state_indices[padding] == -1).all())
        assert bool((chunk.out_indices[padding] >= scratch_base).all())
        # A verified cell reads its parent page, writes its own page, outputs its row.
        tokens = chunk.token_indices[valid]
        assert bool((chunk.state_indices[valid] == flat_state[tokens]).all())
        assert bool((chunk.init_state_indices[valid] == flat_parents[tokens]).all())
        assert bool((chunk.out_indices[valid] == tokens).all())
        # Chunk ``index`` owns node columns [index * width, (index + 1) * width).
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
    # Chunk-wide, full, short, empty (padded) and two-chunk rows.
    widths = torch.tensor([width, slots, 7, 0, slots - 3], dtype=torch.int32)
    state, parents = _tree_tables(widths, slots)
    tokens = int(widths.sum())
    values = torch.arange(1, tokens + 1, dtype=torch.float32)
    num_pages = int(widths.numel()) * slots + int(widths.numel()) + 8

    # Reference: the uncapped call the kernel must not be asked to run.
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
    # Rows span both chunks, so a second-chunk first node reloads its parent's state.
    width = GDN_RECURRENT_MAX_VERIFY_TOKENS
    slots = 2 * width
    widths = torch.tensor([slots, slots], dtype=torch.int32)
    state, parents = _tree_tables(widths, slots)
    plan = compute_tree_verify_plan(widths, state, parents, scratch_base=2 * slots)

    assert plan is not None
    for chunk in plan.chunks:
        row_starts = chunk.cu_seqlens.cumsum(0)[:-1]
        for row in range(plan.rows):
            start = int(row_starts[row])
            end = start + int(chunk.cu_seqlens[row + 1])
            assert end > start
            # The row's first cell always carries a real parent page.
            assert int(chunk.init_state_indices[start]) >= 0
            # A continuing node must not reload: the previous token wrote its state.
            for token in range(start + 1, end):
                assert int(chunk.init_state_indices[token]) == int(chunk.state_indices[token - 1])
