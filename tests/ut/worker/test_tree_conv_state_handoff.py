"""Which rows the tree op committed, so the next step re-anchors only those."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_prefix_source,
    compute_tree_resume_column,
    tree_row_mask,
)

_NUM_SPEC = 3
_BUDGET = 4  # wider than the draft chain, which the tree config allows
_WIDTH = 4
_SLOTS = 1 + _BUDGET
# A linear step leaves the next root's window in columns 0..width-2 (identity).
_IDENTITY_SOURCE = [[0, 1, 2]]


@pytest.fixture
def tree_enabled(monkeypatch):
    def _enable(budget=_BUDGET):
        monkeypatch.setattr(
            "vllm_ascend.worker.v2.spec_decode.dflash_tree_spec_enabled",
            lambda vllm_config=None: True,
        )
        monkeypatch.setattr(
            "vllm_ascend.ascend_config.get_ascend_config",
            lambda: SimpleNamespace(
                tree_spec_config=SimpleNamespace(budget=budget, gdn_state_update="snapshot"),
            ),
        )

    return _enable


def _config():
    return SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=_NUM_SPEC),
    )


def _model_state(tree_enabled, num_slots=1):
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridModelState,
    )

    tree_enabled()
    # Object.__new__: only the tree bookkeeping is exercised, all attrs set here.
    state = object.__new__(AscendMambaHybridModelState)
    state.vllm_config = _config()
    state.num_accepted_tokens_gpu = torch.ones((num_slots,), dtype=torch.int32)
    state._tree_prev_path = None
    state._tree_prev_num_sampled = None
    state._tree_resume_column = None
    state._tree_path = None
    state._tree_num_sampled = None
    state._tree_is_tree_row = None
    return state


def _silence_super(monkeypatch):
    """Keep the inherited scatter out: only the bookkeeping is under test."""
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    monkeypatch.setattr(MambaHybridModelState, "postprocess_state", lambda self, *args, **kwargs: None)


def _reanchor_source(state, row: int) -> torch.Tensor:
    """The history columns the next tree step reads for ``row``."""
    prev_path = state._tree_prev_path[row : row + 1].to(torch.int32)
    prev_num_sampled = state._tree_prev_num_sampled[row : row + 1].to(torch.int32)
    return compute_tree_prefix_source(prev_path, prev_num_sampled, _WIDTH)


def _accept(
    state,
    path: torch.Tensor,
    num_sampled: torch.Tensor,
    idx: torch.Tensor,
    is_tree_row: torch.Tensor | None = None,
) -> None:
    """Stage the sampler output of one step, then commit it."""
    if is_tree_row is None:
        is_tree_row = torch.ones_like(num_sampled, dtype=torch.bool)
    state.stage_tree_resume_column(
        compute_tree_resume_column(path, num_sampled),
        path,
        num_sampled,
        is_tree_row=is_tree_row,
    )
    state.postprocess_state(idx, num_sampled)


def test_a_step_without_a_tree_path_does_not_reanchor_the_next_step(tree_enabled, monkeypatch):
    _silence_super(monkeypatch)
    state = _model_state(tree_enabled)
    idx = torch.tensor([0])
    path = torch.tensor([[1, 2, -1, -1]], dtype=torch.int32)

    # Step 1: decoded with the tree, so the next step has a path to re-anchor.
    _accept(state, path, torch.tensor([3], dtype=torch.int32), idx)
    assert state._tree_prev_num_sampled[0].item() == 3

    # Step 2: a prompt-only step, so the stale tree path must be dropped.
    state.postprocess_state(idx, torch.tensor([1], dtype=torch.int32))

    assert _reanchor_source(state, 0).tolist() == _IDENTITY_SOURCE, (
        "a step that did not run the tree op leaves the window the next root continues from in "
        f"columns 0..width-2; the next tree step re-anchored from the stale path instead "
        f"(_tree_prev_num_sampled={state._tree_prev_num_sampled.tolist()})"
    )


def test_a_masked_entry_does_not_clobber_another_slot(tree_enabled, monkeypatch):
    _silence_super(monkeypatch)
    state = _model_state(tree_enabled, num_slots=2)
    state._tree_prev_path = torch.zeros((2, _SLOTS - 1), dtype=torch.int32)
    state._tree_prev_num_sampled = torch.zeros((2,), dtype=torch.int32)
    state._tree_prev_path[0, :2] = torch.tensor([1, 2], dtype=torch.int32)
    state._tree_prev_path[1, :2] = torch.tensor([1, 2], dtype=torch.int32)
    state._tree_prev_num_sampled[:] = 3

    # A masked-only step: no request was committed, so no path may change.
    state.postprocess_state(torch.tensor([-1]), torch.tensor([0], dtype=torch.int32))

    assert state._tree_prev_num_sampled.tolist() == [3, 3], (
        "a ``-1`` entry addresses the *last* slot, so a masked row zeroed a live request's path"
    )
    assert _reanchor_source(state, 0).tolist() == [[1, 2, _WIDTH - 1]]
    assert _reanchor_source(state, 1).tolist() == [[1, 2, _WIDTH - 1]]


def test_a_row_without_draft_tokens_does_not_reanchor_the_next_step(tree_enabled, monkeypatch):
    _silence_super(monkeypatch)
    state = _model_state(tree_enabled)
    idx = torch.tensor([0])
    path = torch.tensor([[1, 2, -1, -1]], dtype=torch.int32)

    _accept(state, path, torch.tensor([3], dtype=torch.int32), idx)

    # Step 2: no draft tokens for this row, so the linear operator ran.
    no_path = torch.full((1, _BUDGET), -1, dtype=torch.int32)
    neutral = torch.tensor([1], dtype=torch.int32)
    _accept(state, no_path, neutral, idx, is_tree_row=torch.tensor([False]))

    assert _reanchor_source(state, 0).tolist() == _IDENTITY_SOURCE, (
        "a row with no draft tokens is numerically identical to a tree row that accepted only its "
        "root, but its conv history was written by the linear operator, so it has to be staged as "
        f"``0`` (_tree_prev_num_sampled={state._tree_prev_num_sampled.tolist()})"
    )


def test_a_root_only_tree_step_still_reanchors_from_its_node(tree_enabled, monkeypatch):
    _silence_super(monkeypatch)
    state = _model_state(tree_enabled)
    idx = torch.tensor([0])
    path = torch.tensor([[1, 2, -1, -1]], dtype=torch.int32)

    _accept(state, path, torch.tensor([3], dtype=torch.int32), idx)

    # Step 2: a tree step that rejected every draft token.
    no_path = torch.full((1, _BUDGET), -1, dtype=torch.int32)
    neutral = torch.tensor([1], dtype=torch.int32)
    _accept(state, no_path, neutral, idx, is_tree_row=torch.tensor([True]))

    assert _reanchor_source(state, 0).tolist() == [[1, 2, _WIDTH - 1]]


def test_graph_replay_reuses_the_captured_treewy_context_dictionary():
    from vllm.config.compilation import CUDAGraphMode
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridModelState,
    )

    def metadata():
        return SimpleNamespace(
            spec_decode_metadata=SimpleNamespace(
                gdn_state_update_method="treewy",
                treewy_contexts={},
            )
        )

    state = object.__new__(AscendMambaHybridModelState)
    state.attn_metadata = {"layer": metadata()}
    state._bind_treewy_contexts(CUDAGraphMode.NONE, 2, 10, for_capture=True)
    captured_contexts = state.attn_metadata["layer"].spec_decode_metadata.treewy_contexts
    captured_contexts["layer"] = object()

    # Runtime metadata is rebuilt each iteration; state-commit needs the captured
    # intermediates.
    state.attn_metadata = {"layer": metadata()}
    state._bind_treewy_contexts(CUDAGraphMode.FULL, 2, 10, for_capture=False)
    replay_contexts = state.attn_metadata["layer"].spec_decode_metadata.treewy_contexts

    assert replay_contexts is captured_contexts
    assert replay_contexts["layer"] is not None
    assert state._treewy_contexts_to_commit == [captured_contexts]


def test_eager_treewy_context_does_not_replace_captured_graph_context():
    from vllm.config.compilation import CUDAGraphMode
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridModelState,
    )

    def metadata():
        return SimpleNamespace(
            spec_decode_metadata=SimpleNamespace(
                gdn_state_update_method="treewy",
                treewy_contexts={},
            )
        )

    state = object.__new__(AscendMambaHybridModelState)
    state.attn_metadata = {"layer": metadata()}
    state._bind_treewy_contexts(CUDAGraphMode.NONE, 2, 10, for_capture=True)
    captured_contexts = state.attn_metadata["layer"].spec_decode_metadata.treewy_contexts

    state.attn_metadata = {"layer": metadata()}
    state._bind_treewy_contexts(CUDAGraphMode.NONE, 2, 10, for_capture=False)
    eager_contexts = state.attn_metadata["layer"].spec_decode_metadata.treewy_contexts

    assert eager_contexts is not captured_contexts
    assert state._treewy_contexts_to_commit == [eager_contexts]


def test_treewy_state_cursor_stays_on_its_single_recurrent_page(monkeypatch):
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridModelState,
    )

    observed = []

    def capture_cursor(self, idx_mapping, num_sampled, num_computed_tokens=None):
        observed.append(num_sampled.clone() if isinstance(num_sampled, torch.Tensor) else num_sampled)

    monkeypatch.setattr(MambaHybridModelState, "postprocess_state", capture_cursor)
    state = object.__new__(AscendMambaHybridModelState)
    state._tree_resume_column = torch.tensor([4], dtype=torch.int32)
    state._tree_path = None
    state._tree_num_sampled = None
    state._tree_is_tree_row = None
    state._tree_prev_path = None
    state._tree_prev_num_sampled = None
    state._treewy_contexts_to_commit = []
    state._treewy_state_layout = True

    state.postprocess_state(torch.tensor([0]), torch.tensor([4], dtype=torch.int32))

    assert observed[0].tolist() == [1]


def test_tree_row_mask_follows_the_verified_token_count():
    assert tree_row_mask(
        torch.tensor([1], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
    ).tolist() == [False]
    # A tree step that accepted the root only: 1 accepted + k rejected.
    assert tree_row_mask(
        torch.tensor([1], dtype=torch.int32), torch.tensor([_NUM_SPEC], dtype=torch.int32)
    ).tolist() == [True]
    # A chunked prefill: the sampler zeroes both counters.
    assert tree_row_mask(
        torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
    ).tolist() == [False]
    # Every draft accepted: k + 1 accepted, nothing rejected.
    assert tree_row_mask(
        torch.tensor([_NUM_SPEC + 1], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)
    ).tolist() == [True]


def test_a_reused_slot_does_not_inherit_the_tree_path(tree_enabled, monkeypatch):
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridModelState,
    )

    state = _model_state(tree_enabled)
    state._tree_prev_path = torch.zeros((1, _SLOTS - 1), dtype=torch.int32)
    state._tree_prev_num_sampled = torch.zeros((1,), dtype=torch.int32)
    state._tree_prev_path[0, :2] = torch.tensor([1, 2], dtype=torch.int32)
    state._tree_prev_num_sampled[0] = 3
    monkeypatch.setattr(MambaHybridModelState, "add_request", lambda self, req_index, new_req_data: None)

    AscendMambaHybridModelState.add_request(state, 0, object())

    assert _reanchor_source(state, 0).tolist() == _IDENTITY_SOURCE, (
        "a new request prefilled into a reused slot must not re-anchor its first tree step from the "
        f"previous occupant's path (_tree_prev_num_sampled={state._tree_prev_num_sampled.tolist()})"
    )


def test_tree_metadata_pads_the_padded_rows_for_the_builder():
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
        AscendMambaHybridAttnMetadata,
    )

    num_reqs = 5
    real_reqs = 2
    metadata = AscendMambaHybridAttnMetadata(
        is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
        # The runner copies the tree layout of the real requests only.
        tree_parents=torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
        tree_num_nodes=torch.tensor([2, 2], dtype=torch.int32),
        tree_prev_path=torch.zeros((num_reqs, _SLOTS - 1), dtype=torch.int32),
        tree_prev_num_sampled=torch.zeros((num_reqs,), dtype=torch.int32),
    )
    builder = object.__new__(AscendGDNAttentionMetadataBuilder)

    kwargs = metadata.get_extra_attn_kwargs(builder, num_reqs)

    for key in ("tree_parents", "tree_num_nodes"):
        assert kwargs[key].shape[0] == num_reqs, (
            f"{key} carries {kwargs[key].shape[0]} rows but the builder selects rows of a padded batch "
            f"({num_reqs}); the padded rows make index_select run past the end of the table"
        )
    assert kwargs["prev_path_node_ids"].shape[0] == num_reqs
    assert kwargs["prev_num_sampled"].shape[0] == num_reqs
    assert real_reqs < num_reqs  # the padding is what makes the tables short
