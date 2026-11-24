# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.attention import gdn_recoverssm_metadata as metadata_module
from vllm_ascend.attention.gdn_attn_builder import (
    AscendGDNAttentionMetadataBuilder,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
)
from vllm_ascend.attention.gdn_recoverssm_metadata import (
    AscendGDNRecoverSSMAttentionBackend,
    AscendGDNRecoverSSMCommitMetadata,
    AscendGDNRecoverSSMMetadata,
    AscendGDNRecoverSSMMetadataBuilder,
)
from vllm_ascend.attention.recoverssm_gdn import GDNRecoverSSMCommitContext
from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder
from vllm_ascend.worker.v2.model_states.tree_mamba_hybrid import AscendTreeMambaHybridModelState


@pytest.mark.parametrize("root_only", [True, False])
def test_missing_sampler_path_only_allowed_for_root_decode(root_only):
    context = GDNRecoverSSMCommitContext.__new__(GDNRecoverSSMCommitContext)
    context.commit = Mock()
    meta = AscendGDNRecoverSSMMetadata.__new__(AscendGDNRecoverSSMMetadata)
    meta.recoverssm_context = context
    meta.recoverssm_commit = SimpleNamespace(
        state_indices=torch.tensor([[7]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1 if root_only else 2], dtype=torch.int32),
        request_indices=None,
        block_table=None,
    )
    meta.num_spec_decodes = 1
    meta.node_path_slots = torch.tensor([[0]], dtype=torch.int32)
    meta.tree_root_only = root_only
    meta.spec_sequence_masks = torch.tensor([True])
    sampled = torch.tensor([1], dtype=torch.int32)
    state = SimpleNamespace(
        _tree_resume_state_cursor=None, _tree_path=None, _tree_is_tree_row=None,
        _tree_num_sampled=None, _tree_prev_path=torch.zeros((1, 3), dtype=torch.int32),
        recoverssm=object(), attn_metadata={"layer": meta},
        _postprocess_tree_state_tensors=Mock(),
    )
    mapping = torch.tensor([0], dtype=torch.int32)
    if not root_only:
        with pytest.raises(RuntimeError, match="multi-node tree commit requires"):
            AscendTreeMambaHybridModelState.postprocess_state(state, mapping, sampled)
        context.commit.assert_not_called()
        return
    AscendTreeMambaHybridModelState.postprocess_state(state, mapping, sampled)
    args = state._postprocess_tree_state_tensors.call_args.args
    assert args[2].shape == (1, 0)  # No draft nodes; commit prep adds root 0.
    assert args[3] is sampled  # Conv history must include the verified root.
    assert args[4] is meta.spec_sequence_masks
    meta.commit_recoverssm_state(sampled)
    context.commit.assert_called_once()
    call = context.commit.call_args
    assert call.args[0].tolist() == [1]
    assert call.args[1].tolist() == [7]
    assert call.kwargs["path_record_indices"].tolist() == [[0]]
    assert state._tree_path is None
    assert state._tree_num_sampled is None


def test_tree_commit_groups_share_preparation_and_keep_group_page_ids():
    contexts = [GDNRecoverSSMCommitContext.__new__(GDNRecoverSSMCommitContext) for _ in range(2)]
    metadata = []
    for context, pages, rows in zip(contexts, ([2, 3, 4], [9, 7]), (None, [2, 0])):
        context.commit = Mock()
        metadata.append(
            SimpleNamespace(
                recoverssm_context=context,
                recoverssm_commit=SimpleNamespace(
                    state_indices=torch.tensor(pages, dtype=torch.int32)[:, None],
                    query_start_loc=torch.arange(len(pages) + 1, dtype=torch.int32) * 3,
                    request_indices=None if rows is None else torch.tensor(rows),
                    block_table=None,
                ),
                num_spec_decodes=len(pages),
                tree_full_graph=False,
            )
        )
    path = torch.tensor([[1, 3], [2, 4], [5, 6]], dtype=torch.int32)
    mapping = torch.tensor([0, -1, 2])
    counts = torch.tensor([3, 2, 1], dtype=torch.int32)
    metadata_module.stage_tree_commit_groups(metadata, path, mapping)
    assert metadata[0].tree_commit_batch is metadata[1].tree_commit_batch
    # The upstream loop calls every metadata entry: state must advance once.
    for meta in metadata:
        AscendGDNRecoverSSMMetadata.commit_recoverssm_state(meta, counts)
        assert meta.tree_commit_batch is None
    for context in contexts:
        context.commit.assert_called_once()
    first, second = [context.commit.call_args for context in contexts]
    torch.testing.assert_close(first.args[0], torch.tensor([3, 0, 1], dtype=torch.int32))
    torch.testing.assert_close(second.args[0], torch.tensor([1, 3], dtype=torch.int32))
    torch.testing.assert_close(first.args[1], torch.tensor([2, 3, 4], dtype=torch.int32))
    torch.testing.assert_close(second.args[1], torch.tensor([9, 7], dtype=torch.int32))
    torch.testing.assert_close(
        second.kwargs["path_record_indices"], torch.tensor([[0, 5, 6], [0, 1, 3]], dtype=torch.int32)
    )
    # A later step must consume fresh values, even when metadata is reused.
    metadata_module.stage_tree_commit_groups(metadata, path, mapping)
    for meta in metadata:
        AscendGDNRecoverSSMMetadata.commit_recoverssm_state(meta, counts + 1)
    assert all(context.commit.call_count == 2 for context in contexts)


def test_gdn_builder_uses_gdn_base_without_kda_inheritance():
    assert AscendGDNRecoverSSMMetadataBuilder.__bases__ == (AscendGDNAttentionMetadataBuilder,)
    assert not issubclass(AscendGDNRecoverSSMMetadataBuilder, KimiK3KDAMetadataBuilder)
    assert AscendKDARecoverSSMMetadataBuilder.__bases__ == (KimiK3KDAMetadataBuilder,)
    assert (
        AscendGDNRecoverSSMMetadataBuilder._normalize_graph_padding
        is AscendKDARecoverSSMMetadataBuilder._normalize_graph_padding
    )


def test_gdn_recoverssm_backend_identity():
    assert AscendGDNRecoverSSMMetadataBuilder._cudagraph_support is AttentionCGSupport.UNIFORM_BATCH
    assert AscendGDNRecoverSSMAttentionBackend.get_name() == "GDN_RECOVERSSM"
    assert AscendGDNRecoverSSMAttentionBackend.get_builder_cls() is AscendGDNRecoverSSMMetadataBuilder


def test_tree_graph_staging_refreshes_leaf_owners_and_clears_padding():
    builder = object.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder._graph_tree_paths = torch.empty((4, 2), dtype=torch.int32)
    builder._graph_tree_lengths = torch.empty(4, dtype=torch.int32)
    builder._graph_tree_leaf_owners = torch.empty(4, dtype=torch.int32)
    paths = torch.tensor([[0, -1], [0, 1]], dtype=torch.int32)
    lengths = torch.tensor([1, 2], dtype=torch.int32)
    for owners in (torch.tensor([1, 1]), torch.tensor([0, 1])):
        _, staged_lengths, staged_owners = builder._stage_tree_graph_inputs(paths, lengths, owners, 4)
        assert staged_owners.data_ptr() == builder._graph_tree_leaf_owners.data_ptr()
        assert staged_owners.tolist() == owners.tolist() + [-1, -1]
        assert staged_lengths.tolist() == [1, 2, 0, 0]


@pytest.mark.parametrize(
    ("max_num_seqs", "graph_rows"),
    [(8, 4), (4, 8)],
    ids=["eager_batch_exceeds_graph_capacity", "graph_padding_exceeds_scheduler_capacity"],
)
def test_ones_buffer_covers_scheduler_and_graph_capacity(monkeypatch, max_num_seqs, graph_rows):
    def init_parent(builder, kv_cache_spec, layer_names, vllm_config, device):
        builder.spec_state_indices_tensor = torch.empty((graph_rows, 5), dtype=torch.int32, device=device)

    monkeypatch.setattr(AscendGDNAttentionMetadataBuilder, "__init__", init_parent)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    builder = AscendGDNRecoverSSMMetadataBuilder(SimpleNamespace(), [], config, torch.device("cpu"))

    torch.testing.assert_close(builder._ones[:max_num_seqs], torch.ones(max_num_seqs, dtype=torch.int32))
    torch.testing.assert_close(builder._ones[:graph_rows], torch.ones(graph_rows, dtype=torch.int32))


def test_build_forces_active_decodes_onto_recoverssm_path(monkeypatch):
    builder = AscendGDNRecoverSSMMetadataBuilder.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder.use_spec_decode = True
    builder._capturing = False
    builder._ones = torch.ones(3, dtype=torch.int32)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
    builder.kv_cache_spec = SimpleNamespace(block_size=16)
    builder._get_context = Mock(return_value=object())

    starts = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=3,
        is_prefilling=torch.tensor([False, False, True]),
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        block_table_tensor=torch.tensor([[4], [5], [6]], dtype=torch.int32),
        compute_num_computed_tokens=Mock(),
        pcp_context=object(),
    )
    state_indices = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    conv = GDNSpecCausalConv1dMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        cache_indices=state_indices,
        num_accepted_tokens=torch.tensor([3, 2], dtype=torch.int32),
    )
    base_metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=2,
        num_actual_tokens=2,
        spec_query_start_loc=conv.query_start_loc,
        spec_state_indices_tensor=state_indices,
        num_accepted_tokens=conv.num_accepted_tokens,
    )
    base_metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
        spec_causal_conv1d=conv,
        actual_seq_lengths=None,
    )
    parent_build = Mock(return_value=base_metadata)
    monkeypatch.setattr(AscendGDNAttentionMetadataBuilder, "build", parent_build)

    result = builder.build(
        0,
        common,
        num_accepted_tokens=torch.tensor([3, 2, 1], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([-1, 0, -1], dtype=torch.int32),
    )

    assert isinstance(result, AscendGDNRecoverSSMMetadata)
    assert parent_build.call_args.kwargs["num_decode_draft_tokens_cpu"].tolist() == [1, 1, -1]
    assert result.spec_state_indices_tensor.tolist() == [[10], [20]]
    assert result.num_accepted_tokens.tolist() == [1, 1]
    assert result.spec_decode_metadata.spec_causal_conv1d.cache_indices.tolist() == [10, 20]
    assert result.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens.tolist() == [1, 1]
    assert result.recoverssm_commit.request_indices is None
    assert result.pcp_context is common.pcp_context


def test_commit_uses_gdn_context_and_returns_align_postprocess_metadata():
    context = Mock()
    block_table = torch.tensor([[7], [8]], dtype=torch.int32)
    num_computed_tokens = torch.tensor([16, 32], dtype=torch.int32)
    request_indices = torch.tensor([1, 0], dtype=torch.int32)
    commit = AscendGDNRecoverSSMCommitMetadata(
        state_indices=torch.tensor([[3], [4]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        request_indices=request_indices,
        block_table=block_table,
        num_computed_tokens=num_computed_tokens,
        block_size=64,
    )
    metadata = AscendGDNRecoverSSMMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=3,
        num_actual_tokens=3,
        recoverssm_commit=commit,
        recoverssm_context=context,
    )
    accepted = torch.tensor([2, 1], dtype=torch.int32)

    postprocess = metadata.commit_recoverssm_state(accepted)

    context.commit.assert_called_once()
    args = context.commit.call_args.args
    kwargs = context.commit.call_args.kwargs
    assert args[0] is accepted
    torch.testing.assert_close(args[1], torch.tensor([3, 4], dtype=torch.int32))
    torch.testing.assert_close(args[2], torch.tensor([0, 2, 3], dtype=torch.int32))
    assert kwargs["request_indices"] is request_indices
    assert kwargs["block_table"] is block_table
    assert kwargs["num_computed_tokens"] is num_computed_tokens
    assert kwargs["mamba_block_size"] == 64
    assert postprocess.num_spec_decodes == 2
    assert postprocess.request_indices is request_indices
    assert postprocess.block_table is block_table
    assert postprocess.num_computed_tokens is num_computed_tokens
    assert postprocess.block_size == 64


def test_context_factory_uses_gdn_pages_without_global_kda_patch(monkeypatch):
    builder = AscendGDNRecoverSSMMetadataBuilder.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder._context = None
    builder.layer_names = ["gdn.1", "gdn.2"]
    builder.num_spec = 4
    layers = [object(), object()]
    builder.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=dict(zip(builder.layer_names, layers))),
        scheduler_config=SimpleNamespace(max_num_seqs=12),
    )
    create = Mock(return_value=object())
    monkeypatch.setattr(metadata_module.GDNRecoverSSMCommitContext, "create", create)
    assert builder._get_context() is create.return_value
    assert builder._get_context() is create.return_value
    create.assert_called_once_with(layers, spec_query_len=5, max_num_reqs=12)


@pytest.mark.parametrize("request_id", ["current", "replacement"])
def test_full_tree_refresh_updates_pages_and_preserves_buffers(monkeypatch, request_id):
    builder = object.__new__(AscendGDNRecoverSSMMetadataBuilder)
    builder.vllm_config = SimpleNamespace(
        num_speculative_tokens=2, cache_config=SimpleNamespace(mamba_cache_mode="all")
    )
    builder.kv_cache_spec = SimpleNamespace()
    builder.spec_tree_slots = 3
    builder.spec_conv_width = 2
    builder._full_tree_request_id = "current"
    pages = torch.tensor([[4]], dtype=torch.int32)
    windows = torch.zeros((3, 2), dtype=torch.int32)
    prefix = torch.zeros((1, 1), dtype=torch.int32)
    metadata = SimpleNamespace(
        spec_state_indices_tensor=pages,
        spec_decode_metadata=SimpleNamespace(conv_window_rows=windows, conv_prefix_src=prefix),
        tree_commit_paths=object(), tree_live_sampler_rows=object(),
        recoverssm_commit=SimpleNamespace(block_table=None),
    )
    builder._full_tree_metadata = metadata
    builder._stage_tree_graph_inputs = Mock()
    tree = SimpleNamespace(
        request_id=request_id, slots=3, conv_width=2,
        conv_window_rows=torch.arange(6, dtype=torch.int32).reshape(3, 2),
        conv_prefix_src=torch.tensor([[2]], dtype=torch.int32),
        replay_paths={3: (
            torch.zeros((3, 3), dtype=torch.int32), torch.ones(3, dtype=torch.int32),
            torch.full((3,), 2, dtype=torch.int32),
        )},
    )
    monkeypatch.setattr(metadata_module, "mamba_get_block_table_tensor", lambda table, *args: table)
    result = builder.try_refresh_tree_group(
        torch.tensor([[9]], dtype=torch.int32), torch.tensor([20]), tree, torch.tensor([2])
    )
    if request_id == "replacement":
        assert result is None
        assert pages.tolist() == [[4]]
        builder._stage_tree_graph_inputs.assert_not_called()
    else:
        assert result is metadata
        assert metadata.spec_state_indices_tensor is pages
        assert pages.tolist() == [[9]]
        torch.testing.assert_close(windows, tree.conv_window_rows)
        torch.testing.assert_close(prefix, tree.conv_prefix_src)
        assert metadata.tree_commit_paths is None
        assert metadata.tree_live_sampler_rows is None
        builder._stage_tree_graph_inputs.assert_called_once()
