# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN backend for the shared RecoverSSM metadata lifecycle."""

from dataclasses import dataclass, field, fields

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)
from vllm_ascend.attention.recoverssm_gdn import GDNRecoverSSMCommitContext
from vllm_ascend.attention.recoverssm_metadata import (
    recoverssm_request_indices,
    recoverssm_spec_rows,
)
from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder


def build_tree_replay_paths(
    parents: torch.Tensor,
    num_nodes: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_tokens: int,
    path_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack forward ancestor paths for the nodes actually sent to verify."""
    rows = torch.repeat_interleave(
        torch.arange(parents.shape[0], device=parents.device),
        query_start_loc.diff().long(),
        output_size=num_tokens,
    )
    nodes = torch.arange(num_tokens, device=parents.device) - query_start_loc[rows]
    # Zero-node layouts can carry a linear query. Node 0 always ends the
    # backward walk, including for rows that verify only their leading token.
    chain = torch.arange(parents.shape[1], device=parents.device)
    parents = torch.where(num_nodes[:, None] > 0, parents, chain).long()
    parents = torch.nn.functional.pad(parents, (1, 0), value=-1)
    ancestors = [nodes]
    for _ in range(1, path_width):
        node = ancestors[-1]
        ancestors.append(torch.where(node >= 0, parents[rows, node.clamp_min(0)], -1))
    reverse_paths = torch.stack(ancestors, dim=1)
    lengths = (reverse_paths >= 0).sum(dim=1)
    forward_columns = lengths[:, None] - 1 - torch.arange(path_width, device=parents.device)
    paths = reverse_paths.gather(1, forward_columns.clamp_min(0)).masked_fill(forward_columns < 0, -1)
    return paths.to(torch.int32), lengths.to(torch.int32)


# TODO: Once the pinned vLLM revision includes #61309, might replace this class
# with upstream GDNRecoverSSMCommitMetadata.
# Keep supplying the Ascend convolution descriptors from the builder.
@dataclass
class AscendGDNRecoverSSMCommitMetadata:
    state_indices: torch.Tensor
    query_start_loc: torch.Tensor
    request_indices: torch.Tensor | None
    block_table: torch.Tensor | None
    num_computed_tokens: torch.Tensor | None
    block_size: int | None


# TODO: Once the pinned vLLM revision includes #61309, might replace this class
# with upstream GDNRecoverSSMMetadata.
# Keep supplying the Ascend NPU commit context through the same commit() interface.
@dataclass
class AscendGDNRecoverSSMMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    recoverssm_commit: AscendGDNRecoverSSMCommitMetadata | None = None
    recoverssm_context: GDNRecoverSSMCommitContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    node_path_slots: torch.Tensor | None = None
    node_path_lengths: torch.Tensor | None = None
    tree_full_graph: bool = False
    tree_commit_paths: torch.Tensor | None = None
    tree_live_sampler_rows: torch.Tensor | None = None

    def stage_tree_commit(self, accepted_path: torch.Tensor, idx_mapping: torch.Tensor) -> None:
        """Record sampler-order draft nodes and prepend the verified root."""
        self.tree_commit_paths = torch.nn.functional.pad(accepted_path, (1, 0), value=0)
        self.tree_live_sampler_rows = idx_mapping >= 0

    def commit_recoverssm_state(
        self,
        num_accepted_tokens: torch.Tensor,
    ) -> RecoverSSMPostprocessMetadata | None:
        commit = self.recoverssm_commit
        if commit is None:
            return None
        context = self.recoverssm_context
        assert context is not None
        num_spec_decodes = self.num_spec_decodes
        path_record_indices = None
        tree = self.node_path_slots is not None
        if tree:
            if self.tree_commit_paths is None or self.tree_live_sampler_rows is None:
                raise RuntimeError("GDN RecoverSSM tree commit requires the current sampler path")
            paths = self.tree_commit_paths
            num_accepted_tokens = num_accepted_tokens.masked_fill(~self.tree_live_sampler_rows, 0)
            path_record_indices = (
                paths[:num_spec_decodes]
                if commit.request_indices is None
                else paths.index_select(0, commit.request_indices)
            )
            self.tree_commit_paths = None
            self.tree_live_sampler_rows = None
        state_indices = commit.state_indices[:num_spec_decodes, 0]
        query_start_loc = commit.query_start_loc[: num_spec_decodes + 1]
        if self.tree_full_graph:
            assert tree and path_record_indices is not None
            lengths = (
                num_accepted_tokens[:num_spec_decodes]
                if commit.request_indices is None
                else num_accepted_tokens.index_select(0, commit.request_indices)
            )
            context.commit_tree_graph(lengths, state_indices, query_start_loc, path_record_indices)
            return None
        context.commit(
            num_accepted_tokens,
            state_indices,
            query_start_loc,
            request_indices=commit.request_indices,
            block_table=commit.block_table,
            num_computed_tokens=commit.num_computed_tokens,
            mamba_block_size=commit.block_size,
            commit_conv=not tree,
            path_record_indices=path_record_indices,
        )
        if commit.block_table is None:
            return None
        assert commit.num_computed_tokens is not None
        assert commit.block_size is not None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=num_spec_decodes,
            request_indices=commit.request_indices,
            block_table=commit.block_table,
            num_computed_tokens=commit.num_computed_tokens,
            block_size=commit.block_size,
        )


class AscendGDNRecoverSSMMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """Build GDN RecoverSSM metadata without inheriting the KDA builder."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    _TREE_RECOVERSSM = True

    # TODO: Remove this temporary compatibility binding once the pinned vLLM
    # includes #61309 or equivalent native GDN RecoverSSM metadata support.
    _normalize_graph_padding = AscendKDARecoverSSMMetadataBuilder._normalize_graph_padding

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if self.spec_tree_slots is not None:
            # Both verify and short conv read the same request-level page.
            self.spec_state_indices_tensor = self.spec_state_indices_tensor.new_empty(
                (self.spec_state_indices_tensor.shape[0], 1)
            )
            if self.use_full_cuda_graph:
                path_width = vllm_config.num_speculative_tokens + 1
                self._graph_tree_paths = torch.empty(
                    (self.decode_cudagraph_max_bs, path_width), dtype=torch.int32, device=device
                )
                self._graph_tree_lengths = torch.empty(
                    self.decode_cudagraph_max_bs, dtype=torch.int32, device=device
                )
        self._ones = torch.ones(
            max(vllm_config.scheduler_config.max_num_seqs, self.spec_state_indices_tensor.shape[0]),
            dtype=torch.int32,
            device=self.spec_state_indices_tensor.device,
        )
        self._context: GDNRecoverSSMCommitContext | None = None
        self._capturing = False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
        tree_parents: torch.Tensor | None = None,
        tree_num_nodes: torch.Tensor | None = None,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
        tree_full_graph: bool = False,
    ) -> AscendGDNRecoverSSMMetadata:
        tree_full_graph = self.spec_tree_slots is not None and (tree_full_graph or self._capturing)
        m, num_decode_draft_tokens_cpu = self._normalize_graph_padding(
            common_attn_metadata,
            num_actual_reqs,
            num_decode_draft_tokens_cpu,
        )
        rows = None
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            if self._capturing and (tree_full_graph or m.is_prefilling is None):
                m = m.replace(is_prefilling=torch.zeros(m.num_reqs, dtype=torch.bool))
            if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
                raise ValueError("GDN RecoverSSM needs the CPU is_prefilling mask")
            num_reqs = m.num_reqs if num_actual_reqs is None else num_actual_reqs
            num_decode_draft_tokens_cpu, rows = recoverssm_spec_rows(
                m.is_prefilling,
                m.query_start_loc_cpu,
                num_decode_draft_tokens_cpu,
                num_reqs,
            )

        meta = super().build(
            common_prefix_len,
            m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
            tree_parents=tree_parents,
            tree_num_nodes=tree_num_nodes,
            prev_path_node_ids=prev_path_node_ids,
            prev_num_sampled=prev_num_sampled,
        )
        base = {f.name: getattr(meta, f.name) for f in fields(meta)}
        commit = None
        node_path_slots = node_path_lengths = None
        if meta.num_spec_decodes > 0:
            n = meta.num_spec_decodes
            assert meta.spec_state_indices_tensor is not None
            assert meta.spec_query_start_loc is not None
            base["spec_state_indices_tensor"] = meta.spec_state_indices_tensor[:, :1]
            # The checkpoint already holds the accepted state. Size the
            # one-token counts to the potentially graph-padded indices.
            base["num_accepted_tokens"] = self._ones[: base["spec_state_indices_tensor"].shape[0]]
            if rows is None:
                raise ValueError("GDN RecoverSSM: spec rows without draft counts")
            request_indices = recoverssm_request_indices(rows, n, m.query_start_loc.device)
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            commit = AscendGDNRecoverSSMCommitMetadata(
                state_indices=base["spec_state_indices_tensor"],
                query_start_loc=meta.spec_query_start_loc,
                request_indices=request_indices,
                block_table=m.block_table_tensor if align else None,
                num_computed_tokens=m.compute_num_computed_tokens() if align else None,
                block_size=self.kv_cache_spec.block_size if align else None,
            )

            # The convolution descriptor uses the same running pages. Tree
            # convolution reads the prior path instead of the linear count.
            spec_decode_metadata = getattr(meta, "spec_decode_metadata", None)
            if spec_decode_metadata is not None:
                conv_metadata = spec_decode_metadata.spec_causal_conv1d
                num_rows = conv_metadata.query_start_loc.shape[0] - 1
                pages = base["spec_state_indices_tensor"][:num_rows]
                if self.spec_tree_slots is not None:
                    assert tree_parents is not None and tree_num_nodes is not None
                    row_ids = rows.to(device=tree_parents.device, non_blocking=True)
                    node_path_slots, node_path_lengths = build_tree_replay_paths(
                        tree_parents.index_select(0, row_ids),
                        tree_num_nodes.index_select(0, row_ids),
                        meta.spec_query_start_loc[: n + 1],
                        meta.num_spec_decode_tokens,
                        self.vllm_config.num_speculative_tokens + 1,
                    )
                    if tree_full_graph:
                        input_tokens = getattr(common_attn_metadata, "num_input_tokens", 0) or meta.num_actual_tokens
                        node_path_slots, node_path_lengths = self._stage_tree_graph_inputs(
                            node_path_slots, node_path_lengths, input_tokens
                        )
                        base["num_actual_tokens"] = input_tokens
                conv_metadata.cache_indices = pages[:, 0]
                conv_metadata.num_accepted_tokens = base["num_accepted_tokens"][:num_rows]

        result_meta = AscendGDNRecoverSSMMetadata(
            **base,
            recoverssm_commit=commit,
            recoverssm_context=(self._get_context() if commit is not None else None),
            node_path_slots=node_path_slots,
            node_path_lengths=node_path_lengths,
            tree_full_graph=tree_full_graph and node_path_slots is not None,
        )
        self._attach_ascend_metadata(result_meta, meta, base, common_attn_metadata)
        return result_meta

    def _stage_tree_graph_inputs(
        self, paths: torch.Tensor, lengths: torch.Tensor, input_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_tokens = paths.shape[0]
        if not active_tokens <= input_tokens <= self._graph_tree_paths.shape[0]:
            raise ValueError("GDN RecoverSSM tree exceeds its ACLGraph token capacity.")
        self._graph_tree_paths[:active_tokens].copy_(paths, non_blocking=True)
        self._graph_tree_lengths[:active_tokens].copy_(lengths, non_blocking=True)
        self._graph_tree_paths[active_tokens:input_tokens].fill_(-1)
        self._graph_tree_lengths[active_tokens:input_tokens].zero_()
        return self._graph_tree_paths[:input_tokens], self._graph_tree_lengths[:input_tokens]

    @staticmethod
    def _attach_ascend_metadata(
        result_meta: AscendGDNRecoverSSMMetadata,
        meta: GDNAttentionMetadata,
        base: dict[str, object],
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        # Ascend GDN attaches device-specific convolution and PCP descriptors
        # dynamically rather than declaring them on the upstream dataclass.
        for name, value in vars(meta).items():
            if name not in base:
                setattr(result_meta, name, value)
        result_meta.pcp_context = getattr(common_attn_metadata, "pcp_context", None)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False

    # TODO: Once the pinned vLLM revision includes #61309, reuse upstream context
    # creation and caching. Remove this override once the upstream path can
    # instantiate Ascend's GDNRecoverSSMCommitContext.
    def _get_context(self):
        if self._context is None:
            layers = self.vllm_config.compilation_config.static_forward_context
            self._context = GDNRecoverSSMCommitContext.create(
                [layers[name] for name in self.layer_names],
                spec_query_len=self.spec_tree_slots or self.num_spec + 1,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self._context


class AscendGDNRecoverSSMAttentionBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[AscendGDNRecoverSSMMetadataBuilder]:
        return AscendGDNRecoverSSMMetadataBuilder
