# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time, tree_timer_enabled

logger = logging.getLogger(__name__)


def _pad_batch_rows(table: torch.Tensor, num_reqs: int) -> torch.Tensor:
    """Extend a per-request draft-tree table to the padded batch size.

    ``0`` makes a padded row linear (``tree_num_nodes == 0``) owning no token.
    """
    if table.shape[0] >= num_reqs:
        return table[:num_reqs]
    filler = table.new_zeros((num_reqs - table.shape[0], *table.shape[1:]))
    return torch.cat((table, filler), dim=0)


@dataclass
class AscendMambaHybridAttnMetadata(MambaHybridAttnMetadata):
    """Mamba metadata that also carries the draft-tree layout for the GDN builder."""

    tree_parents: torch.Tensor | None = None
    tree_num_nodes: torch.Tensor | None = None
    # Prev step's accepted nodes and sampled count, ``[num_reqs]`` each; ``None`` keeps old history.
    tree_prev_path: torch.Tensor | None = None
    tree_prev_num_sampled: torch.Tensor | None = None

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_attn_kwargs(attn_metadata_builder, num_reqs)
        if not kwargs or self.tree_parents is None:
            return kwargs
        # Lazy import to avoid a model-state <-> attention-builder cycle.
        from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

        if not isinstance(attn_metadata_builder, AscendGDNAttentionMetadataBuilder):
            # Mamba2 / short-conv have no draft-tree state rollback.
            return kwargs
        # The runner's tables cover verified requests only; FULL replays padded rows too.
        kwargs["tree_parents"] = _pad_batch_rows(self.tree_parents, num_reqs)
        if self.tree_num_nodes is not None:
            kwargs["tree_num_nodes"] = _pad_batch_rows(self.tree_num_nodes, num_reqs)
        if self.tree_prev_path is not None and self.tree_prev_num_sampled is not None:
            kwargs["prev_path_node_ids"] = self.tree_prev_path[:num_reqs]
            kwargs["prev_num_sampled"] = self.tree_prev_num_sampled[:num_reqs]
        return kwargs


class AscendMambaHybridModelState(MambaHybridModelState, AscendModelState):
    """Mamba state with Ascend-specific attention metadata construction.

    Mamba request lifecycle and cache-state handling are inherited from
    :class:`MambaHybridModelState`. ``AscendModelState`` remains the second
    base so cooperative ``super()`` calls retain the Ascend model-state MRO.
    """

    # GDN state cursor staged by the runner while verifying a draft tree (tree.state_rollback).
    _tree_resume_column: torch.Tensor | None = None
    # Accepted path of the step being committed, per request; ``0`` means no fresh tree path.
    _tree_path: torch.Tensor | None = None
    _tree_num_sampled: torch.Tensor | None = None
    # Rows the tree op actually verified (``tree_row_mask``); the rest came from the linear op.
    _tree_is_tree_row: torch.Tensor | None = None
    _tree_prev_path: torch.Tensor | None = None
    _tree_prev_num_sampled: torch.Tensor | None = None
    _tree_commit_graphs: dict[tuple[int, int], Any]
    _tree_commit_graph_failures: set[tuple[int, int]]
    # Layer set whose TreeWY intermediates were written by a captured graph; ``None`` disables
    # the commit graph for this step (eager verification, uncaptured batch shape, recapture).
    _treewy_commit_token: tuple[dict, int] | None = None
    _treewy_graph_epoch: int = 0
    _treewy_captured_keys: dict[tuple[str, int, int], int]
    _treewy_commit_graphs: dict[tuple, Any]
    _treewy_commit_graph_failures: set[tuple]
    _treewy_commit_buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        """Reset the tree-side bookkeeping of a request slot."""
        super().add_request(req_index, new_req_data)
        # Previous occupant's path: a new request prefills its own conv state, so keep it (``0``).
        if self._tree_prev_num_sampled is not None:
            self._tree_prev_num_sampled[req_index] = 0
        if self._tree_prev_path is not None:
            self._tree_prev_path[req_index] = 0

    def stage_tree_resume_column(
        self,
        resume_column: torch.Tensor,
        path_node_ids: torch.Tensor | None = None,
        num_sampled: torch.Tensor | None = None,
        is_tree_row: torch.Tensor | None = None,
    ) -> None:
        """Stage the accepted path and the recurrent-state resume cursor."""
        self._tree_resume_column = resume_column
        self._tree_path = path_node_ids
        self._tree_num_sampled = num_sampled
        self._tree_is_tree_row = is_tree_row

    def _tree_buffers(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request storage of the previous step's accepted draft path.

        Sized by the tree width; sizing by the sampler path would zero them per step.
        """
        # Imported here to keep the model state import graph flat.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            tree_verify_slot_count,
        )

        slots = tree_verify_slot_count(self.vllm_config)
        assert slots is not None, "draft-tree buffers need an active tree_spec_config"
        path_width = slots - 1
        # Per-request storage: ``idx_mapping`` values address the persistent spec-decode slots.
        num_reqs = self.num_accepted_tokens_gpu.shape[0]
        path = self._tree_prev_path
        if path is None or path.shape != (num_reqs, path_width) or path.device != device:
            path = torch.zeros((num_reqs, path_width), dtype=torch.int32, device=device)
            self._tree_prev_num_sampled = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
            self._tree_prev_path = path
        assert self._tree_prev_num_sampled is not None
        return path, self._tree_prev_num_sampled

    def _run_snapshot_commit_graph(
        self,
        idx_mapping: torch.Tensor,
        resume_column: torch.Tensor,
        path: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> bool:
        """Replay fixed-shape snapshot bookkeeping when all rows are valid.

        The graph covers the three persistent writes; the upstream postprocess stays eager.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            selected_gdn_state_update_name,
        )

        if selected_gdn_state_update_name() != "snapshot":
            return False
        if idx_mapping.ndim != 1 or path.ndim != 2:
            return False
        # PP may put -1 sentinels in idx_mapping: keep PP eager, no per-step D2H check.
        if self.vllm_config.parallel_config.pipeline_parallel_size > 1:
            return False
        rows = idx_mapping.shape[0]
        if rows == 0 or resume_column.shape[0] != rows or path.shape[0] != rows:
            return False
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return False
        if hasattr(torch.npu, "is_current_stream_capturing") and torch.npu.is_current_stream_capturing():
            return False
        graph_key = (rows, path.shape[1])
        graphs = getattr(self, "_tree_commit_graphs", None)
        failures = getattr(self, "_tree_commit_graph_failures", None)
        if graphs is None:
            graphs = self._tree_commit_graphs = {}
            failures = self._tree_commit_graph_failures = set()
        graph = graphs.get(graph_key)
        if graph is None and graph_key in failures:
            return False
        if not hasattr(self, "num_accepted_tokens_gpu"):
            return False
        prev_path, prev_num_sampled = self._tree_buffers(idx_mapping.device)
        if (
            not hasattr(self, "_tree_commit_idx")
            or self._tree_commit_idx.shape[0] < rows
            or self._tree_commit_path.shape[1] != path.shape[1]
        ):
            self._tree_commit_idx = torch.empty_like(idx_mapping)
            self._tree_commit_resume = torch.empty_like(resume_column)
            self._tree_commit_path = torch.empty(
                (rows, path.shape[1]), dtype=prev_path.dtype, device=prev_path.device
            )
            self._tree_commit_num_sampled = torch.empty_like(num_sampled)
        self._tree_commit_idx[:rows].copy_(idx_mapping)
        self._tree_commit_resume[:rows].copy_(resume_column)
        self._tree_commit_path[:rows].copy_(path.to(dtype=prev_path.dtype))
        self._tree_commit_num_sampled[:rows].copy_(num_sampled.to(dtype=prev_num_sampled.dtype))

        def commit_ops() -> None:
            prev_path[:, : path.shape[1]].index_copy_(
                0, self._tree_commit_idx[:rows], self._tree_commit_path[:rows]
            )
            prev_num_sampled.index_copy_(0, self._tree_commit_idx[:rows], self._tree_commit_num_sampled[:rows])
            self.num_accepted_tokens_gpu.index_copy_(0, self._tree_commit_idx[:rows], self._tree_commit_resume[:rows])

        if graph is not None:
            graph.replay()
            return True
        try:
            from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
            from vllm.platforms import current_platform

            validate_cudagraph_capturing_enabled()
            pool = current_platform.get_global_graph_pool()
            with torch.inference_mode():
                commit_ops()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=pool):
                    commit_ops()
            graphs[graph_key] = graph
            logger.info("Captured GDN snapshot state-commit ACLGraph rows=%s", rows)
            return True
        except Exception as exc:
            failures.add(graph_key)
            logger.warning(
                "GDN snapshot state-commit ACLGraph failed for rows=%s; using eager. %s",
                rows,
                exc,
            )
            return False

    def _treewy_commit_inputs(
        self,
        rows: int,
        path_width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stable row-mapped sampler inputs, one pair per batch shape.

        A replay reads them by address, so they are kept per shape instead of reallocated when
        the runner alternates between captured batch sizes.
        """
        buffers = getattr(self, "_treewy_commit_buffers", None)
        if not isinstance(buffers, dict):
            buffers = self._treewy_commit_buffers = {}
        pair = buffers.get((rows, path_width))
        if pair is None or pair[0].device != device:
            pair = (
                torch.empty((rows, path_width), dtype=torch.long, device=device),
                torch.empty((rows,), dtype=torch.int32, device=device),
            )
            buffers[(rows, path_width)] = pair
        return pair

    def _run_treewy_commit_graph(
        self,
        layer_contexts: dict[str, Any],
        path: torch.Tensor,
        num_sampled: torch.Tensor,
        *,
        bookkeeping: tuple[torch.Tensor, torch.Tensor | None] | None = None,
    ) -> bool:
        """Replay one ACLGraph that commits every GDN layer's accepted state.

        Verification runs inside the model graph, so each layer's ``key``, ``pseudo_values``,
        ``log_decay`` and grabbed initial state live at captured addresses that only a replay
        refreshes. One graph therefore turns the eager tail of the decode step -- one
        ``treewy_commit_torch`` per GDN layer, 48 of them on a hybrid model -- into a single
        launch. Only the sampler output changes per step, so it is copied into stable buffers
        first; everything else is read from the capture. When eligible, the same launch also
        saves the request histories and resets the single-page TreeWY cursor.
        """
        token = getattr(self, "_treewy_commit_token", None)
        # Only a layer set produced by a captured graph is replayable; see _bind_treewy_contexts.
        if token is None or token[0] is not layer_contexts or not layer_contexts:
            return False
        if not isinstance(path, torch.Tensor) or not isinstance(num_sampled, torch.Tensor):
            return False
        if path.ndim != 2 or num_sampled.ndim != 1:
            return False
        if num_sampled.device != path.device:
            return False
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return False
        if hasattr(torch.npu, "is_current_stream_capturing") and torch.npu.is_current_stream_capturing():
            return False
        first = next(iter(layer_contexts.values()))
        rows = first.row_widths.shape[0]
        if rows <= 0 or first.key.device != path.device:
            return False

        from vllm_ascend.worker.v2.spec_decode.tree.treewy import (
            treewy_commit_layers_torch,
            treewy_commit_plan,
            treewy_map_commit_rows,
        )

        graphs = getattr(self, "_treewy_commit_graphs", None)
        if graphs is None:
            graphs = self._treewy_commit_graphs = {}
        failures = getattr(self, "_treewy_commit_graph_failures", None)
        if failures is None:
            failures = self._treewy_commit_graph_failures = set()
        epoch = token[1]
        bookkeeping_key = None
        bookkeeping_inputs = None
        if bookkeeping is not None:
            idx_mapping, is_tree_row = bookkeeping
            prev_path, prev_num_sampled = self._tree_buffers(path.device)
            assert path.shape[1] <= prev_path.shape[1], "accepted path exceeds the tree budget"
            bookkeeping_key = (
                idx_mapping.shape[0],
                idx_mapping.dtype,
                prev_path.data_ptr(),
                prev_num_sampled.data_ptr(),
                self.num_accepted_tokens_gpu.data_ptr(),
            )
        graph_key = (id(layer_contexts), epoch, rows, path.shape[1])
        if bookkeeping_key is not None:
            graph_key += (bookkeeping_key,)
        path_buffer, sampled_buffer = self._treewy_commit_inputs(rows, path.shape[1], path.device)
        sampled, accepted_path = treewy_map_commit_rows(
            first.sequence_indices, path, num_sampled, rows
        )
        path_buffer.copy_(accepted_path)
        sampled_buffer.copy_(sampled)

        entry = graphs.get(graph_key)
        if bookkeeping is not None:
            if entry is not None:
                bookkeeping_inputs = entry[5]
            else:
                bookkeeping_inputs = (
                    torch.empty_like(idx_mapping),
                    torch.empty_like(path, dtype=prev_path.dtype),
                    torch.empty_like(num_sampled, dtype=prev_num_sampled.dtype),
                    torch.empty_like(num_sampled, dtype=torch.bool),
                )
            idx_buffer, history_path, history_sampled, history_tree_mask = bookkeeping_inputs
            idx_buffer.copy_(idx_mapping)
            history_path.copy_(path)
            history_sampled.copy_(num_sampled)
            if is_tree_row is None:
                history_tree_mask.fill_(True)
            else:
                history_tree_mask.copy_(is_tree_row)
        fingerprint = (
            len(layer_contexts),
            id(first.key),
            id(first.pseudo_values),
            id(first.initial_state),
        )
        if entry is not None:
            if entry[1] != fingerprint:
                # An eager forward replaced the captured intermediates; the recorded addresses
                # no longer describe this batch, so drop the graph and stay eager.
                graphs.pop(graph_key, None)
                failures.add(graph_key)
                return False
            entry[0].replay()
            return True
        if graph_key in failures:
            return False

        try:
            from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
            from vllm.platforms import current_platform

            validate_cudagraph_capturing_enabled()

            def commit_ops():
                # Rebuilt inside the captured region: the accepted node and weight tensors are
                # derived from the sampler buffers, so a replay must recompute them, not reuse
                # the values this step happened to produce.
                plan = treewy_commit_plan(
                    first.tree_parents,
                    first.tree_num_nodes,
                    first.row_widths,
                    sampled_buffer,
                    path_buffer,
                    first.initial_state.shape[1],
                    first.key.shape[1],
                    first.key.shape[0],
                    ancestors=first.ancestors,
                    device=first.key.device,
                )
                treewy_commit_layers_torch(plan, layer_contexts.values())
                if bookkeeping_inputs is not None:
                    self._commit_treewy_bookkeeping(*bookkeeping_inputs, prev_path, prev_num_sampled)
                return plan

            pool = current_platform.get_global_graph_pool()
            with torch.inference_mode():
                # Capture records without executing, so commit this step eagerly first; it also
                # warms the allocator. A failure below re-runs the eager loop, which is
                # idempotent because the reconstruction reads the verify-time initial state.
                commit_ops()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=pool):
                    plan = commit_ops()
            for stale in [key for key in graphs if key[0] == graph_key[0] and key[1] != epoch]:
                # A recaptured model graph replaced this layer set; its commit graph is dead.
                graphs.pop(stale, None)
            # Keep the plan: it owns the captured intermediates a replay reads.
            graphs[graph_key] = (graph, fingerprint, plan, path_buffer, sampled_buffer, bookkeeping_inputs)
            logger.info(
                "Captured TreeWY state-commit ACLGraph rows=%s layers=%s path=%s bookkeeping=%s",
                rows,
                len(layer_contexts),
                path.shape[1],
                bookkeeping_inputs is not None,
            )
            return True
        except Exception as exc:
            failures.add(graph_key)
            logger.warning(
                "TreeWY state-commit ACLGraph failed for rows=%s layers=%s; using eager. %s",
                rows,
                len(layer_contexts),
                exc,
            )
            return False

    def _commit_treewy_bookkeeping(
        self,
        idx_mapping: torch.Tensor,
        path: torch.Tensor,
        num_sampled: torch.Tensor,
        is_tree_row: torch.Tensor,
        prev_path: torch.Tensor,
        prev_num_sampled: torch.Tensor,
    ) -> None:
        """Fixed-shape writes for valid, unique request indices in a non-PP batch."""
        idx_mapping = idx_mapping.to(torch.int64)
        prev_path[:, : path.shape[1]].index_copy_(0, idx_mapping, path)
        prev_num_sampled.index_copy_(0, idx_mapping, torch.where(is_tree_row, num_sampled, 0))
        self.num_accepted_tokens_gpu.index_fill_(0, idx_mapping, 1)

    def _can_fuse_treewy_bookkeeping(self, idx_mapping: torch.Tensor) -> bool:
        """Leave upstream recovery/alignment and PP sentinel handling on their eager paths."""
        return (
            getattr(self, "_treewy_state_layout", False)
            and self.vllm_config.parallel_config.pipeline_parallel_size == 1
            and self.recoverssm is None
            and not self._align_mode
            and self._tree_path is not None
            and self._tree_num_sampled is not None
            and idx_mapping.ndim == 1
            and idx_mapping.shape[0] > 0
            and self._tree_path.shape[0] == idx_mapping.shape[0]
            and self._tree_num_sampled.shape[0] == idx_mapping.shape[0]
            and idx_mapping.device == self._tree_path.device
            and (self._tree_is_tree_row is None or self._tree_is_tree_row.shape == idx_mapping.shape)
        )

    def _bind_treewy_contexts(
        self,
        cudagraph_mode: CUDAGraphMode,
        num_reqs: int,
        num_tokens: int,
        for_capture: bool,
    ) -> None:
        """Keep graph-captured TreeWY intermediates reachable for state commit.

        Verification runs in the graph, sampling and commit later in Python, so share the dict.
        """
        graph_kind = None
        if for_capture:
            graph_kind = "full"
        elif cudagraph_mode.name.startswith("FULL"):
            graph_kind = "full"
        elif cudagraph_mode.name == "PIECEWISE":
            graph_kind = "piecewise"

        graph_contexts = None
        if graph_kind is not None:
            registry = getattr(self, "_treewy_graph_contexts", None)
            if registry is None:
                registry = {}
                self._treewy_graph_contexts = registry
            key = (graph_kind, num_reqs, num_tokens)
            graph_contexts = registry.setdefault(key, {})
            if for_capture:
                # A graph can be recaptured; drop references to the old static outputs first.
                graph_contexts.clear()
                # Those tensors were just replaced, so any commit graph captured for the
                # previous addresses is stale: a fresh epoch forces a fresh capture.
                epoch = getattr(self, "_treewy_graph_epoch", 0) + 1
                self._treewy_graph_epoch = epoch
                captured_keys = getattr(self, "_treewy_captured_keys", None)
                if captured_keys is None:
                    captured_keys = self._treewy_captured_keys = {}
                captured_keys[key] = epoch

        # Only a shape whose verification was captured feeds the commit graph: an eager forward
        # allocates fresh intermediates whose addresses a replay would not refresh. Piecewise
        # pieces are left eager: their per-piece static inputs are not this layer-set's.
        self._treewy_commit_token = None
        if graph_kind == "full":
            epoch = getattr(self, "_treewy_captured_keys", {}).get(key)
            if epoch is not None:
                self._treewy_commit_token = (graph_contexts, epoch)

        contexts_to_commit = []
        seen_metadata = set()
        for metadata in self.attn_metadata.values():
            if id(metadata) in seen_metadata:
                continue
            seen_metadata.add(id(metadata))
            spec_metadata = getattr(metadata, "spec_decode_metadata", None)
            if getattr(spec_metadata, "gdn_state_update_method", None) != "treewy":
                continue
            if graph_contexts is not None:
                spec_metadata.treewy_contexts = graph_contexts
                contexts = graph_contexts
            else:
                contexts = spec_metadata.treewy_contexts
            if contexts is not None and all(contexts is not existing for existing in contexts_to_commit):
                contexts_to_commit.append(contexts)
        self._treewy_contexts_to_commit = contexts_to_commit

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Includes TreeWY reconstruction, path writes/clears and the accepted-cursor scatter.
        if tree_timer_enabled() and (self._tree_path is not None or self._tree_prev_path is not None):
            with tree_time("gdn_state_commit"):
                self._postprocess_state_impl(idx_mapping, num_sampled, num_computed_tokens)
        else:
            self._postprocess_state_impl(idx_mapping, num_sampled, num_computed_tokens)

    def _postprocess_state_impl(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        contexts_to_commit = getattr(self, "_treewy_contexts_to_commit", None)
        bookkeeping_done = False
        fuse_bookkeeping = self._can_fuse_treewy_bookkeeping(idx_mapping)
        if self._tree_path is not None and self._tree_num_sampled is not None and contexts_to_commit:
            from vllm_ascend.worker.v2.spec_decode.tree.treewy import (
                treewy_commit_contexts_torch,
            )

            with tree_time("gdn_treewy_commit_total"):
                for contexts in contexts_to_commit:
                    if not contexts:
                        continue
                    if not self._run_treewy_commit_graph(
                        contexts,
                        self._tree_path,
                        self._tree_num_sampled,
                        bookkeeping=(
                            (idx_mapping, self._tree_is_tree_row)
                            if fuse_bookkeeping and not bookkeeping_done
                            else None
                        ),
                    ):
                        treewy_commit_contexts_torch(
                            contexts,
                            self._tree_path,
                            self._tree_num_sampled,
                        )
                    elif fuse_bookkeeping:
                        bookkeeping_done = True
        self._treewy_contexts_to_commit = []
        if bookkeeping_done:
            self._tree_resume_column = None
            self._tree_path = None
            self._tree_num_sampled = None
            self._tree_is_tree_row = None
            return

        # Snapshot pages resume at the last accepted node; TreeWY rebuilds it into its sole page.
        if self._tree_resume_column is not None:
            num_sampled = self._tree_resume_column
            self._tree_resume_column = None
        path = self._tree_path
        if path is not None or self._tree_prev_path is not None:
            prev_path, prev_num_sampled = self._tree_buffers(idx_mapping.device)
            if path is not None and self._tree_num_sampled is not None:
                commit_num_sampled = self._tree_num_sampled
                if self._tree_is_tree_row is not None:
                    commit_num_sampled = torch.where(
                        self._tree_is_tree_row.to(torch.bool),
                        commit_num_sampled,
                        torch.zeros_like(commit_num_sampled),
                    )
                if self._run_snapshot_commit_graph(
                    idx_mapping, num_sampled, path, commit_num_sampled
                ):
                    self._tree_path = None
                    self._tree_num_sampled = None
                    self._tree_is_tree_row = None
                    if getattr(self, "_treewy_state_layout", False):
                        num_sampled = (
                            torch.ones_like(num_sampled, dtype=torch.int32)
                            if isinstance(num_sampled, torch.Tensor)
                            else 1
                        )
                    super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
                    return
            # ``idx_mapping`` may carry ``-1``, which indexes the *last* slot: drop those rows.
            valid = idx_mapping >= 0
            rows = idx_mapping[valid]
            # All rows of this batch are re-read next step, so clear every path (not just tree rows).
            prev_num_sampled[rows] = 0
            if path is not None and self._tree_num_sampled is not None:
                # Sampler path is narrower than the buffer, both left aligned (``-1`` pads).
                assert path.shape[1] <= prev_path.shape[1], (
                    f"accepted path of {path.shape[1]} nodes does not fit the tree budget of "
                    f"{prev_path.shape[1]}; tree_spec_config.budget must be >= num_speculative_tokens"
                )
                prev_path[rows, : path.shape[1]] = path[valid].to(torch.int32)
                num_sampled_batch = self._tree_num_sampled
                if self._tree_is_tree_row is not None:
                    # A one-token linear row looks like a root-only tree row: stage ``0``.
                    num_sampled_batch = torch.where(
                        self._tree_is_tree_row.to(torch.bool),
                        num_sampled_batch,
                        torch.zeros_like(num_sampled_batch),
                    )
                prev_num_sampled[rows] = num_sampled_batch[valid].to(torch.int32)
            self._tree_path = None
            self._tree_num_sampled = None
            self._tree_is_tree_row = None
        if getattr(self, "_treewy_state_layout", False):
            # TreeWY owns one committed recurrent page; the upstream cursor must address it.
            if isinstance(num_sampled, torch.Tensor):
                num_sampled = torch.ones_like(num_sampled, dtype=torch.int32)
            else:
                num_sampled = 1
        super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        # Match the upstream Mamba contract without enabling DBO.
        assert ubatch_idx == 0, "DBO is not supported on Ascend"
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(input_batch.is_prefilling_np)

        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[input_batch.idx_mapping]

            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req,
                    -1,
                )
                if cudagraph_mode == CUDAGraphMode.FULL and num_reqs > input_batch.num_reqs and spec_decode_mask.all():
                    padded_query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])[input_batch.num_reqs :]
                    # A draft tree verifies ``1 + budget`` tokens; take the tree width when one is active.
                    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
                        tree_verify_slot_count,
                    )

                    expected_query_len = (
                        tree_verify_slot_count(self.vllm_config) or self.vllm_config.num_speculative_tokens + 1
                    )
                    if np.all(padded_query_lens == expected_query_len):
                        # Full graph capture represents every padded request as
                        # a speculative decode request. Keep replay on the same
                        # pure-spec GDN path so its persistent state metadata is
                        # refreshed before graph replay.
                        num_decode_draft_tokens_np[input_batch.num_reqs :] = padded_query_lens - 1
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        tree_prev_path = None
        tree_prev_num_sampled = None
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            tree_verify_slot_count,
        )

        tree_slots = tree_verify_slot_count(self.vllm_config)
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            selected_gdn_state_update_name,
        )

        self._treewy_state_layout = (
            tree_slots is not None and selected_gdn_state_update_name() == "treewy"
        )
        if tree_slots is not None:
            # Batch-order view of the previous step's paths; 0 leaves conv history untouched.
            path_buffer, num_sampled_buffer = self._tree_buffers(
                self.num_accepted_tokens_gpu.device,
            )
            tree_prev_path = torch.zeros(
                (num_reqs, tree_slots - 1),
                dtype=torch.int32,
                device=path_buffer.device,
            )
            tree_prev_path[: input_batch.num_reqs] = path_buffer[input_batch.idx_mapping]
            tree_prev_num_sampled = torch.zeros(
                (num_reqs,),
                dtype=torch.int32,
                device=num_sampled_buffer.device,
            )
            tree_prev_num_sampled[: input_batch.num_reqs] = num_sampled_buffer[input_batch.idx_mapping]

        model_specific_metadata = AscendMambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            tree_parents=getattr(input_batch, "tree_parents", None),
            tree_num_nodes=getattr(input_batch, "tree_num_nodes", None),
            tree_prev_path=tree_prev_path,
            tree_prev_num_sampled=tree_prev_num_sampled,
        )
        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=model_specific_metadata,
            for_cudagraph_capture=for_capture,
            # Attention must share the tree with GDN, else eager masks causally and FULL keeps dummy.
            tree_visibility=input_batch.tree_visibility,
        )
        self._bind_treewy_contexts(
            cudagraph_mode,
            input_batch.num_reqs_after_padding,
            input_batch.num_tokens_after_padding,
            for_capture,
        )
        return self.attn_metadata
