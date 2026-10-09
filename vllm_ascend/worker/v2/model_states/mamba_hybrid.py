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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@dataclass
class AscendMambaHybridAttnMetadata(MambaHybridAttnMetadata):
    """Mamba metadata that also carries the draft-tree layout.

    ``AscendGDNAttentionMetadataBuilder`` needs the tree parent ids to point each
    node's recurrent state init at its parent's slot; every other builder ignores
    them.
    """

    tree_parents: torch.Tensor | None = None
    tree_num_nodes: torch.Tensor | None = None
    # Accepted draft nodes of the *previous* step and its sampled token count,
    # both ``[num_reqs]``-sized batches. The short conv needs them to rebuild the
    # committed history from the node the sampler accepted; ``None`` (or
    # ``num_sampled == 0``) keeps the history the previous op left behind.
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
        kwargs["tree_parents"] = self.tree_parents[:num_reqs]
        if self.tree_num_nodes is not None:
            kwargs["tree_num_nodes"] = self.tree_num_nodes[:num_reqs]
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

    # GDN state slot the next step has to resume from, staged by the model
    # runner while verifying a draft tree (see ``tree.state_rollback``); ``None``
    # means the linear-chain convention (``num_sampled``) applies.
    _tree_resume_column: torch.Tensor | None = None
    # Accepted draft path of the step being committed, in batch order, plus its
    # sampled token count; kept per request so the next step can re-anchor the
    # short-conv history (``0`` marks a request without a fresh tree path).
    _tree_path: torch.Tensor | None = None
    _tree_num_sampled: torch.Tensor | None = None
    _tree_prev_path: torch.Tensor | None = None
    _tree_prev_num_sampled: torch.Tensor | None = None

    def stage_tree_resume_column(
        self,
        resume_column: torch.Tensor,
        path_node_ids: torch.Tensor | None = None,
        num_sampled: torch.Tensor | None = None,
    ) -> None:
        """Record the state slot committed at the end of the current step."""
        self._tree_resume_column = resume_column
        self._tree_path = path_node_ids
        self._tree_num_sampled = num_sampled

    def _tree_buffers(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request storage of the previous step's accepted draft path.

        The buffers are sized by the tree width (``budget``, the widest path the
        sampler can hand over) so that they survive the step: the sampler's
        narrower path is copied into their leading columns by
        ``postprocess_state``. Sizing them by that narrower path instead would
        re-create (and zero) them on every step, which silently disables the
        short-conv re-anchor.
        """
        # Imported here to keep the model state import graph flat.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            tree_state_slot_count,
        )

        slots = tree_state_slot_count(self.vllm_config)
        assert slots is not None, "draft-tree buffers need an active tree_spec_config"
        path_width = slots - 1
        # Per-request storage: ``idx_mapping`` values address the request slots
        # of the persistent spec-decode state.
        num_reqs = self.num_accepted_tokens_gpu.shape[0]
        path = self._tree_prev_path
        if path is None or path.shape != (num_reqs, path_width) or path.device != device:
            path = torch.zeros((num_reqs, path_width), dtype=torch.int32, device=device)
            self._tree_prev_num_sampled = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
            self._tree_prev_path = path
        assert self._tree_prev_num_sampled is not None
        return path, self._tree_prev_num_sampled

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Under a draft tree the accepted tokens are a root-to-leaf path instead
        # of a prefix, so the state to resume from is the last accepted node and
        # not ``num_sampled``. It feeds the same scatter as the chain path (state
        # index for the next step, cross-block state copy bias and the align
        # postprocess), so only the value changes.
        if self._tree_resume_column is not None:
            num_sampled = self._tree_resume_column
            self._tree_resume_column = None
        path = self._tree_path
        if path is not None:
            # Requests the next step re-reads are exactly the ones in this
            # batch; invalidate them first so a stale path from an earlier tree
            # cannot re-anchor the conv history, then store the fresh one.
            # ``num_sampled == 0`` keeps the history the previous op left.
            prev_path, prev_num_sampled = self._tree_buffers(idx_mapping.device)
            prev_num_sampled[idx_mapping] = 0
            if self._tree_num_sampled is not None:
                # The sampler's path is only as wide as the draft chain, the
                # buffer as wide as the tree budget; both are left aligned
                # (``-1`` pads the walk), so the leading columns line up.
                assert path.shape[1] <= prev_path.shape[1], (
                    f"accepted path of {path.shape[1]} nodes does not fit the tree budget of "
                    f"{prev_path.shape[1]}; tree_spec_config.budget must be >= num_speculative_tokens"
                )
                prev_path[idx_mapping, : path.shape[1]] = path.to(torch.int32)
                prev_num_sampled[idx_mapping] = self._tree_num_sampled.to(torch.int32)
            self._tree_path = None
            self._tree_num_sampled = None
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
                    # A draft tree verifies ``1 + budget`` tokens per request,
                    # while the speculation config only counts the linear chain
                    # width; take the tree width when one is active.
                    # Imported here to keep the model state import graph flat.
                    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
                        tree_state_slot_count,
                    )

                    expected_query_len = (
                        tree_state_slot_count(self.vllm_config) or self.vllm_config.num_speculative_tokens + 1
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
        # Imported here to keep the model state import graph flat.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            tree_state_slot_count,
        )

        tree_slots = tree_state_slot_count(self.vllm_config)
        if tree_slots is not None:
            # Batch-order view of the paths recorded at the end of the previous
            # step. A padded or path-less request keeps 0, which leaves the
            # committed conv history untouched.
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
            # ``None`` unless the runner populated the draft-tree layout.
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
        )
        return self.attn_metadata
