# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/aclgraph_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import set_graph_params, update_full_graph_params
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.tree_graph_diag import TreeGraphAudit
from vllm_ascend.worker.v2.utils import communicator_switch


# Temporary diagnostics for the target/draft graph mismatch investigation.
# Keep this constant local to the module so it can be removed after the issue
# is diagnosed without introducing a user-facing environment variable.
TREE_DIAG_ENABLED = True


def tree_diag_claim(seen: set[Any], key: Any) -> bool:
    if not TREE_DIAG_ENABLED or key in seen:
        return False
    seen.add(key)
    return True


def tree_diag_once(seen: set[Any], key: Any, msg: str, *args: Any) -> None:
    if not tree_diag_claim(seen, key):
        return
    logger.warning("[TREE-DIAG] " + msg, *args)


def _diag_descs(capture_descs: dict) -> list[tuple[Any, ...]]:
    return [
        (
            str(mode),
            getattr(desc, "num_tokens", None),
            getattr(desc, "num_reqs", None),
            getattr(desc, "uniform_token_count", None),
            getattr(desc, "max_query_len", None),
            getattr(desc, "num_active_loras", None),
        )
        for mode, descs in capture_descs.items()
        for desc in descs
    ]


def _prepare_pcp_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    model_state: ModelState,
    input_buffers: InputBuffers,
    _block_tables: BlockTables,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    full_cudagraph: bool,
    max_query_len: int | None = None,
    *,
    pcp_manager: Any,
) -> cudagraph_utils.AttentionState:
    """Build graph inputs with the same PCP-local layout used on replay."""
    # vLLM #53515 passes PCP-local input buffers into graph capture, so the
    # dummy batch must not be partitioned a second time. vLLM #53869
    # supplies capture-only PCP metadata instead. The block tables must
    # retain the same PCP-local backing that runtime prepare_attn updates,
    # because the SFA full graph cannot rebind their captured pointer.
    # The Ascend dummy carries the seq_lens_np/attn_state views consumed
    # by Ascend metadata builders and doubles as the capture-time PCP
    # global batch (is_dummy=True).
    input_batch = AscendInputBatch.make_dummy(  # type: ignore[call-arg]
        num_reqs, num_tokens, input_buffers, max_query_len=max_query_len
    )
    input_block_tables = pcp_manager.get_dummy_block_tables(num_reqs)
    slot_mappings = pcp_manager.get_dummy_slot_mappings(num_tokens)
    slot_mappings_by_layer = cudagraph_utils.build_slot_mappings_by_layer(slot_mappings, kv_cache_config)

    attn_metadata = model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        input_block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=full_cudagraph,
    )
    return cudagraph_utils.AttentionState(attn_metadata, slot_mappings_by_layer)


def collect_sorted_captured_token_sizes(capture_descs: dict) -> list[int]:
    """Collect the actual per-graph token counts that will be captured.

    With speculative decoding under FULL_DECODE_ONLY, each raw
    ``cudagraph_capture_size`` is rounded up to a multiple of
    ``decode_query_len`` (see ``CudaGraphManager._init_candidates``), so the
    real graph sizes differ from ``compilation_config.cudagraph_capture_sizes``.
    The attention backend keys its per-size graph params (events/handles/...)
    by these rounded token counts, so they must be derived from the actual
    capture descriptors, not the raw config sizes.
    """
    return sorted({desc.num_tokens for descs in capture_descs.values() for desc in descs})


def _get_graph_update_backend(
    attn_groups: list[list[AttentionGroup]],
) -> type[AttentionBackend]:
    for groups in attn_groups:
        for group in groups:
            backend = group.backend
            if backend.get_impl_cls() is not None:
                return backend
    raise RuntimeError("No executable attention backend is available for full-graph parameter updates.")


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    def __init__(  # type: ignore[misc]
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        model_runner: Any,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
    ):
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
        )
        self.breakable_cg_runner: BreakableACLGraphWrapper | None = None
        self.model_runner = model_runner
        self.tree_graph_audit = TreeGraphAudit()
        self.update_stream = self.model_runner.update_stream
        self._tree_diag_seen: set[Any] = set()
        self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
        if super().needs_capture():
            set_graph_params(self.capture_sizes)
        if TREE_DIAG_ENABLED:
            logger.warning(
                "[TREE-DIAG] graph init: model=%s mode=%s decode_query_len=%s "
                "configured_sizes=%s resolved_sizes=%s max_num_reqs=%s "
                "descs=%s needs_capture=%s",
                getattr(getattr(vllm_config, "model_config", None), "model", "unknown"),
                cudagraph_mode,
                decode_query_len,
                vllm_config.compilation_config.cudagraph_capture_sizes,
                self.capture_sizes,
                getattr(self, "max_num_reqs", None),
                _diag_descs(self._capture_descs),
                super().needs_capture(),
            )

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
    ) -> BatchExecutionDescriptor:
        desc = super().dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len=max_query_len,
        )
        if TREE_DIAG_ENABLED:
            model_name = getattr(getattr(self.vllm_config, "model_config", None), "model", "unknown")
            key = (
                "dispatch",
                str(model_name),
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                max_query_len,
                str(desc.cg_mode),
                desc.num_tokens,
            )
            tree_diag_once(
                self._tree_diag_seen,
                key,
                "dispatch: model=%s runtime(tokens=%s reqs=%s uniform=%s "
                "max_query=%s loras=%s) -> desc(mode=%s tokens=%s reqs=%s "
                "uniform=%s max_query=%s) candidates=%s%s",
                model_name,
                num_tokens,
                num_reqs,
                uniform_token_count,
                max_query_len,
                num_active_loras,
                desc.cg_mode,
                desc.num_tokens,
                desc.num_reqs,
                desc.uniform_token_count,
                desc.max_query_len,
                _diag_descs(self._capture_descs),
                " FALLBACK_EAGER" if desc.cg_mode == CUDAGraphMode.NONE else "",
            )
        return desc

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableACLGraphWrapper(model, self.vllm_config)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        self.tree_graph_audit.report(self.model_runner.model_state.attn_metadata, logger.warning)
        logger.info_once("run_fullgraph with num_tokens=%s", num_tokens)
        assert self.update_stream is not None
        self.update_stream.wait_stream(torch.npu.current_stream())
        ret = super().run_fullgraph(desc)

        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.model_runner.dp_size], num_tokens)
        # sfa_v1.py:AscendSFABackend.get_impl_cls reaches
        # sfa_cp.py:resolve_sfa_impl, whose SFA CP selector reads the current
        # ModelConfig. Publish the target config because set_forward_context()
        # does not update it.
        # TODO: Remove this explicit current-config scope once ACL graph replay
        # passes VllmConfig directly through the graph-update interfaces.
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                self.model_runner.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ),
        ):
            forward_context = get_forward_context()
            attn_backend = _get_graph_update_backend(self.model_runner.attn_groups)
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
            )
        return ret

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
        # vLLM #53869 supplies PCP slot mappings during graph capture.
        pcp_manager: Any = None,
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        if TREE_DIAG_ENABLED:
            logger.warning(
                "[TREE-DIAG] capture begin: descs=%s", _diag_descs(self._capture_descs)
            )
        model = ModelWithContext(model, tree_graph_audit=self.tree_graph_audit)
        pcp_manager = getattr(self.model_runner, "pcp_manager", None)
        if pcp_manager is not None:
            cudagraph_utils.prepare_inputs_to_capture = partial(
                _prepare_pcp_inputs_to_capture,
                pcp_manager=pcp_manager,
            )
        try:
            with communicator_switch():
                # vLLM #53869 added pcp_manager to ModelCudaGraphManager.capture on
                # main; v0.28.0 still uses the older signature without that kwarg.
                if not vllm_version_is("0.28.0"):
                    result = super().capture(
                        model,
                        model_state,
                        input_buffers,
                        intermediate_tensors,
                        block_tables,
                        attn_groups,
                        kv_cache_config,
                        pcp_manager=pcp_manager,
                        has_lora=has_lora,
                        use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                        lora_capture_hook=lora_capture_hook,
                        progress_bar_desc=progress_bar_desc,
                    )
                else:
                    result = super().capture(
                        model,
                        model_state,
                        input_buffers,
                        intermediate_tensors,
                        block_tables,
                        attn_groups,
                        kv_cache_config,
                        has_lora=has_lora,
                        use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                        lora_capture_hook=lora_capture_hook,
                        progress_bar_desc=progress_bar_desc,
                    )
        except Exception:
            logger.exception("[TREE-DIAG] capture failed")
            raise
        if TREE_DIAG_ENABLED:
            logger.warning(
                "[TREE-DIAG] capture done: graphs=%s captured_tokens=%s",
                len(self.graphs),
                self.captured_token_counts(),
            )
        return result


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context.
    so we can inherit vllm's CudaGraphManager._capture_full_graph.
    """

    def __init__(self, original_model, is_draft_model=False, is_draft_model_prefill=False, tree_graph_audit=None):
        super().__init__()
        self.original_model = original_model
        self.is_draft_model = is_draft_model
        self.is_draft_model_prefill = is_draft_model_prefill
        self.tree_graph_audit = tree_graph_audit

    def forward(self, *args, **kwargs):
        forward_context = get_forward_context()
        # In warmup phase, capturing=False by default.
        # when capturing, we need to set capturing=True in forward context.
        is_capturing = (
            torch.npu.is_current_stream_capturing()
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        )
        _EXTRA_CTX.capturing = is_capturing
        if is_capturing and self.tree_graph_audit is not None:
            self.tree_graph_audit.record(forward_context.attn_metadata)
        if self.is_draft_model:
            _EXTRA_CTX.is_draft_model = True
        if self.is_draft_model_prefill:
            _EXTRA_CTX.is_draft_model_prefill = True

        if is_capturing:
            logger.warning(
                "[TREE-DIAG] model forward capture begin: role=%s tokens=%s runtime_mode=%s",
                "draft" if self.is_draft_model else "target",
                getattr(forward_context, "num_tokens", None),
                forward_context.cudagraph_runtime_mode,
            )
        try:
            output = self.original_model(*args, **kwargs)
        except Exception:
            if is_capturing:
                logger.exception(
                    "[TREE-DIAG] model forward capture failed: role=%s tokens=%s "
                    "runtime_mode=%s",
                    "draft" if self.is_draft_model else "target",
                    getattr(forward_context, "num_tokens", None),
                    forward_context.cudagraph_runtime_mode,
                )
            raise
        if is_capturing:
            logger.warning(
                "[TREE-DIAG] model forward capture done: role=%s tokens=%s runtime_mode=%s",
                "draft" if self.is_draft_model else "target",
                getattr(forward_context, "num_tokens", None),
                forward_context.cudagraph_runtime_mode,
            )
        return output

    def get_original_model(self):
        return self.original_model

    def compute_logits(self, hidden_states: torch.Tensor):
        # draft model has `compute_logits`, which is not in ModelWithContext
        return self.original_model.compute_logits(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor):
        return self.original_model.compute_draft_logits(hidden_states)

    def markov_embed(self, token_ids: torch.Tensor):
        return self.original_model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor):
        return self.original_model.markov_bias(markov_embed)

    def map_draft_to_target(self, draft_ids: torch.Tensor):
        return self.original_model.map_draft_to_target(draft_ids)

    def embed_input_ids(self, *args, **kwargs):
        return self.original_model.embed_input_ids(*args, **kwargs)


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
