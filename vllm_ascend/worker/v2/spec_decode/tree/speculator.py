import logging
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import graph_capture
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.builder import (
    create_tree_builder,
)
from vllm_ascend.worker.v2.spec_decode.tree.dflash2_selector import (
    DFlash2TreeSelector,
    is_dflash2_architecture,
)
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, ensure_finalize_scratch
from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_stat, tree_time
from vllm_ascend.worker.v2.utils import communicator_switch, torch_cuda_wrapper

logger = logging.getLogger(__name__)


def _hf_dflash_config(hf_config) -> dict:
    raw = getattr(hf_config, "dflash_config", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    return vars(raw)


class AscendTreeSpeculator(AscendDFlashSpeculator):
    """Parallel-draft tree host (DFlash, DFlash2 or DSpark draft) + topology builder.

    Draft model forward is one parallel pass. Tree construction replaces
    per-position single-token sampling via ``self.tree_builder``.

    Domino checkpoints with ``shift_label=true`` use an N-query layout
    (bonus + N-1 masks) and sample the bonus hidden as draft slot 0.
    Vanilla DFlash / priority / beam keep the 1+N mask-only layout (beam uses
    DSpark's own query layout).

    DFlash2 has no draft LM head, so its depth rows are scored by the candidate
    selector (:mod:`tree.dflash2_selector`) instead of ``compute_logits``.

    ``propose()`` still returns flattened non-root tokens so the existing
    runner call stays valid. The tree itself is ``self.tree`` after
    ``propose()`` / ``_generate_draft()``.
    """

    _speculator_name = "DFlashTree"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # DFlashSpeculator.__init__ rejects dflash_config.sample_from_anchor.
        # Clear that so super() can run. DSpark and Domino+shift_label then
        # set the instance flags (N queries, sample the bonus as slot 0).
        draft_hf = vllm_config.speculative_config.draft_model_config.hf_config
        dflash_cfg = getattr(draft_hf, "dflash_config", None)
        if isinstance(dflash_cfg, dict) and dflash_cfg.get("sample_from_anchor"):
            draft_hf.dflash_config = {**dflash_cfg, "sample_from_anchor": False}
        elif dflash_cfg is not None and getattr(dflash_cfg, "sample_from_anchor", False):
            dflash_cfg.sample_from_anchor = False
        super().__init__(vllm_config, device)
        if self.speculative_config.use_dspark():
            self.sample_from_anchor = getattr(draft_hf, "sample_from_anchor", True)
            if self.sample_from_anchor:
                self.num_query_per_req = self.num_speculative_steps
            else:
                self.num_query_per_req = 1 + self.num_speculative_steps
        if self.use_local_argmax_reduction:
            raise ValueError(
                "DFlash tree speculator needs full draft logits; "
                "disable use_local_argmax_reduction."
            )

        tree_cfg = get_ascend_config().tree_spec_config
        self.method = tree_cfg.method
        self.budget = tree_cfg.budget
        self.topk = tree_cfg.topk
        self.params = tree_cfg.params
        self.draft_backend = (
            "dspark" if self.speculative_config.use_dspark() else "dflash"
        )
        # DFlash2 has no draft LM head of its own, so its rows come from the
        # candidate selector instead of compute_logits / compute_draft_logits.
        self.dflash2_selector: DFlash2TreeSelector | None = None
        if is_dflash2_architecture(self.speculative_config.draft_model_config):
            if self.speculative_config.draft_sample_method != "greedy":
                raise NotImplementedError(
                    "DFlash2 draft tree scoring produces rows from the candidate "
                    "selector and is greedy-only; set "
                    "speculative_config.draft_sample_method='greedy'."
                )
            self.dflash2_selector = DFlash2TreeSelector(
                self.speculative_config.draft_model_config,
                self.num_speculative_steps,
                self.vocab_size,
            )
            if self.topk is not None and self.topk > self.dflash2_selector.top_k:
                logger.warning(
                    "tree_spec_config.topk=%s exceeds the DFlash2 selector_top_k=%s; "
                    "the selector only ranks that many children per depth.",
                    self.topk,
                    self.dflash2_selector.top_k,
                )
            logger.info(
                "DFlash2 tree scoring enabled: selector_top_k=%s",
                self.dflash2_selector.top_k,
            )

        self.tree_builder = None
        self._domino_scorer = None
        self._domino_prefix_len = 0
        self._tree_finalized = True
        self.tree_kv_compact = None
        dflash_cfg = _hf_dflash_config(draft_hf)
        self._domino_shift_label = (
            self.method == "prefix"
            and self.draft_backend == "dflash"
            and dflash_cfg.get("projector_type") == "domino"
            and bool(dflash_cfg.get("shift_label", False))
        )
        if self._domino_shift_label:
            self.sample_from_anchor = True
            self.num_query_per_req = self.num_speculative_steps
        # Persistent so FULL replay can update hidden without re-entering Python.
        self._draft_hidden_buf = torch.empty(
            self.max_num_reqs * self.num_query_per_req,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        if self.budget < self.num_speculative_steps:
            raise ValueError(
                "tree_spec_config.budget must be >= num_speculative_tokens "
                f"({self.num_speculative_steps}), got {self.budget}"
            )
        if self.budget > self.draft_tokens.shape[1]:
            self.draft_tokens = torch.zeros(
                self.max_num_reqs,
                self.budget,
                dtype=self.draft_tokens.dtype,
                device=device,
            )

        self.tree_parents = torch.full(
            (self.max_num_reqs, self.budget),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree_depths = torch.zeros(
            (self.max_num_reqs, self.budget),
            dtype=torch.int32,
            device=device,
        )
        self.tree_num_nodes = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.tree_visibility = torch.zeros(
            (self.max_num_reqs, self.budget, self.budget),
            dtype=torch.bool,
            device=device,
        )
        self.tree_first_child = torch.full(
            (self.max_num_reqs, self.budget + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree_next_sibling = torch.full(
            (self.max_num_reqs, self.budget + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree = self._load_layout_from_buffers(self.max_num_reqs)
        self.tree_proposal_logits: torch.Tensor | None = None
        if tree_cfg.rejection_sampler == "magicmtp":
            # Node-indexed proposal for MagicMTP: column j = M_s at node j.
            # Must carry Domino / Markov corrections from the builders.
            self.tree_proposal_logits = torch.full(
                (
                    self.max_num_reqs,
                    self.budget + 1,
                    self.vocab_size,
                ),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )
        logger.info(
            "Tree speculator enabled: method=%s budget=%s topk=%s "
            "depth=%s sample_from_anchor=%s num_query_per_req=%s "
            "draft_backend=%s domino_shift_label=%s magicmtp=%s dflash2_selector=%s",
            self.method,
            self.budget,
            self.topk,
            self.num_speculative_steps,
            self.sample_from_anchor,
            self.num_query_per_req,
            self.draft_backend,
            self._domino_shift_label,
            self.tree_proposal_logits is not None,
            self.dflash2_selector is not None,
        )
        from vllm_ascend.worker.v2.spec_decode.tree.timer import configure_tree_timer
        from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import (
            use_tree_triton,
        )

        self._prefix_graphs: dict[int, Any] = {}
        self._beam_graphs: dict[int, Any] = {}
        # Query compaction runs between the target graph and draft execution; one graph
        # per fixed shape, eager if the target backend changes an address.
        self._query_path_graphs: dict[tuple[int, int, tuple[int, ...]], Any] = {}
        self._query_path_graph_failures: set[tuple[int, int, tuple[int, ...]]] = set()
        self._query_path_node_buffers: dict[tuple[int, int], torch.Tensor] = {}
        timer_backend = "triton" if use_tree_triton() else "torch"
        configure_tree_timer(
            enabled=bool(tree_cfg.enable_timer),
            backend=timer_backend,
            meta={
                "method": self.method,
                "budget": self.budget,
                "topk": self.topk,
                "depth": self.num_speculative_steps,
                "rejection_sampler": tree_cfg.rejection_sampler,
                "enable_triton": bool(tree_cfg.enable_triton),
                "gdn_state_update": tree_cfg.gdn_state_update,
            },
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        if self.draft_backend == "dspark":
            from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

            return load_dspark_model(target_model, self.vllm_config)
        return super().load_draft_model(target_model, target_attn_layer_names)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._bind_correction_heads(target_model)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.dflash2_selector is not None:
            # The selector top-k's the full vocabulary and all-gathers across TP,
            # which the linear DFlash2 speculator also refuses to capture. Accept
            # either enforce_eager knob; raise only when neither asked for eager.
            model_eager = bool(
                getattr(self.vllm_config.model_config, "enforce_eager", False)
            )
            if not self.speculative_config.enforce_eager and not model_eager:
                raise NotImplementedError(
                    "dflash2 draft tree scoring does not support graph mode yet; "
                    "enable enforce_eager in speculative_config (or on the model)."
                )
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)

    def _bind_correction_heads(self, target_model: nn.Module) -> None:
        """Resolve Domino heads (prefix) then construct the tree builder."""
        from vllm_ascend.worker.v2.spec_decode.tree.prefix import (
            DominoCorrectionScorer,
        )

        model = self.model
        draft_model = None
        if (
            self.method == "beam"
            and hasattr(model, "markov_embed")
            and hasattr(model, "markov_bias")
        ):
            # Only DSpark carries the Markov head that corrects each beam parent.
            # A draft without it (DFlash / DFlash2) expands the depth rows as-is.
            draft_model = model
        if (
            self.method == "prefix"
            and getattr(model, "projector_type", None) == "domino"
        ):
            self._domino_prefix_len = int(model.pure_draft_prefix_len)
            language_model = (
                target_model.get_language_model()
                if hasattr(target_model, "get_language_model")
                else target_model
            )
            self._domino_scorer = DominoCorrectionScorer(model, language_model)

        self.tree_builder = create_tree_builder(
            method=self.method,
            budget=self.budget,
            topk=self.topk,
            draft_backend=self.draft_backend,
            draft_model=draft_model,
            correction_scorer=self._domino_scorer,
            prefix_len=self._domino_prefix_len,
            params=self.params,
        )
        logger.info(
            "Tree correction heads: markov=%s domino=%s prefix_len=%s "
            "shift_label=%s",
            draft_model is not None,
            self._domino_scorer is not None,
            self._domino_prefix_len,
            self._domino_shift_label,
        )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        dp_sync: Any = None,
    ) -> torch.Tensor:
        path_node_ids = getattr(input_batch, "path_node_ids", None)
        if path_node_ids is not None and not dummy_run:
            from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time

            tree_q = 1 + (get_ascend_config().tree_spec_config.budget or 0)
            # Skip when this step is not a full tree verify (e.g. 16 tokens
            # padded to the 17-token FULL gear at max_seq_len).
            if last_hidden_states.shape[0] >= path_node_ids.shape[0] * tree_q:
                tensors = [last_hidden_states]
                if aux_hidden_states:
                    tensors.extend(aux_hidden_states)
                with tree_time("compact_query_path"):
                    if not self._run_query_path_graph(
                        tensors,
                        input_batch.query_start_loc,
                        path_node_ids,
                        input_batch.positions,
                    ):
                        compact_tree_query_along_path(
                            tensors,
                            input_batch.query_start_loc,
                            path_node_ids,
                            linearize_positions=input_batch.positions,
                        )
        self._tree_finalized = False
        tokens = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile=is_profile,
            dp_sync=dp_sync,
        )
        # FULL replay only runs draft forward; tree finalization has its own graph.
        if not dummy_run:
            self._finalize_tree(input_batch.num_reqs)
        return tokens

    def _run_query_path_graph(self, tensors, query_start_loc, path_node_ids, positions) -> bool:
        """Capture/replay fixed-shape accepted-path query compaction.

        The graph captures the real target output buffers; a backend that supplies a
        new address falls back to eager compaction for that step.
        """
        if not self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            return False
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return False
        if hasattr(torch.npu, "is_current_stream_capturing") and torch.npu.is_current_stream_capturing():
            return False
        rows, _ = path_node_ids.shape
        token_count = tensors[0].shape[0]
        path_key = (rows, path_node_ids.shape[1])
        path_buffer = self._query_path_node_buffers.get(path_key)
        if path_buffer is None:
            path_buffer = torch.empty_like(path_node_ids)
            self._query_path_node_buffers[path_key] = path_buffer
        path_buffer.copy_(path_node_ids)
        tensor_layout = tuple((tuple(t.shape), t.stride(), t.dtype) for t in tensors)
        key = (
            rows,
            token_count,
            tensor_layout,
            tuple(t.data_ptr() for t in (*tensors, query_start_loc, path_buffer, positions)),
        )
        graph = self._query_path_graphs.get(key)
        if graph is not None:
            graph.replay()
            return True
        if key in self._query_path_graph_failures:
            return False
        # Capturing executes the in-place compaction once; back up live outputs for fallback.
        backups = [tensor.clone() for tensor in tensors]
        positions_backup = positions.clone()
        try:
            from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
            from vllm.platforms import current_platform

            validate_cudagraph_capturing_enabled()
            pool = current_platform.get_global_graph_pool()
            with torch.inference_mode():
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=pool):
                    compact_tree_query_along_path(tensors, query_start_loc, path_buffer, positions)
            self._query_path_graphs[key] = graph
            logger.info("Captured tree query-path ACLGraph rows=%s tokens=%s", rows, token_count)
            # The capture itself performed this step's in-place reordering.
            return True
        except Exception as exc:
            for tensor, backup in zip(tensors, backups):
                tensor.copy_(backup)
            positions.copy_(positions_backup)
            self._query_path_graph_failures.add(key)
            logger.warning(
                "Tree query-path ACLGraph capture failed for rows=%s tokens=%s; using eager. %s",
                rows,
                token_count,
                exc,
            )
            return False

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        self.query_cudagraph_manager.capture(
            self._run_draft_forward,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )
        self._capture_prefix_graphs()
        self._capture_beam_graphs()
        if self.tree_kv_compact is not None:
            self.tree_kv_compact.capture(self._draft_capture_num_reqs())

    def _draft_capture_num_reqs(self) -> list[int]:
        manager = self.query_cudagraph_manager
        if manager is None:
            return []
        seen: set[int] = set()
        qlen = max(int(self.num_query_per_req), 1)
        for descs in manager._capture_descs.values():
            for desc in descs:
                n = desc.num_reqs
                if not n:
                    n = desc.num_tokens // qlen
                if n:
                    seen.add(int(n))
        return sorted(seen)

    def _prepare_prefix_graph_scratch(self) -> None:
        from vllm_ascend.worker.v2.spec_decode.tree.layout import (
            ensure_finalize_scratch,
        )

        gru_hidden_dim = 0
        if self._domino_scorer is not None:
            gru_hidden_dim = self._domino_scorer.gru_hidden_dim
        self.tree_builder._ensure_scratch(
            self.max_num_reqs,
            self.vocab_size,
            self.num_speculative_steps,
            self.device,
            need_proposal=self.tree_proposal_logits is not None,
            gru_hidden_dim=gru_hidden_dim,
            dtype=self.dtype,
        )
        self.tree_builder.freeze_scratch()
        ensure_finalize_scratch(self.max_num_reqs, self.device)

    def _draft_logits_rows(
        self,
        sample_hidden_states: torch.Tensor,
        num_reqs: int,
        root_token_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return draft rows or DFlash2 candidate IDs and edge scores.

        Three drafts, three sources: DSpark projects with
        ``compute_draft_logits``, plain DFlash uses its own draft head through
        ``compute_logits``, and DFlash2 replaces both with the candidate
        selector, whose edge scores retain their predecessor axis.
        """
        if self.dflash2_selector is not None:
            return self.dflash2_selector.score(
                self.model, sample_hidden_states, root_token_ids, num_reqs
            )
        if self.method == "beam":
            logits = self.model.compute_draft_logits(sample_hidden_states)
        else:
            logits = self.model.compute_logits(sample_hidden_states)
        return logits.view(num_reqs, self.num_speculative_steps, -1)

    def _run_prefix_finalize(self, num_reqs: int, *, graph_safe: bool) -> None:
        """Draft rows + prefix build; captured as one NPUGraph per num_reqs."""
        hidden = self._draft_hidden_buf
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = hidden[self.sample_indices[:num_sample]]
        layout = self._load_layout_from_buffers(num_reqs)
        nqp = self.num_query_per_req
        root_token_ids = self.input_buffers.input_ids[: num_reqs * nqp].view(
            num_reqs, nqp
        )[:, 0]
        logits = self._draft_logits_rows(sample_hidden_states, num_reqs, root_token_ids)
        proposal = None
        if self.tree_proposal_logits is not None:
            proposal = self.tree_proposal_logits[:num_reqs, : self.budget + 1]
        self.tree = self.tree_builder.build(
            logits,
            layout,
            root_token_ids=root_token_ids,
            draft_hidden=sample_hidden_states.view(
                num_reqs, self.num_speculative_steps, -1
            ),
            proposal_logits=proposal,
            graph_safe=graph_safe,
        )

    def _capture_prefix_graphs(self) -> None:
        if self.method != "prefix" or self.tree_builder is None:
            return
        if self.dflash2_selector is not None:
            logger.info(
                "Prefix tree capture skipped: DFlash2 draft rows come from the "
                "candidate selector and run eagerly."
            )
            return
        manager = self.query_cudagraph_manager
        if manager is None or not manager.needs_capture():
            return
        sizes = self._draft_capture_num_reqs()
        if not sizes:
            return
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return
        from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
        from vllm.platforms import current_platform

        self._prepare_prefix_graph_scratch()
        pool = current_platform.get_global_graph_pool()
        for num_reqs in sizes:
            try:
                validate_cudagraph_capturing_enabled()
                with torch.inference_mode():
                    self._run_prefix_finalize(num_reqs, graph_safe=True)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        self._run_prefix_finalize(num_reqs, graph_safe=True)
                self._prefix_graphs[num_reqs] = graph
                logger.info("Captured prefix tree ACLGraph num_reqs=%s", num_reqs)
            except Exception as exc:
                logger.warning(
                    "Prefix tree ACLGraph capture failed for num_reqs=%s; "
                    "eager fallback for this size. %s",
                    num_reqs,
                    exc,
                )

    def _run_beam_finalize(self, num_reqs: int, *, graph_safe: bool) -> None:
        """Project logits, apply Markov corrections and write the persistent tree.

        All inputs are views refreshed before replay; intermediates stay in the graph pool.
        """
        num_sample = num_reqs * self.num_speculative_steps
        with nullcontext() if graph_safe else tree_time("beam_gather_hidden"):
            hidden = self._draft_hidden_buf[self.sample_indices[:num_sample]]
        nqp = self.num_query_per_req
        roots = self.input_buffers.input_ids[: num_reqs * nqp].view(num_reqs, nqp)[:, 0]
        with nullcontext() if graph_safe else tree_time("beam_compute_logits"):
            draft_rows = self._draft_logits_rows(hidden, num_reqs, roots)
        selector_kwargs = {}
        if self.dflash2_selector is not None:
            candidate_ids, selector_scores = draft_rows
            logits = None
            selector_kwargs = {
                "candidate_ids": candidate_ids,
                "selector_scores": selector_scores,
                "candidate_vocab_size": self.dflash2_selector.vocab_size,
            }
        else:
            logits = draft_rows
        proposal = None
        if self.tree_proposal_logits is not None:
            proposal = self.tree_proposal_logits[:num_reqs, : self.budget + 1]
        self.tree = self.tree_builder.build(
            logits,
            self._load_layout_from_buffers(num_reqs),
            root_token_ids=roots,
            draft_hidden=hidden.view(num_reqs, self.num_speculative_steps, -1),
            proposal_logits=proposal,
            graph_safe=graph_safe,
            **selector_kwargs,
        )

    def _capture_beam_graphs(self) -> None:
        if self.method != "beam" or self.tree_builder is None:
            return
        if self.dflash2_selector is not None:
            logger.info(
                "Beam tree capture skipped: DFlash2 draft rows come from the "
                "candidate selector and run eagerly."
            )
            return
        manager = self.query_cudagraph_manager
        if manager is None or not manager.needs_capture():
            logger.warning(
                "Beam tree capture skipped: no draft capture descriptors; "
                "num_query_per_req=%s max_num_reqs=%s. Check draft enforce_eager "
                "and cudagraph_capture_sizes against the actual draft query length.",
                self.num_query_per_req,
                self.max_num_reqs,
            )
            return
        sizes = self._draft_capture_num_reqs()
        sizes = sorted({size for size in sizes if 0 < size <= self.max_num_reqs})
        logger.info(
            "Beam tree capture plan: num_query_per_req=%s max_num_reqs=%s request_gears=%s",
            self.num_query_per_req,
            self.max_num_reqs,
            sizes,
        )
        if not sizes or not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return
        from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
        from vllm.platforms import current_platform
        from vllm_ascend.worker.v2.aclgraph_utils import model_capture_wrapper

        # Capture only counts with draft graph descriptors; other counts use eager
        # finalization rather than multiplying startup capture cost.
        max_reqs = max(sizes)
        self.tree_builder._ensure_scratch(
            self.max_num_reqs,
            self.vocab_size,
            self.num_speculative_steps,
            self.device,
            need_proposal=self.tree_proposal_logits is not None,
        )
        self.tree_builder.freeze_scratch()
        ensure_finalize_scratch(self.max_num_reqs, self.device)
        # Startup capture has no live proposals: use valid dummy tokens and finite hidden
        # states, including rows the last (possibly smaller) capture never touched.
        self._draft_hidden_buf.zero_()
        self.sample_indices.zero_()
        self.input_buffers.input_ids[: max_reqs * self.num_query_per_req].zero_()
        pool = current_platform.get_global_graph_pool()
        with (
            torch_cuda_wrapper(),
            communicator_switch(),
            model_capture_wrapper(self, False),
            graph_capture(device=self.device),
        ):
            for num_reqs in sizes:
                try:
                    validate_cudagraph_capturing_enabled()
                    with torch.inference_mode():
                        self._run_beam_finalize(num_reqs, graph_safe=True)
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph, pool=pool):
                            self._run_beam_finalize(num_reqs, graph_safe=True)
                    self._beam_graphs[num_reqs] = graph
                    logger.info("Captured beam tree ACLGraph num_reqs=%s", num_reqs)
                except Exception as exc:
                    # Rank-local fallback would let one rank replay collectives while
                    # another runs eager; fail startup under TP instead.
                    if self.vllm_config.parallel_config.tensor_parallel_size > 1:
                        raise RuntimeError(
                            "Beam tree ACLGraph capture failed with tensor parallelism "
                            f"for num_reqs={num_reqs}"
                        ) from exc
                    logger.warning(
                        "Beam tree ACLGraph capture failed for num_reqs=%s; "
                        "eager fallback for this size. %s",
                        num_reqs,
                        exc,
                    )
        # Warmup/capture wrote dummy trees. The next proposal must finalize again.
        self._tree_finalized = False

    def _run_draft_forward(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Draft model forward only; tree finalization uses a separate ACLGraph."""
        from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time

        with tree_time("draft_model_forward"):
            hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        with tree_time("draft_hidden_copy"):
            self._draft_hidden_buf[:num_tokens_padded].copy_(hidden)
        self._tree_finalized = False

    def _finalize_tree(self, num_reqs: int) -> None:
        if self.tree_builder is None or self._tree_finalized:
            return
        if self.method == "beam":
            graph = self._beam_graphs.get(num_reqs)
            with tree_time("build_draft_tree"):
                if graph is None:
                    tree_stat(f"beam_eager_reqs_{num_reqs}")
                    with tree_time("beam_eager_finalize"):
                        self._run_beam_finalize(num_reqs, graph_safe=False)
                else:
                    tree_stat(f"beam_graph_replay_reqs_{num_reqs}")
                    with tree_time("beam_graph_replay"):
                        graph.replay()
            if graph is not None:
                with tree_time("beam_bind_layout"):
                    self.tree = self._load_layout_from_buffers(num_reqs)
            self._tree_finalized = True
            return

        if self.method == "prefix":
            graph = self._prefix_graphs.get(num_reqs)
            if graph is not None:
                graph.replay()
                self.tree = self._load_layout_from_buffers(num_reqs)
                self._tree_finalized = True
                return
            self._run_prefix_finalize(num_reqs, graph_safe=False)
            self._tree_finalized = True
            return

        hidden = self._draft_hidden_buf
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = hidden[self.sample_indices[:num_sample]]
        layout = self._load_layout_from_buffers(num_reqs)
        nqp = self.num_query_per_req
        root_token_ids = self.input_buffers.input_ids[: num_reqs * nqp].view(
            num_reqs, nqp
        )[:, 0]
        logits = self._draft_logits_rows(sample_hidden_states, num_reqs, root_token_ids)
        proposal = None
        if self.tree_proposal_logits is not None:
            proposal = self.tree_proposal_logits[:num_reqs, : self.budget + 1]
        build_kwargs = dict(
            root_token_ids=root_token_ids,
            draft_hidden=sample_hidden_states.view(
                num_reqs, self.num_speculative_steps, -1
            ),
            proposal_logits=proposal,
        )
        with tree_time("build_draft_tree"):
            self.tree = self.tree_builder.build(logits, layout, **build_kwargs)
        self._tree_finalized = True

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._run_draft_forward(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._finalize_tree(num_reqs)

    def _load_layout_from_buffers(self, num_reqs: int) -> TreeLayout:
        """Views into persistent buffers."""
        budget = self.budget
        return TreeLayout(
            tokens=self.draft_tokens[:num_reqs, :budget],
            depths=self.tree_depths[:num_reqs, :budget],
            parents=self.tree_parents[:num_reqs, :budget],
            num_nodes=self.tree_num_nodes[:num_reqs],
            visibility=self.tree_visibility[:num_reqs, :budget, :budget],
            first_child=self.tree_first_child[:num_reqs, : budget + 1],
            next_sibling=self.tree_next_sibling[:num_reqs, : budget + 1],
        )
