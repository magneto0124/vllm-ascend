from contextlib import nullcontext

import torch

from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout
from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time


def _markov_correct_logits(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
) -> torch.Tensor:
    """Beam-only one-depth logit correction (optional DSpark Markov bias).

    When ``draft_model`` is set, mirrors ``DSparkSpeculator._sample_sequential``:
    bias from the previous **target-vocab** token via ``markov_embed`` +
    ``markov_bias``. When ``draft_model`` is None, expand the shared-depth base
    logits across the frontier.
    """
    base = depth_logits.unsqueeze(1)
    if draft_model is None:
        return base.expand(-1, frontier_tokens.size(1), -1)
    markov_emb = draft_model.markov_embed(frontier_tokens)
    # NPU unquantized gemm is 2D-only; flatten then restore the frontier layout.
    bias = draft_model.markov_bias(markov_emb.reshape(-1, markov_emb.shape[-1]))
    return base + bias.view(*markov_emb.shape[:-1], -1)


class BeamTreeBuilder(TreeBuilder):
    """Level-wise beam (PCTree) expansion; optional DSpark Markov bias."""

    required_backend = "dspark"
    method = "beam"

    def __init__(self, budget: int, topk: int, draft_model=None):
        super().__init__(budget, topk)
        self.draft_model = draft_model
        self._scratch_reqs = 0
        self._scratch_vocab = 0
        self._scratch_spec = 0
        self._scratch_frozen = False
        self._pool_tokens = None
        self._pool_scores = None
        self._pool_parents = None
        self._pool_depth = None
        self._frontier_tokens = None
        self._frontier_scores = None
        self._frontier_pools = None
        self._next_frontier_tokens = None
        self._next_frontier_scores = None
        self._next_frontier_pools = None
        self._prop_temp = None
        self._remap = None

    def freeze_scratch(self) -> None:
        """Prevent graph replay from growing beam intermediate buffers."""
        self._scratch_frozen = True

    def _ensure_scratch(
        self,
        num_reqs: int,
        vocab: int,
        spec_num: int,
        device: torch.device,
        *,
        need_proposal: bool,
    ) -> None:
        if self._scratch_frozen:
            return
        k = min(self.topk, vocab)
        pool_size = k + max(spec_num - 1, 0) * k * k
        grows = (
            self._pool_tokens is None
            or num_reqs > self._scratch_reqs
            or vocab > self._scratch_vocab
            or spec_num > self._scratch_spec
            or self._pool_tokens.device != device
        )
        if grows:
            self._scratch_reqs = max(self._scratch_reqs, num_reqs)
            self._scratch_vocab = max(self._scratch_vocab, vocab)
            self._scratch_spec = max(self._scratch_spec, spec_num)
            r = self._scratch_reqs
            v = self._scratch_vocab
            s = self._scratch_spec
            k = min(self.topk, v)
            pool_size = k + max(s - 1, 0) * k * k
            self._pool_tokens = torch.empty((r, pool_size), dtype=torch.long, device=device)
            self._pool_scores = torch.empty((r, pool_size), dtype=torch.float32, device=device)
            self._pool_parents = torch.empty((r, pool_size), dtype=torch.long, device=device)
            self._pool_depth = torch.empty((r, pool_size), dtype=torch.long, device=device)
            self._frontier_tokens = torch.empty((r, k), dtype=torch.long, device=device)
            self._frontier_scores = torch.empty((r, k), dtype=torch.float32, device=device)
            self._frontier_pools = torch.empty((r, k), dtype=torch.long, device=device)
            self._next_frontier_tokens = torch.empty((r, k), dtype=torch.long, device=device)
            self._next_frontier_scores = torch.empty((r, k), dtype=torch.float32, device=device)
            self._next_frontier_pools = torch.empty((r, k), dtype=torch.long, device=device)
            self._remap = torch.empty((r, pool_size), dtype=torch.long, device=device)
            self._prop_temp = None
            if need_proposal:
                self._prop_temp = torch.empty(
                    (r, pool_size + 1, v), dtype=torch.float32, device=device
                )
        elif need_proposal and self._prop_temp is None:
            self._prop_temp = torch.empty(
                (self._scratch_reqs, self._pool_tokens.shape[1] + 1, self._scratch_vocab),
                dtype=torch.float32,
                device=device,
            )

    def build(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
        proposal_logits: torch.Tensor | None = None,
        graph_safe: bool = False,
    ) -> TreeLayout:
        tick = (lambda _segment: nullcontext()) if graph_safe else tree_time
        budget = self.budget
        topk = self.topk
        draft_model = self.draft_model
        num_reqs, spec_num, vocab = draft_logits.shape
        k = min(topk, vocab)
        device = draft_logits.device
        need_proposal = proposal_logits is not None
        with tick("beam_scratch"):
            self._ensure_scratch(
                num_reqs,
                vocab,
                spec_num,
                device,
                need_proposal=need_proposal,
            )
        if self._pool_tokens is None:
            raise RuntimeError("beam scratch buffers were not initialized")
        pool_capacity = self._pool_tokens.shape[1]
        if self._scratch_frozen and (
            num_reqs > self._scratch_reqs
            or vocab > self._scratch_vocab
            or spec_num > self._scratch_spec
        ):
            raise ValueError("beam graph scratch capacity is smaller than the requested batch")
        if need_proposal and self._prop_temp is None:
            raise RuntimeError("beam proposal scratch was not initialized before graph replay")

        with tick("beam_prepare_buffers"):
            frontier_tokens = self._frontier_tokens[:num_reqs]
            frontier_scores = self._frontier_scores[:num_reqs]
            frontier_pools = self._frontier_pools[:num_reqs]
            next_frontier_tokens = self._next_frontier_tokens[:num_reqs]
            next_frontier_scores = self._next_frontier_scores[:num_reqs]
            next_frontier_pools = self._next_frontier_pools[:num_reqs]
            frontier_tokens[:, 0] = root_token_ids
            frontier_scores.zero_()
            frontier_pools.fill_(-1)

            pool_tokens = self._pool_tokens[:num_reqs]
            pool_scores = self._pool_scores[:num_reqs]
            pool_parents = self._pool_parents[:num_reqs]
            pool_depth = self._pool_depth[:num_reqs]

            # Temp proposal indexed by root=0 and pool slot i -> i+1. Remapped later.
            # FP32 to match tree_proposal_logits; NPU IndexPut rejects BF16.
            prop_temp = None
            if need_proposal:
                prop_temp = self._prop_temp[:num_reqs, : pool_capacity + 1, :vocab]
                prop_temp.fill_(float("-inf"))

        pool_offset = 0
        for depth in range(spec_num):
            batch = 1 if depth == 0 else k
            frontier = frontier_tokens[:, :batch]
            with tick("beam_markov_logits"):
                step_logits = _markov_correct_logits(
                    draft_model, draft_logits[:, depth], frontier
                )
            if prop_temp is not None:
                with tick("beam_proposal_store"):
                    req_idx = torch.arange(num_reqs, device=device)
                    step_f = step_logits.to(dtype=prop_temp.dtype)
                    for j in range(batch):
                        parent_pool = frontier_pools[:, j]
                        temp_id = torch.where(
                            parent_pool < 0,
                            torch.zeros((), dtype=torch.long, device=device),
                            parent_pool + 1,
                        ).clamp(max=prop_temp.shape[1] - 1)
                        prop_temp[req_idx, temp_id] = step_f[:, j]
            with tick("beam_softmax_topk"):
                log_probs = torch.log_softmax(step_logits.float(), dim=-1)
                top_vals, top_ids = log_probs.topk(k, dim=-1)
                if draft_model is not None:
                    top_ids = draft_model.map_draft_to_target(top_ids)
            with tick("beam_pool_write"):
                candidate_scores = frontier_scores[:, :batch].unsqueeze(-1) + top_vals
                num_candidates = batch * k
                pool_end = pool_offset + num_candidates
                pool_tokens[:, pool_offset:pool_end] = top_ids.reshape(num_reqs, -1)
                pool_scores[:, pool_offset:pool_end] = candidate_scores.reshape(num_reqs, -1)
                pool_parents[:, pool_offset:pool_end] = frontier_pools[:, :batch].repeat_interleave(
                    k, dim=-1
                )
                pool_depth[:, pool_offset:pool_end].fill_(depth)

            with tick("beam_frontier_select"):
                keep = k
                cand_flat = candidate_scores.reshape(num_reqs, -1)
                top_ids_flat = top_ids.reshape(num_reqs, -1)
                selected = cand_flat.topk(keep, dim=-1).indices
                next_frontier_tokens[:, :keep] = torch.gather(top_ids_flat, 1, selected)
                next_frontier_scores[:, :keep] = torch.gather(cand_flat, 1, selected)
                next_frontier_pools[:, :keep] = pool_offset + selected
                frontier_tokens, next_frontier_tokens = next_frontier_tokens, frontier_tokens
                frontier_scores, next_frontier_scores = next_frontier_scores, frontier_scores
                frontier_pools, next_frontier_pools = next_frontier_pools, frontier_pools
                pool_offset = pool_end

        with tick("beam_prune_sort"):
            num_pool = pool_offset
            num_nodes = min(int(budget), num_pool)
            pool_tokens = pool_tokens[:, :num_pool]
            pool_scores = pool_scores[:, :num_pool]
            pool_parents = pool_parents[:, :num_pool]
            pool_depth = pool_depth[:, :num_pool]
            order = torch.argsort(pool_depth, dim=-1, stable=True)
            order = torch.gather(
                order,
                1,
                torch.argsort(
                    torch.gather(pool_scores, 1, order),
                    dim=-1,
                    descending=True,
                    stable=True,
                ),
            )
            packed = order[:, :num_nodes].sort(dim=-1).values

        with tick("beam_remap"):
            remap = self._remap[:num_reqs, :num_pool]
            remap.zero_()
            node_ids = torch.arange(
                1, num_nodes + 1, dtype=torch.long, device=device
            ).unsqueeze(0)
            remap.scatter_(1, packed, node_ids.expand(num_reqs, num_nodes))
            pool_parent_packed = torch.gather(pool_parents, 1, packed)
            non_root = pool_parent_packed >= 0
            raw_parent = torch.gather(remap, 1, pool_parent_packed.clamp(min=0))
            parent_ids = torch.where(
                non_root, raw_parent, torch.zeros_like(raw_parent)
            )

            tokens = torch.gather(pool_tokens, 1, packed)
            depths = (torch.gather(pool_depth, 1, packed) + 1).to(torch.int32)
        with tick("beam_finalize_layout"):
            finalize_tree_layout(out, tokens, depths, parent_ids, num_nodes, force_torch=graph_safe)

        if proposal_logits is not None and prop_temp is not None:
            with tick("beam_proposal_remap"):
                proposal_logits.fill_(float("-inf"))
                proposal_logits[:, 0] = prop_temp[:, 0]
                req_idx = torch.arange(num_reqs, device=device)
                for pool_idx in range(num_pool):
                    final_id = remap[:, pool_idx]
                    has = final_id > 0
                    proposal_logits[req_idx, final_id.clamp(min=0)] = torch.where(
                        has.unsqueeze(-1),
                        prop_temp[:, pool_idx + 1],
                        proposal_logits[req_idx, final_id.clamp(min=0)],
                    )
        return out
