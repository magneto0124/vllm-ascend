from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout

SUPPORTED_TREE_METHODS: tuple[str, ...] = ("priority", "beam", "prefix")

# tree_spec_config.method -> speculative draft backends that can expand the tree.
# ``beam`` was DSpark-only while it relied on the Markov head to correct each
# parent; a draft without that head (DFlash / DFlash2) expands its depth rows
# directly, so it is allowed here too.
METHOD_REQUIRED_BACKEND: dict[str, tuple[str, ...]] = {
    "priority": ("dflash",),
    "beam": ("dspark", "dflash"),
    "prefix": ("dflash",),
}

_shared_req_arange: torch.Tensor | None = None


def fill_shared_depth_proposal_logits(
    proposal_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    parents: torch.Tensor,
    depths: torch.Tensor,
    num_nodes: int,
) -> None:
    """Write depth-shared draft rows into node-indexed ``proposal_logits``.

    ``proposal_logits`` is ``[R, budget + 1, V]`` (node id axis). Column ``j`` is
    the proposal used to sample children of node ``j``. For parallel-draft
    builders without per-parent correction, every parent at depth ``d`` shares
    ``draft_logits[:, d]``.
    """
    global _shared_req_arange
    num_reqs, spec_num, _vocab = draft_logits.shape
    device = draft_logits.device
    # Cast to proposal dtype (FP32); NPU IndexPut rejects BF16 selfRef.
    draft_f = draft_logits.to(dtype=proposal_logits.dtype)
    proposal_logits.fill_(float("-inf"))
    proposal_logits[:, 0] = draft_f[:, 0]
    if num_nodes <= 0:
        return
    if (
        _shared_req_arange is None
        or _shared_req_arange.numel() < num_reqs
        or _shared_req_arange.device != device
    ):
        _shared_req_arange = torch.arange(num_reqs, device=device, dtype=torch.long)
    req_idx = _shared_req_arange[:num_reqs]
    for slot in range(num_nodes):
        depth = depths[:, slot].to(torch.long)
        parent = parents[:, slot].to(torch.long)
        valid = depth > 0
        row = (depth - 1).clamp(min=0, max=spec_num - 1)
        parent_logits = draft_f[req_idx, row]
        safe_parent = parent.clamp(min=0, max=proposal_logits.shape[1] - 1)
        proposal_logits[req_idx, safe_parent] = torch.where(
            valid.unsqueeze(-1),
            parent_logits,
            proposal_logits[req_idx, safe_parent],
        )
        # Node itself as a future parent: depth-d row proposes depth d+1.
        node = slot + 1
        as_parent = valid & (depth < spec_num)
        child_row = depth.clamp(min=0, max=spec_num - 1)
        node_logits = draft_f[req_idx, child_row]
        proposal_logits[req_idx, node] = torch.where(
            as_parent.unsqueeze(-1),
            node_logits,
            proposal_logits[req_idx, node],
        )


class TreeBuilder(ABC):
    """Base for draft-tree topology builders.

    Axis A (draft backend) is ``required_backend`` / ``speculative_config``.
    Axis B (topology) is the concrete subclass selected by
    ``tree_spec_config.method``.

    ``build()`` writes into ``out`` and returns it. Subclasses ignore kwargs
    they do not need (no probing fallbacks). When ``proposal_logits`` is set
    (``[R, budget + 1, V]``), builders fill the actual expansion distribution
    at each node (raw depth rows, or Domino / Markov corrected).
    """

    required_backend: ClassVar[str]
    method: ClassVar[str]

    def __init__(self, budget: int, topk: int):
        self.budget = budget
        self.topk = topk

    @abstractmethod
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
        """Expand ``draft_logits`` [R, spec_num, vocab] into ``out``.

        ``graph_safe`` selects graph-compatible implementations for operations
        that have an eager-only backend. Builders that cannot support captured
        execution should reject this flag.
        """


def validate_tree_method_backend(method: str, draft_backend: str) -> None:
    """Raise if ``tree_spec_config.method`` cannot run on ``draft_backend``."""
    required = METHOD_REQUIRED_BACKEND.get(method)
    if required is None:
        raise ValueError(
            f"tree_spec_config.method must be one of {SUPPORTED_TREE_METHODS}, "
            f"got {method!r}"
        )
    if draft_backend not in required:
        allowed = " or ".join(repr(backend) for backend in required)
        raise ValueError(
            f"tree_spec_config.method={method!r} requires "
            f"speculative_config method {allowed}, got {draft_backend!r}"
        )


def create_tree_builder(
    method: str,
    budget: int,
    topk: int,
    draft_backend: str,
    *,
    draft_model=None,
    correction_scorer=None,
    prefix_len: int = 0,
    params: dict | None = None,
) -> TreeBuilder:
    """Construct the builder for ``method`` after backend pairing check.

    ``params`` is forwarded to prefix (``candidate_size`` only).
    Priority and beam ignore it.
    """
    validate_tree_method_backend(method, draft_backend)
    if method == "priority":
        from vllm_ascend.worker.v2.spec_decode.tree.priority import (
            PriorityTreeBuilder,
        )

        return PriorityTreeBuilder(budget, topk)
    if method == "beam":
        from vllm_ascend.worker.v2.spec_decode.tree.beam import BeamTreeBuilder

        return BeamTreeBuilder(budget, topk, draft_model=draft_model)
    if method == "prefix":
        from vllm_ascend.worker.v2.spec_decode.tree.prefix import PrefixTreeBuilder

        return PrefixTreeBuilder(
            budget,
            topk,
            correction_scorer=correction_scorer,
            prefix_len=prefix_len,
            params=params,
        )
    raise ValueError(
        f"tree_spec_config.method must be one of {SUPPORTED_TREE_METHODS}, "
        f"got {method!r}"
    )
