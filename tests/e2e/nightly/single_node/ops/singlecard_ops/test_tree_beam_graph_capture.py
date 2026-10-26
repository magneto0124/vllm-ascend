# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU graph replay coverage for the DSpark beam tree-finalization path."""

import torch
import torch.nn as nn

from vllm_ascend.worker.v2.spec_decode.tree.beam import BeamTreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout


class _DraftHeads:
    def __init__(self, vocab_size: int, rank: int, device: torch.device):
        self.embedding = nn.Embedding(vocab_size, rank, device=device)
        self.bias = nn.Linear(rank, vocab_size, bias=False, device=device)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

    def markov_bias(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.bias(embedded)

    @staticmethod
    def map_draft_to_target(token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids


def _tree_layout(rows: int, budget: int, device: torch.device) -> TreeLayout:
    return TreeLayout(
        tokens=torch.empty(rows, budget, dtype=torch.int64, device=device),
        depths=torch.empty(rows, budget, dtype=torch.int32, device=device),
        parents=torch.empty(rows, budget, dtype=torch.int32, device=device),
        num_nodes=torch.empty(rows, dtype=torch.int32, device=device),
        visibility=torch.empty(rows, budget, budget, dtype=torch.bool, device=device),
        first_child=torch.empty(rows, budget + 1, dtype=torch.int32, device=device),
        next_sibling=torch.empty(rows, budget + 1, dtype=torch.int32, device=device),
    )


def _layout_tensors(layout: TreeLayout) -> tuple[torch.Tensor, ...]:
    return (
        layout.tokens,
        layout.depths,
        layout.parents,
        layout.num_nodes,
        layout.visibility,
        layout.first_child,
        layout.next_sibling,
    )


def test_beam_tree_finalize_graph_replays_updated_inputs():
    """Capture logits + Markov correction + beam expansion and replay new data."""
    torch.manual_seed(219)
    device = torch.device("npu")
    rows, spec_num, hidden_size, vocab_size, budget, topk = 2, 3, 16, 32, 5, 3
    heads = _DraftHeads(vocab_size, rank=4, device=device)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device)
    builder = BeamTreeBuilder(budget=budget, topk=topk, draft_model=heads)
    hidden = torch.randn(rows, spec_num, hidden_size, device=device)
    roots = torch.randint(vocab_size, (rows,), device=device)
    graph_layout = _tree_layout(rows, budget, device)
    eager_layout = _tree_layout(rows, budget, device)

    def finalize(layout: TreeLayout, *, graph_safe: bool = True) -> TreeLayout:
        logits = lm_head(hidden.reshape(rows * spec_num, hidden_size))
        logits = logits.view(rows, spec_num, vocab_size)
        return builder.build(
            logits,
            layout,
            root_token_ids=roots,
            graph_safe=graph_safe,
        )

    with torch.inference_mode():
        finalize(graph_layout)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
            finalize(graph_layout)

        hidden.normal_()
        roots.copy_(torch.tensor([7, 19], dtype=roots.dtype, device=device))
        expected = finalize(eager_layout)
        dispatched_layout = _tree_layout(rows, budget, device)
        dispatched = finalize(dispatched_layout, graph_safe=False)
        graph.replay()

    for actual, reference, dispatch in zip(
        _layout_tensors(graph_layout),
        _layout_tensors(expected),
        _layout_tensors(dispatched),
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(reference, dispatch, rtol=0, atol=0)
