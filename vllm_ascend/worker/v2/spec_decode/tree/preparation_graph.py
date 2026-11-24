# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private graphs for idempotent tree preparation with fixed tensor shapes."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _PreparationGraph:
    graph: Any
    inputs: tuple[torch.Tensor, ...]
    output: Any


class TreePreparationGraphs:
    """Owner-local cache; callers key every captured non-tensor dependency.

    Only pure computations or idempotent overwrites may be captured here:
    first use runs warmup, capture, and replay with the current valid inputs.
    Outputs remain graph-owned and are overwritten at the next replay.
    """

    def __init__(self):
        self.entries: dict[tuple, _PreparationGraph] = {}

    def run(self, key, operation, inputs):
        key = (key, tuple((t.shape, t.dtype, t.device) for t in inputs))
        entry = self.entries.get(key)
        if entry is None:
            stable = tuple(t.clone(memory_format=torch.contiguous_format) for t in inputs)
            operation(*stable)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = operation(*stable)
            entry = _PreparationGraph(graph, stable, output)
            self.entries[key] = entry
        else:
            for stable, current in zip(entry.inputs, inputs):
                stable.copy_(current, non_blocking=True)
        entry.graph.replay()
        return entry.output
