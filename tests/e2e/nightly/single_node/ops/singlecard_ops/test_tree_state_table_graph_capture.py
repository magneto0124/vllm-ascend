# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU correctness and timing checks for snapshot state-table ACLGraphs."""

import time

import pytest
import torch

from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_conv_rows,
    compute_tree_init_state_indices,
    compute_tree_prefix_source,
)


def _table_builder(rows: int, slots: int, width: int):
    builder = object.__new__(AscendGDNAttentionMetadataBuilder)
    builder.spec_tree_slots = slots
    builder.spec_conv_width = width
    builder._tree_state_table_graphs = {}
    builder._tree_state_table_graph_failures = set()
    shapes = {
        "spec_tree_parents": (rows, slots - 1),
        "spec_tree_num_nodes": (rows,),
        "spec_init_state_indices": (rows, slots),
        "spec_conv_window_rows": (rows * slots, width),
        "spec_conv_prefix_src": (rows, width - 1),
        "spec_tree_table_state_pages": (rows, slots),
        "spec_tree_table_resume_slots": (rows,),
        "spec_tree_table_prev_path": (rows, slots - 1),
        "spec_tree_table_prev_num_sampled": (rows,),
    }
    for name, shape in shapes.items():
        setattr(builder, name, torch.zeros(shape, device="npu", dtype=torch.int32))
    return builder


def _mean_ms(fn, repeats=30):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


@pytest.mark.parametrize("rows", [1, 2, 8])
@torch.inference_mode()
def test_snapshot_state_table_graph_updated_inputs(rows):
    slots, width = 16, 4
    builder = _table_builder(rows, slots, width)
    indices = torch.arange(rows, device="npu", dtype=torch.long)
    states = torch.arange(rows * slots, device="npu", dtype=torch.int32).reshape(rows, slots)
    accepted = torch.ones(rows, device="npu", dtype=torch.int32)
    paths = torch.arange(1, slots, device="npu", dtype=torch.int32).expand(rows, -1).contiguous()
    sampled = torch.zeros(rows, device="npu", dtype=torch.int32)
    args = (states, accepted, indices, paths, sampled)
    outputs = builder._run_snapshot_tree_state_table_graph(*args, capture=True)
    assert outputs is not None, "State-table graph capture failed; eager fallback is not a passing graph test."
    assert rows in builder._tree_state_table_graphs
    pointers = tuple(output.data_ptr() for output in outputs)

    for step in range(4):
        # Change every runtime input after capture: owners, paths, topology, rows.
        parents = torch.arange(slots - 1, dtype=torch.int32, device="npu")
        parents = parents // (step + 1)
        builder.spec_tree_parents.copy_(parents.expand(rows, -1))
        builder.spec_tree_num_nodes.fill_(slots - 1 if step % 2 else 0)
        states.add_(rows * slots)
        accepted.fill_(1 + step)
        sampled.fill_(step)
        paths.copy_(torch.arange(1, slots, dtype=torch.int32, device="npu").flip(0).expand(rows, -1))
        reference = (
            compute_tree_init_state_indices(states, accepted, builder.spec_tree_parents, builder.spec_tree_num_nodes),
            compute_tree_conv_rows(builder.spec_tree_parents, builder.spec_tree_num_nodes, width, rows * (width - 1)),
            compute_tree_prefix_source(paths, sampled, width),
        )
        for output in outputs:
            output.fill_(-99)
        outputs = builder._run_snapshot_tree_state_table_graph(*args, capture=False)
        assert outputs is not None
        assert tuple(output.data_ptr() for output in outputs) == pointers
        for actual, expected in zip(outputs, reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def eager():
        builder._copy_tree_state_table_inputs(*args)
        builder._compute_snapshot_tree_state_tables(rows)

    def replay():
        builder._run_snapshot_tree_state_table_graph(*args, capture=False)

    # Timings include the stable-input refresh; this is a microbenchmark only.
    eager_ms = _mean_ms(eager)
    graph_ms = _mean_ms(replay)
    print(f"[GDN-TABLE] rows={rows} exact=True eager_ms={eager_ms:.3f} graph_ms={graph_ms:.3f}", flush=True)
