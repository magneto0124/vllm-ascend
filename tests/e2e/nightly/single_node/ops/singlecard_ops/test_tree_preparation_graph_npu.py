# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preparation replay must consume current trees and preserve unrelated rows."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.attention.gdn_recoverssm_metadata import build_tree_replay_paths  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.tree.preparation_graph import TreePreparationGraphs  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata  # noqa: E402


@pytest.fixture
def device():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    return torch.device("npu")


def test_publish_and_gather_refresh_inputs(device):
    graphs = TreePreparationGraphs()
    target = torch.full((3, 4), -1, dtype=torch.int32, device=device)
    expected = target.clone()

    def publish(rows, values):
        target[rows] = values
        return target[rows]

    for row in (1, 2, 0, 1):
        rows = torch.tensor([row], device=device)
        values = torch.full((1, 4), row + 10, dtype=torch.int32, device=device)
        expected[rows] = values
        output = graphs.run("publish", publish, (rows, values))
        torch.testing.assert_close(target, expected)
        torch.testing.assert_close(output, values)
    assert len(graphs.entries) == 1


def test_topology_and_paths_match_eager_on_replay(device):
    graphs = TreePreparationGraphs()

    def prepare(parents, nodes, depths, accepted, previous, sampled):
        tree = SharedGDNTreeMetadata.create(
            parents, nodes, depths, accepted, previous, sampled, 4, use_fused_topology=True
        )
        starts = torch.arange(2, dtype=torch.int32, device=device) * 5
        paths, lengths, owners = build_tree_replay_paths(parents, nodes, starts, 5, 5)
        return tree.init_state_columns, tree.conv_window_rows, tree.conv_prefix_src, paths, lengths, owners

    for parents, depths, nodes, count in (
        ([0, 1, 2, 3], [1, 2, 3, 4], 4, 3),
        ([0, 0, 1, 1], [1, 1, 2, 2], 4, 1),
        ([0, 0, 0, 0], [1, 1, 1, 1], 0, 0),
    ):
        inputs = tuple(
            torch.tensor(value, dtype=torch.int32, device=device)
            for value in ([parents], [nodes], [depths], [count], [[1, 2, 3, -1]], [count])
        )
        expected = prepare(*inputs)
        actual = graphs.run("topology", prepare, inputs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert len(graphs.entries) == 1
