# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState
from vllm_ascend.worker.v2.spec_decode.tree.treewy import (
    treewy_ancestor_mask,
    treewy_commit_torch,
    treewy_verify_torch,
)


@torch.inference_mode()
def test_treewy_bookkeeping_graph_replays_changed_requests():
    state = object.__new__(AscendMambaHybridModelState)
    device = torch.device("npu")
    state.num_accepted_tokens_gpu = torch.full((5,), 9, dtype=torch.int32, device=device)
    prev_path = torch.full((5, 6), -7, dtype=torch.int32, device=device)
    prev_sampled = torch.full((5,), 9, dtype=torch.int32, device=device)
    idx = torch.tensor([3, 1], device=device)
    path = torch.tensor([[1, 3, -1, -1], [2, -1, -1, -1]], dtype=torch.int32, device=device)
    sampled = torch.tensor([3, 2], dtype=torch.int32, device=device)
    mask = torch.tensor([True, False], device=device)

    def commit():
        state._commit_treewy_bookkeeping(idx, path, sampled, mask, prev_path, prev_sampled)

    for _ in range(3):
        commit()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        commit()

    expected_path = torch.full((5, 6), -7, dtype=torch.int32)
    expected_sampled = torch.full((5,), 9, dtype=torch.int32)
    expected_cursor = torch.full((5,), 9, dtype=torch.int32)
    for step in range(4):
        slots = torch.tensor([step, 4])
        new_path = torch.tensor([[step, -1, -1, -1], [1, 2, -1, -1]], dtype=torch.int32)
        new_sampled = torch.tensor([1, 3], dtype=torch.int32)
        new_mask = torch.tensor([step % 2 == 0, step % 2 != 0])
        idx.copy_(slots)
        path.copy_(new_path)
        sampled.copy_(new_sampled)
        mask.copy_(new_mask)
        # Restore the same sentinels before each replay to also detect inactive-slot writes.
        prev_path.fill_(-7)
        prev_sampled.fill_(9)
        state.num_accepted_tokens_gpu.fill_(9)
        graph.replay()
        expected_path.fill_(-7)
        expected_sampled.fill_(9)
        expected_cursor.fill_(9)
        expected_path[slots, :4] = new_path
        expected_sampled[slots] = torch.where(new_mask, new_sampled, 0)
        expected_cursor[slots] = 1
        torch.testing.assert_close(prev_path.cpu(), expected_path)
        torch.testing.assert_close(prev_sampled.cpu(), expected_sampled)
        torch.testing.assert_close(state.num_accepted_tokens_gpu.cpu(), expected_cursor)


def _native_tree_recurrence(inputs, parents_cpu):
    """Run the production AscendC recurrence one node at a time as an oracle."""
    states, outputs = [], []
    device = inputs["query"].device
    lengths = torch.tensor([0, 1], dtype=torch.int32, device=device)
    page = torch.zeros(1, dtype=torch.int32, device=device)
    for node in range(inputs["query"].shape[0]):
        previous = inputs["initial_state"] if node == 0 else states[int(parents_cpu[node - 1])]
        state = previous.clone()
        outputs.append(
            torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=inputs["query"][node : node + 1].contiguous(),
                key=inputs["key"][node : node + 1].contiguous(),
                value=inputs["value"][node : node + 1].contiguous(),
                g=inputs["g"][node : node + 1].contiguous(),
                beta=inputs["beta"][node : node + 1].contiguous(),
                state=state,
                scale=inputs["scale"],
                actual_seq_lengths=lengths,
                ssm_state_indices=page,
            )
        )
        states.append(state)
    return torch.cat(outputs), states


@pytest.mark.parametrize("slots", [17, 33])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_treewy_matches_native_recurrence_across_steps(slots, use_graph):
    """Replay changed trees/inputs/state; compare to the native decode operator."""
    torch.manual_seed(61)
    device = torch.device("npu")
    inputs = dict(
        query=torch.empty(slots, 2, 128, dtype=torch.bfloat16, device=device),
        key=torch.empty(slots, 2, 128, dtype=torch.bfloat16, device=device),
        value=torch.empty(slots, 4, 128, dtype=torch.bfloat16, device=device),
        g=torch.empty(slots, 4, dtype=torch.float32, device=device),
        beta=torch.empty(slots, 4, dtype=torch.bfloat16, device=device),
        initial_state=torch.randn(1, 4, 128, 128, device=device),
        tree_parents=torch.zeros(1, slots - 1, dtype=torch.int32, device=device),
        row_widths=torch.tensor([slots], dtype=torch.int32, device=device),
        scale=128**-0.5,
        tree_num_nodes=torch.tensor([slots - 1], dtype=torch.int32, device=device),
        ancestors=torch.empty(1, slots, slots, dtype=torch.bool, device=device),
    )
    expected_state = inputs["initial_state"].clone()
    graph = None
    for step in range(6):
        parents_cpu = torch.arange(slots - 1, dtype=torch.int32)
        if step % 2:
            parents_cpu //= 2
        inputs["tree_parents"].copy_(parents_cpu[None])
        # The builder prepares this stable buffer before replay; a stale topology
        # would change the commit after a branch switch.
        inputs["ancestors"].copy_(treewy_ancestor_mask(inputs["tree_parents"], inputs["tree_num_nodes"]))
        for name in ("query", "key"):
            inputs[name].copy_(torch.nn.functional.normalize(torch.randn(slots, 2, 128, device=device), dim=-1))
        inputs["value"].normal_()
        inputs["g"].uniform_(-0.05, 0)
        inputs["beta"].uniform_()
        if use_graph:
            if graph is None:
                for _ in range(3):
                    treewy_verify_torch(**inputs)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    verified = treewy_verify_torch(**inputs)
            graph.replay()
        else:
            verified = treewy_verify_torch(**inputs)
        oracle_inputs = dict(inputs, initial_state=expected_state)
        expected_output, expected_states = _native_tree_recurrence(oracle_inputs, parents_cpu)
        torch.testing.assert_close(verified.output, expected_output, rtol=5e-3, atol=5e-3)
        leaf = 0 if step % 3 == 0 else slots - 1
        path, node = [], leaf
        while node > 0:
            path.append(node)
            node = int(parents_cpu[node - 1])
        path.reverse()
        paths = torch.full((1, slots - 1), -1, dtype=torch.long, device=device)
        paths[0, : len(path)] = torch.tensor(path, dtype=torch.long, device=device)
        sampled = torch.tensor([len(path) + 1], dtype=torch.int32, device=device)
        committed = treewy_commit_torch(
            inputs["initial_state"],
            inputs["key"],
            verified.pseudo_values,
            verified.log_decay,
            inputs["tree_parents"],
            inputs["row_widths"],
            paths,
            sampled,
            inputs["tree_num_nodes"],
            ancestors=verified.ancestors,
        )
        expected_state = expected_states[leaf]
        torch.testing.assert_close(committed, expected_state, rtol=2e-4, atol=2e-4)
        inputs["initial_state"].copy_(committed)
