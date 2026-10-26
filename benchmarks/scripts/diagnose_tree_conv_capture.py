# SPDX-License-Identifier: Apache-2.0
"""Compare the actual tree conv function in eager and NPU graph replay.

Both runs share inputs and conv state, so no weights or dataset are needed.
Loads only that function's AST (no model/plugin init); conv capture only, no recurrent GDN.
"""

import argparse
import ast
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load_conv(dim_first):
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/gdn.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "tree_causal_conv1d")
    namespace = {"torch": torch, "is_conv_state_dim_first": lambda: dim_first}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["tree_causal_conv1d"]


def windows(parents, width):
    result = []
    for node in range(len(parents)):
        history = []
        cursor = node
        while cursor >= 0:
            history.append(width - 1 + cursor)
            cursor = parents[cursor]
        history.reverse()
        result.append((list(range(width - 1)) + history)[-width:])
    return result


def compare(label, actual, expected):
    delta = (actual.float() - expected.float()).abs()
    print(
        f"[CONV-CHECK] {label} exact={torch.equal(actual, expected)} "
        f"max_abs={delta.max().item():.8g} unequal={(actual != expected).sum().item()}",
        flush=True,
    )


def run_case(dim_first, channels):
    conv = load_conv(dim_first)
    slots, width = 16, 4
    state_shape = (2, channels, width - 1 + slots) if dim_first else (2, width - 1 + slots, channels)
    state = torch.randn(state_shape, device="npu", dtype=torch.bfloat16)
    x = torch.randn((slots, channels), device="npu", dtype=torch.bfloat16)
    weight = torch.randn((channels, width), device="npu", dtype=torch.bfloat16)
    bias = torch.randn((channels,), device="npu", dtype=torch.bfloat16)
    pages = torch.zeros((1, slots), device="npu", dtype=torch.int32)
    starts = torch.tensor([0, slots], device="npu", dtype=torch.int32)
    table = torch.tensor(windows([-1] + list(range(slots - 1)), width), device="npu", dtype=torch.int32)
    prefix = torch.tensor([[0, 1, 2]], device="npu", dtype=torch.int32)

    def forward():
        return conv(x, weight, state, pages, starts, table, prefix, bias, True)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            forward()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = forward()
    torch.npu.synchronize()
    # Mutate page, tree topology, history columns and activations after capture, same buffers.
    parents = [-1] + [(node - 1) // 2 for node in range(1, slots)]
    for step in range(3):
        pages.fill_(step % 2)
        table.copy_(torch.tensor(windows(parents, width), device="npu", dtype=torch.int32))
        prefix.copy_(torch.tensor([[0, 3, 4 + step]], device="npu", dtype=torch.int32))
        x.normal_()
        initial = torch.randn_like(state)
        state.copy_(initial)
        expected = forward().clone()
        expected_state = state.clone()
        state.copy_(initial)
        graph.replay()
        torch.npu.synchronize()
        label = f"dim_first={dim_first} step={step}"
        compare(label + " output", output, expected)
        compare(label + " state", state, expected_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical NPU index")
    parser.add_argument("--channels", type=int, default=128)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(1234)
    for dim_first in (False, True):
        run_case(dim_first, args.channels)


if __name__ == "__main__":
    main()
