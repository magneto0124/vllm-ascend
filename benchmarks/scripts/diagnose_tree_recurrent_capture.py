# SPDX-License-Identifier: Apache-2.0
"""Isolate snapshot recurrent GDN eager versus NPU graph replay."""

import argparse
import importlib
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm_ascend.utils import bootstrap_custom_op_env


def compare(label, actual, expected):
    delta = (actual.float() - expected.float()).abs()
    print(
        f"[REC-CHECK] {label} exact={torch.equal(actual, expected)} "
        f"max_abs={delta.max().item():.8g} unequal={(actual != expected).sum().item()}",
        flush=True,
    )


def run_case(state_dtype):
    tokens, key_heads, value_heads, dim = 16, 2, 4, 128
    device = "npu"
    query = torch.nn.functional.normalize(torch.randn(tokens, key_heads, dim, device=device), dim=-1).bfloat16()
    key = torch.nn.functional.normalize(torch.randn_like(query).float(), dim=-1).bfloat16()
    value = torch.randn(tokens, value_heads, dim, device=device, dtype=torch.bfloat16)
    g = -torch.rand(tokens, value_heads, device=device)
    beta = torch.rand(tokens, value_heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(2 * tokens, value_heads, dim, dim, device=device, dtype=state_dtype)
    destinations = torch.arange(tokens, device=device, dtype=torch.int32)
    parents = torch.tensor([tokens - 1] + list(range(tokens - 1)), device=device, dtype=torch.int32)
    lengths = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    accepted = torch.tensor([tokens], device=device, dtype=torch.int32)

    def forward():
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query, key=key, value=value, g=g, beta=beta,
            state=state, scale=dim**-0.5, actual_seq_lengths=lengths,
            ssm_state_indices=destinations, num_accepted_tokens=accepted,
            init_state_indices=parents,
        )

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
    for step in range(3):
        base = (step % 2) * tokens
        destinations.copy_(torch.arange(base, base + tokens, device=device, dtype=torch.int32))
        parents.copy_(torch.tensor(
            [base + tokens - 1] + [base + (node - 1) // 2 for node in range(1, tokens)],
            device=device, dtype=torch.int32,
        ))
        accepted.fill_(step + 1)
        value.normal_()
        initial = torch.randn_like(state)
        state.copy_(initial)
        expected = forward().clone()
        expected_state = state.clone()
        state.copy_(initial)
        graph.replay()
        torch.npu.synchronize()
        label = f"dtype={state_dtype} step={step}"
        compare(label + " output", output, expected)
        compare(label + " state", state, expected_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical NPU index")
    args = parser.parse_args()
    # Match the plugin's initialization order. Importing the extension alone
    # does not select its vendor ACLNN implementation; the built-in implementation
    # may have an older signature without init_state_indices.
    bootstrap_custom_op_env()
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    print(f"[REC-ENV] extension={extension.__file__}", flush=True)
    print(f"[REC-ENV] ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}", flush=True)
    print(f"[REC-ENV] schema={torch.ops._C_ascend.npu_recurrent_gated_delta_rule.default._schema}", flush=True)
    # The runtime can cache operator discovery at device initialization.
    # Publish the vendor path before the first set_device/allocation.
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(1234)
    try:
        for dtype in (torch.bfloat16, torch.float32):
            run_case(dtype)
    except Exception:
        maps = Path("/proc/self/maps")
        if maps.exists():
            libraries = sorted({
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(name in line.lower() for name in ("optiling", "opapi", "cust_op", "vllm_ascend_c"))
            })
            for library in libraries:
                print(f"[REC-LIB] {library}", flush=True)
        raise


if __name__ == "__main__":
    main()
