# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark one layer's TreeWY tensor math, excluding conv/sampling/scatter.

Run with --device npu on Ascend; CPU timings are not NPU speedup estimates.
Use --baseline /path/to/old_treewy.py to compare another source revision.
--count-ops counts ATen dispatches, not device kernel launches.
--accuracy-steps checks repeated commits against independent FP64 CPU recurrence.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


class DispatchCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.scalar_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        self.scalar_reads += func == torch.ops.aten._local_scalar_dense.default
        return func(*args, **(kwargs or {}))


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, args, device):
    for _ in range(args.warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(args.iterations):
        fn()
    synchronize(device)
    result = {"mean_ms": (time.perf_counter() - start) * 1000 / args.iterations}
    if args.count_ops:
        with DispatchCounter() as counter:
            fn()
        synchronize(device)
        result.update(aten_dispatches=counter.count, host_scalar_reads=counter.scalar_reads)
    return result


def make_inputs(args, device):
    dtype = getattr(torch, args.dtype)
    tokens = args.rows * args.slots
    query = torch.nn.functional.normalize(torch.randn(tokens, args.key_heads, args.key_dim, device=device), dim=-1)
    key = torch.nn.functional.normalize(torch.randn_like(query), dim=-1)
    parent_row = torch.arange(args.slots - 1, dtype=torch.int32)
    if args.topology == "branch":
        parent_row //= 2
    path = []
    node = args.slots - 1
    while node > 0:
        path.append(node)
        node = int(parent_row[node - 1])
    path.reverse()
    paths = torch.full((args.rows, args.slots - 1), -1, dtype=torch.long)
    paths[:, : len(path)] = torch.tensor(path, dtype=torch.long)
    inputs = dict(
        query=query.to(dtype),
        key=key.to(dtype),
        value=torch.randn(tokens, args.value_heads, args.value_dim, dtype=dtype, device=device),
        g=-torch.rand(tokens, args.value_heads, device=device),
        beta=torch.rand(tokens, args.value_heads, dtype=dtype, device=device),
        initial_state=torch.randn(args.rows, args.value_heads, args.value_dim, args.key_dim, device=device),
        tree_parents=parent_row.repeat(args.rows, 1).to(device),
        row_widths=torch.full((args.rows,), args.slots, dtype=torch.int32, device=device),
        scale=args.key_dim**-0.5,
        tree_num_nodes=torch.full((args.rows,), args.slots - 1, dtype=torch.int32, device=device),
    )
    return inputs, paths.to(device), torch.full((args.rows,), len(path) + 1, dtype=torch.int32, device=device)


def run(module, inputs, paths, sampled, args, device):
    verify_inputs = dict(inputs)
    prepare_topology = getattr(module, "treewy_ancestor_mask", None)
    topology_result = None
    if prepare_topology is not None:
        # Production metadata refreshes this buffer once per batch and shares
        # it across GDN layers. Report this cost separately from each layer.
        def build_topology():
            return prepare_topology(inputs["tree_parents"], inputs["tree_num_nodes"])

        verify_inputs["ancestors"] = build_topology()
        topology_result = measure(build_topology, args, device)

    def verify():
        return module.treewy_verify_torch(**verify_inputs)

    verified = verify()
    commit_kwargs = {}
    if getattr(verified, "ancestors", None) is not None:
        commit_kwargs["ancestors"] = verified.ancestors

    def commit():
        return module.treewy_commit_torch(
            inputs["initial_state"],
            inputs["key"],
            verified.pseudo_values,
            verified.log_decay,
            inputs["tree_parents"],
            inputs["row_widths"],
            paths,
            sampled,
            inputs["tree_num_nodes"],
            **commit_kwargs,
        )

    committed = commit()
    for tensor in (verified.output, verified.pseudo_values, committed):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("Non-finite output or state; refusing to report performance.")
    timings = {"verify": measure(verify, args, device), "commit": measure(commit, args, device)}
    if topology_result is not None:
        timings["topology_setup_once_per_batch"] = topology_result
    return timings, verified, committed


def recurrence_oracle(inputs, paths, sampled):
    """Independent CPU FP64 recurrence, never the implementation under test."""
    tensors = {name: tensor.detach().cpu() for name, tensor in inputs.items() if isinstance(tensor, torch.Tensor)}
    q, k, v, g, beta = (tensors[name].double() for name in ("query", "key", "value", "g", "beta"))
    state = tensors["initial_state"].double().clone()
    group_size = v.shape[1] // k.shape[1]
    output = torch.empty_like(v)
    pseudo = torch.empty_like(v)
    start = 0
    for row, width in enumerate(tensors["row_widths"].tolist()):
        states = []
        for node in range(width):
            token = start + node
            if node == 0:
                previous = state[row]
            else:
                parent = (
                    node - 1 if tensors["tree_num_nodes"][row] == 0 else int(tensors["tree_parents"][row, node - 1])
                )
                previous = states[parent]
            keys = k[token].repeat_interleave(group_size, dim=0)
            queries = q[token].repeat_interleave(group_size, dim=0) * inputs["scale"]
            decayed = previous * g[token].exp()[:, None, None]
            delta = beta[token, :, None] * (v[token] - (decayed * keys[:, None]).sum(-1))
            current = decayed + delta[:, :, None] * keys[:, None]
            states.append(current)
            pseudo[token] = delta
            output[token] = (current * queries[:, None]).sum(-1)
        count = int(sampled[row])
        if width:
            if tensors["tree_num_nodes"][row] == 0:
                leaf = count - 1 if count > 0 else width - 1
                state[row] = states[leaf]
            elif count > 0:
                leaf = int(paths[row, count - 2]) if count > 1 else 0
                state[row] = states[leaf]
        start += width
    return output, pseudo, state


def check_accuracy(module, args, device):
    """Report output and recurrent drift separately, outside timed sections."""
    expected_state = None
    actual_state = None
    maxima = dict(output_abs=0.0, pseudo_abs=0.0, state_abs=0.0, rounded_output_abs=0.0)
    for step in range(args.accuracy_steps):
        inputs, paths, sampled = make_inputs(args, device)
        if actual_state is not None:
            inputs["initial_state"] = actual_state
        # Alternate root-only and leaf commits to exercise rejected branches.
        if step % 2 == 0:
            sampled.fill_(1)
        oracle_inputs = dict(inputs)
        if expected_state is not None:
            oracle_inputs["initial_state"] = expected_state
        expected_output, expected_pseudo, expected_state = recurrence_oracle(oracle_inputs, paths.cpu(), sampled.cpu())
        result = module.treewy_verify_torch(**inputs)
        commit_kwargs = {}
        if getattr(result, "ancestors", None) is not None:
            commit_kwargs["ancestors"] = result.ancestors
        actual_state = module.treewy_commit_torch(
            inputs["initial_state"],
            inputs["key"],
            result.pseudo_values,
            result.log_decay,
            inputs["tree_parents"],
            inputs["row_widths"],
            paths,
            sampled,
            inputs["tree_num_nodes"],
            **commit_kwargs,
        )
        for label, actual, expected in (
            ("output_abs", result.output, expected_output),
            ("pseudo_abs", result.pseudo_values, expected_pseudo),
            ("state_abs", actual_state, expected_state),
            ("rounded_output_abs", result.output, expected_output.to(result.output.dtype)),
        ):
            actual_cpu = actual.detach().cpu().double()
            if not bool(torch.isfinite(actual_cpu).all()):
                raise RuntimeError(f"Non-finite {label} at step {step}.")
            maxima[label] = max(maxima[label], float((actual_cpu - expected.double()).abs().max()))
    return {"steps": args.accuracy_steps, "max_errors_vs_fp64_recurrence": maxima}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "npu", "cuda"], default="npu")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--slots", type=int, default=33, help="Root plus tree budget.")
    parser.add_argument("--key-heads", type=int, default=8)
    parser.add_argument("--value-heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--topology", choices=["chain", "branch"], default="branch")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--count-ops", action="store_true")
    parser.add_argument("--accuracy-steps", type=int, default=0)
    parser.add_argument(
        "--hf32",
        choices=["default", "on", "off"],
        default="default",
        help="NPU-only diagnostic: compare HF32 matmul precision without changing the serving process.",
    )
    args = parser.parse_args()
    if min(args.rows, args.slots, args.key_heads, args.value_heads, args.key_dim, args.value_dim, args.iterations) < 1:
        parser.error("Dimensions and iterations must be positive.")
    if args.value_heads % args.key_heads:
        parser.error("value-heads must be divisible by key-heads.")
    if args.warmup < 0 or args.accuracy_steps < 0 or args.cpu_threads < 1:
        parser.error("warmup/accuracy-steps must be non-negative and cpu-threads must be positive.")
    if args.device == "npu" and torch_npu is None:
        parser.error("--device npu requires torch_npu.")
    if args.hf32 != "default" and args.device != "npu":
        parser.error("--hf32 on/off requires --device npu.")
    device = torch.device(args.device)
    if args.hf32 != "default":
        torch.npu.matmul.allow_hf32 = args.hf32 == "on"
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(17)
    source = args.source or Path(__file__).resolve().parents[2] / "vllm_ascend/worker/v2/spec_decode/tree/treewy.py"
    module = load_source(source, "treewy_benchmark_current")
    inputs, paths, sampled = make_inputs(args, device)
    with torch.inference_mode():
        result, verified, state = run(module, inputs, paths, sampled, args, device)
        report = {"config": vars(args), "torch": torch.__version__, "current": result}
        if device.type == "npu":
            report["npu_matmul_allow_hf32"] = torch.npu.matmul.allow_hf32
        if args.accuracy_steps:
            report["accuracy"] = check_accuracy(module, args, device)
        if args.baseline:
            baseline = load_source(args.baseline, "treewy_benchmark_baseline")
            old_result, old_verified, old_state = run(baseline, inputs, paths, sampled, args, device)
            tolerance = 5e-3 if args.dtype == "bfloat16" else 1e-5
            torch.testing.assert_close(verified.output, old_verified.output, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(
                verified.pseudo_values, old_verified.pseudo_values, rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(state, old_state, rtol=tolerance, atol=tolerance)
            report.update(
                baseline=old_result,
                max_output_abs_error=float((verified.output.float() - old_verified.output.float()).abs().max()),
                max_state_abs_error=float((state.float() - old_state.float()).abs().max()),
            )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
