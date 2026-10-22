# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for the upstream load_model width reassignment.

Extract the real property so these tests can run without importing torch_npu.
Run directly with Python, or collect with pytest in the normal UT environment.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_runner_width_property():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    methods = [node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "decode_query_len"]
    if len(methods) != 2:
        raise AssertionError("Expected the runner's decode_query_len getter and setter")
    isolated = ast.Module(
        body=[ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[],
    )
    namespace = {"dflash_tree_spec_enabled": lambda config: config.tree_enabled}
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace["Runner"]


class TestTreeDecodeQueryLen(unittest.TestCase):
    def make_runner(self, tree_enabled, budget=15):
        runner = load_runner_width_property()()
        runner.vllm_config = SimpleNamespace(tree_enabled=tree_enabled)
        runner.ascend_config = SimpleNamespace(tree_spec_config=SimpleNamespace(budget=budget))
        runner.num_speculative_steps = 7
        return runner

    def test_upstream_reassignment_preserves_tree_allocations(self):
        runner = self.make_runner(True)
        runner.decode_query_len = 16
        # load_model reassigns before creating sampler/grammar buffers and graphs.
        runner.decode_query_len = runner.num_speculative_steps + 1
        consumers = {
            "sampler_capacity": runner.decode_query_len,
            "grammar_stride": runner.decode_query_len,
            "graph_decode_query_len": runner.decode_query_len,
        }
        self.assertEqual(set(consumers.values()), {16})
        self.assertEqual(runner.num_speculative_steps, 7)

    def test_chain_and_autoregressive_keep_upstream_width(self):
        runner = self.make_runner(False)
        for width in (1, 8, 9):
            with self.subTest(width=width):
                runner.decode_query_len = width
                self.assertEqual(runner.decode_query_len, width)

    def test_tree_budget_is_not_hardcoded(self):
        for budget in (4, 7, 31):
            with self.subTest(budget=budget):
                runner = self.make_runner(True, budget)
                runner.decode_query_len = 8
                self.assertEqual(runner.decode_query_len, budget + 1)


if __name__ == "__main__":
    unittest.main()
