# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for the hybrid attention metadata forwarding contract."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestHybridTreeVisibility(unittest.TestCase):
    def test_builder_receives_runtime_tree_and_capture_none(self):
        path = (Path(__file__).resolve().parents[3]
                / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py")
        module = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(module)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "build_attn_metadata"]
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg: kw.value for kw in calls[0].keywords}
        self.assertIn("tree_visibility", keywords,
                      "Hybrid attention must forward the real tree, not retain the capture dummy mask")
        expression = compile(ast.Expression(keywords["tree_visibility"]), str(path), "eval")
        # A branching tree must reach the ordinary attention builder unchanged.
        # Capture and non-tree batches intentionally leave dummy/default handling
        # to that builder by forwarding None.
        branching_tree = ((True, False), (False, True))
        for visibility in (branching_tree, None):
            with self.subTest(visibility=visibility):
                actual = eval(expression, {"input_batch": SimpleNamespace(tree_visibility=visibility)})
                self.assertIs(actual, visibility)


if __name__ == "__main__":
    unittest.main()
