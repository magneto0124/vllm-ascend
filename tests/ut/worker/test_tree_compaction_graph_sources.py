# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level regression checks for the two optional tree ACLGraphs."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


class TestTreeCompactionGraphSources(unittest.TestCase):
    def test_query_graph_keeps_eager_fallback_and_replay(self):
        funcs = functions(ROOT / "vllm_ascend/worker/v2/spec_decode/tree/speculator.py")
        method = funcs["_run_query_path_graph"]
        source = ast.unparse(method)
        self.assertIn("NPUGraph", source)
        self.assertIn("graph.replay()", source)
        self.assertIn("_query_path_graph_failures", source)
        propose = funcs["propose"]
        self.assertIn("compact_tree_query_along_path", ast.unparse(propose))
        self.assertIn("_run_query_path_graph", ast.unparse(propose))

    def test_commit_graph_only_shortcuts_snapshot_and_keeps_upstream_postprocess(self):
        path = ROOT / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py"
        funcs = functions(path)
        graph = ast.unparse(funcs["_run_snapshot_commit_graph"])
        impl = ast.unparse(funcs["_postprocess_state_impl"])
        self.assertIn("selected_gdn_state_update_name()", graph)
        self.assertIn("snapshot", graph)
        self.assertIn("NPUGraph", graph)
        self.assertIn("index_copy_", graph)
        self.assertIn("pipeline_parallel_size", graph)
        self.assertNotIn("torch.any", graph)
        self.assertIn("super().postprocess_state", impl)
        self.assertIn("_run_snapshot_commit_graph", impl)

    def test_graphs_are_not_attempted_inside_an_active_capture(self):
        for rel, name in (
            ("vllm_ascend/worker/v2/spec_decode/tree/speculator.py", "_run_query_path_graph"),
            ("vllm_ascend/worker/v2/model_states/mamba_hybrid.py", "_run_snapshot_commit_graph"),
        ):
            source = ast.unparse(functions(ROOT / rel)[name])
            self.assertIn("is_current_stream_capturing", source)


if __name__ == "__main__":
    unittest.main()
