# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only graph dispatch tests; also runnable directly without PyTorch."""

import ast
import logging
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch


BUILDER_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/gdn_attn_builder.py"


def load_graph_methods(torch):
    module = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == "AscendGDNAttentionMetadataBuilder")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        "_get_or_capture_snapshot_tree_state_table_graph", "_run_snapshot_tree_state_table_graph"
    }]
    isolated = ast.Module(body=[ast.ClassDef(
        name="Builder", bases=[], keywords=[], body=methods, decorator_list=[]
    )], type_ignores=[])
    namespace = {"torch": torch, "Any": Any, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(isolated), str(BUILDER_PATH), "exec"), namespace)
    return namespace["Builder"]()


class TestTreeStateTableGraph(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.inference_mode.side_effect = nullcontext
        self.torch.npu.graph.side_effect = lambda *args, **kwargs: nullcontext()
        self.torch.npu.is_current_stream_capturing.return_value = False
        self.torch.npu.NPUGraph.side_effect = lambda: Mock()
        self.validate = Mock()
        self.modules = patch.dict(sys.modules, {
            "vllm.compilation.monitor": SimpleNamespace(validate_cudagraph_capturing_enabled=self.validate),
            "vllm.platforms": SimpleNamespace(
                current_platform=SimpleNamespace(get_global_graph_pool=lambda: "pool")
            ),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.builder = load_graph_methods(self.torch)
        self.builder._tree_state_table_graphs = {}
        self.builder._tree_state_table_graph_failures = set()
        self.builder.spec_tree_slots = 16
        self.builder.spec_conv_width = 4
        for name in ("spec_tree_parents", "spec_tree_num_nodes", "spec_init_state_indices",
                     "spec_conv_window_rows", "spec_conv_prefix_src"):
            setattr(self.builder, name, MagicMock())
        self.builder.spec_tree_parents.device.type = "npu"
        self.builder._compute_snapshot_tree_state_tables = Mock()
        self.builder._copy_tree_state_table_inputs = Mock()
        self.indices = Mock()
        self.indices.numel.return_value = 2

    def run_tables(self, capture=False):
        return self.builder._run_snapshot_tree_state_table_graph(
            Mock(), Mock(), self.indices, Mock(), Mock(), capture=capture
        )

    def test_capture_once_and_replay_on_first_and_subsequent_calls(self):
        first = self.run_tables(capture=True)
        graph = self.builder._tree_state_table_graphs[2]
        self.assertIsNotNone(first)
        self.assertEqual(self.builder._compute_snapshot_tree_state_tables.call_count, 2)
        self.assertEqual(graph.replay.call_count, 1)
        self.run_tables(capture=True)  # Metadata prepared again before model capture.
        self.run_tables(capture=False)
        self.assertEqual(graph.replay.call_count, 3)
        self.assertEqual(self.builder._copy_tree_state_table_inputs.call_count, 3)
        self.torch.npu.NPUGraph.assert_called_once()
        self.torch.npu.graph.assert_called_once_with(graph, pool="pool")

    def test_uncaptured_runtime_size_does_not_copy_or_capture(self):
        self.assertIsNone(self.run_tables())
        self.builder._copy_tree_state_table_inputs.assert_not_called()
        self.torch.npu.NPUGraph.assert_not_called()

    def test_failed_capture_falls_back_and_is_not_retried(self):
        self.torch.npu.graph.side_effect = RuntimeError("unsupported capture")
        self.assertIsNone(self.run_tables(capture=True))
        self.assertIn(2, self.builder._tree_state_table_graph_failures)
        self.builder._copy_tree_state_table_inputs.reset_mock()
        self.assertIsNone(self.run_tables(capture=True))
        self.assertIsNone(self.run_tables())
        self.builder._copy_tree_state_table_inputs.assert_not_called()
        self.torch.npu.NPUGraph.assert_called_once()
        self.assertNotIn(2, self.builder._tree_state_table_graphs)

    def test_no_nested_or_cpu_capture(self):
        self.torch.npu.is_current_stream_capturing.return_value = True
        self.assertIsNone(self.run_tables(capture=True))
        self.torch.npu.is_current_stream_capturing.return_value = False
        self.builder.spec_tree_parents.device.type = "cpu"
        self.assertIsNone(self.run_tables(capture=True))
        self.torch.npu.NPUGraph.assert_not_called()

    def test_replay_errors_are_not_hidden_as_eager_fallback(self):
        graph = Mock()
        graph.replay.side_effect = RuntimeError("replay failed")
        self.builder._tree_state_table_graphs[2] = graph
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            self.run_tables()

    def test_build_gate_excludes_short_mixed_treewy_and_padded_batches(self):
        module = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
        gate = next(node for node in ast.walk(module) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "full_width_snapshot_batch"
                            for target in node.targets))
        expr = compile(ast.Expression(gate.value), str(BUILDER_PATH), "eval")
        self.builder.use_full_cuda_graph = True
        self.builder.decode_cudagraph_max_bs = 128
        widths = Mock()
        widths.numel.return_value = 2
        self.torch.all.return_value = True
        values = dict(self=self.builder, torch=self.torch, num_prefills=0, num_decodes=0,
                      num_spec_decodes=2, num_spec_decode_tokens=32,
                      m=SimpleNamespace(num_reqs=2), spec_row_widths=widths,
                      selected_gdn_state_update_name=lambda: "snapshot")
        self.assertTrue(eval(expr, values))
        for changes in (
            {"num_prefills": 1}, {"num_decodes": 1}, {"m": SimpleNamespace(num_reqs=3)},
            {"selected_gdn_state_update_name": lambda: "treewy"}, {"num_spec_decode_tokens": 144},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(eval(expr, {**values, **changes}))
        self.torch.all.return_value = False  # At least one short row.
        self.assertFalse(eval(expr, values))
        self.torch.all.return_value = True
        self.builder.use_full_cuda_graph = False
        self.assertFalse(eval(expr, values))


if __name__ == "__main__":
    unittest.main()
