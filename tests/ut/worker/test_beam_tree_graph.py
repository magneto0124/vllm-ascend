# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise graph dispatch without importing the NPU worker or PyTorch.

Run directly with Python on hosts without the vLLM runtime. The extracted
methods are production code, with only graph/device operations mocked.
"""

import ast
import logging
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch


TREE_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/tree"


def load_speculator(torch):
    path = TREE_DIR / "speculator.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in module.body
        if isinstance(n, ast.ClassDef) and n.name == "AscendTreeSpeculator"
    )
    methods = [
        n for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_capture_beam_graphs", "_finalize_tree", "_run_beam_finalize"}
    ]
    isolated = ast.Module(
        body=[ast.ClassDef(name="Speculator", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "logger": logging.getLogger(__name__),
        "nullcontext": nullcontext,
        "communicator_switch": nullcontext,
        "graph_capture": lambda **kwargs: nullcontext(),
        "torch_cuda_wrapper": nullcontext,
        "ensure_finalize_scratch": Mock(),
    }
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace["Speculator"]()


class TestBeamTreeGraph(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.inference_mode.side_effect = nullcontext
        self.torch.npu.graph.side_effect = lambda *args, **kwargs: nullcontext()
        self.torch.npu.NPUGraph.side_effect = lambda: Mock()
        self.timer = Mock(side_effect=lambda *args: nullcontext())
        self.validate = Mock()
        modules = {
            "vllm_ascend.worker.v2.spec_decode.tree.timer": SimpleNamespace(
                tree_time=self.timer
            ),
            "vllm.compilation.monitor": SimpleNamespace(
                validate_cudagraph_capturing_enabled=self.validate
            ),
            "vllm.platforms": SimpleNamespace(
                current_platform=SimpleNamespace(
                    get_global_graph_pool=Mock(return_value="pool")
                )
            ),
            "vllm_ascend.worker.v2.aclgraph_utils": SimpleNamespace(
                model_capture_wrapper=lambda *args: nullcontext()
            ),
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.spec = load_speculator(self.torch)
        self.spec.method = "beam"
        self.spec.device = "npu"
        self.spec.max_num_reqs = 5
        self.spec.num_query_per_req = 7
        self.spec.num_speculative_steps = 7
        self.spec.vocab_size = 32
        self.spec.dtype = "bf16"
        self.spec.tree_proposal_logits = None
        self.spec.tree_builder = Mock()
        self.spec.query_cudagraph_manager = Mock()
        self.spec._draft_capture_num_reqs = Mock(return_value=[1, 4])
        self.spec._draft_hidden_buf = MagicMock()
        self.spec.sample_indices = MagicMock()
        self.spec.input_buffers = SimpleNamespace(input_ids=MagicMock())
        self.spec._beam_graphs = {}
        self.spec._tree_finalized = False
        self.spec.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=1)
        )
        self.spec._load_layout_from_buffers = Mock(return_value=object())

    def test_capture_uses_the_available_request_gears(self):
        self.spec._run_beam_finalize = Mock()
        self.spec._capture_beam_graphs()
        self.assertEqual(list(self.spec._beam_graphs), [1, 4])
        self.assertEqual(
            self.spec._run_beam_finalize.call_args_list,
            [call(n, graph_safe=True) for n in (1, 1, 4, 4)],
        )
        self.assertFalse(self.spec._tree_finalized)

    def test_disabled_or_unsupported_paths_do_not_capture(self):
        cases = [
            ("priority", True, [1]),
            ("prefix", True, [1]),
            ("beam", False, [1]),
            ("beam", True, []),
        ]
        for method, needs_capture, sizes in cases:
            with self.subTest(method=method, needs_capture=needs_capture, sizes=sizes):
                self.spec.method = method
                self.spec.query_cudagraph_manager.needs_capture.return_value = needs_capture
                self.spec._draft_capture_num_reqs.return_value = sizes
                self.spec._capture_beam_graphs()
        self.torch.npu.NPUGraph.assert_not_called()

    def test_capture_failure_keeps_only_successful_gears(self):
        def finalize(n, **kwargs):
            if n == 3:
                raise RuntimeError("unsupported capture")

        self.spec._draft_capture_num_reqs.return_value = [1, 3, 4]
        self.spec._run_beam_finalize = Mock(side_effect=finalize)
        with self.assertLogs(__name__, level="WARNING"):
            self.spec._capture_beam_graphs()
        self.assertEqual(list(self.spec._beam_graphs), [1, 4])
        self.spec._run_beam_finalize = Mock()
        self.spec._finalize_tree(3)
        self.spec._run_beam_finalize.assert_called_once_with(3, graph_safe=False)

    def test_tensor_parallel_capture_failure_is_not_rank_local_fallback(self):
        self.spec.vllm_config.parallel_config.tensor_parallel_size = 2
        self.spec._run_beam_finalize = Mock(side_effect=RuntimeError("failed"))
        with self.assertRaisesRegex(RuntimeError, "tensor parallelism"):
            self.spec._capture_beam_graphs()
        self.assertFalse(self.spec._beam_graphs)

    def test_replay_rebinds_layout_and_finalizes_once(self):
        graph = Mock()
        self.spec._beam_graphs[2] = graph
        self.spec._run_beam_finalize = Mock()
        self.spec._finalize_tree(2)
        self.spec._finalize_tree(2)
        graph.replay.assert_called_once()
        self.spec._run_beam_finalize.assert_not_called()
        self.spec._load_layout_from_buffers.assert_called_once_with(2)
        self.assertIs(self.spec.tree, self.spec._load_layout_from_buffers.return_value)

    def test_finalize_captures_logits_and_passes_graph_safe_without_timer(self):
        self.spec.num_speculative_steps = 7
        self.spec.budget = 15
        self.spec.model = Mock()
        self.spec.model.compute_draft_logits.return_value = MagicMock()
        self.spec.tree_proposal_logits = None
        self.spec._run_beam_finalize(2, graph_safe=True)
        self.spec.model.compute_draft_logits.assert_called_once()
        kwargs = self.spec.tree_builder.build.call_args.kwargs
        self.assertTrue(kwargs["graph_safe"])
        self.assertIsNone(kwargs["proposal_logits"])
        self.timer.assert_not_called()

    def test_force_torch_bypasses_enabled_triton_dispatch(self):
        path = TREE_DIR / "layout.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "finalize_tree_layout"
        )
        # Annotations are irrelevant to the dispatcher and require torch.
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        namespace = {
            "_finalize_tree_layout_triton": Mock(),
            "finalize_tree_layout_torch": Mock(),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        dispatcher = SimpleNamespace(use_tree_triton=lambda: True)
        with patch.dict(sys.modules, {"vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch": dispatcher}):
            namespace["finalize_tree_layout"](
                None, None, None, None, 15, force_torch=True
            )
            namespace["finalize_tree_layout"](None, None, None, None, 15)
        namespace["finalize_tree_layout_torch"].assert_called_once()
        namespace["_finalize_tree_layout_triton"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
