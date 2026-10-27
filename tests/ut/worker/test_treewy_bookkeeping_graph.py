# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only checks of fused TreeWY bookkeeping routing and eager fallbacks."""

import ast
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


STATE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py"


class TensorStub:
    def __init__(self, shape, device="npu"):
        self.shape = shape
        self.ndim = len(shape)
        self.device = device

    def to(self, dtype):
        return self


class BaseState:
    def postprocess_state(self, *args):
        self.upstream(*args)


def load_state(torch):
    module = ast.parse(STATE.read_text(encoding="utf-8"))
    original = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendMambaHybridModelState"
    )
    names = {"_postprocess_state_impl", "_can_fuse_treewy_bookkeeping"}
    cls = ast.ClassDef(
        name="State",
        bases=[ast.Name(id="BaseState", ctx=ast.Load())],
        keywords=[],
        body=[node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names],
        decorator_list=[],
    )
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    namespace = dict(torch=torch, BaseState=BaseState, tree_time=lambda name: nullcontext())
    exec(compile(ast.fix_missing_locations(isolated), str(STATE), "exec"), namespace)
    return namespace["State"]()


class TestTreeWYBookkeepingRouting(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            Tensor=TensorStub, bool="bool", int32="int32",
            where=Mock(), zeros_like=Mock(), ones_like=Mock(return_value=TensorStub((2,))),
        )
        self.state = load_state(self.torch)
        state = self.state
        state._treewy_state_layout = True
        state.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1))
        state.recoverssm = None
        state._align_mode = False
        state._tree_path = TensorStub((2, 4))
        state._tree_num_sampled = TensorStub((2,))
        state._tree_is_tree_row = TensorStub((2,))
        state._tree_resume_column = TensorStub((2,))
        state._tree_prev_path = None
        state._treewy_contexts_to_commit = [{"layer": object()}]
        state._run_treewy_commit_graph = Mock(return_value=True)
        state._run_snapshot_commit_graph = Mock(return_value=True)
        state._tree_buffers = Mock(return_value=(Mock(), Mock()))
        state.upstream = Mock()
        self.eager = Mock()
        self.module = SimpleNamespace(treewy_commit_contexts_torch=self.eager)
        self.idx = TensorStub((2,))
        self.sampled = TensorStub((2,))

    def run_commit(self):
        module = "vllm_ascend.worker.v2.spec_decode.tree.treewy"
        with patch.dict(sys.modules, {module: self.module}):
            self.state._postprocess_state_impl(self.idx, self.sampled)

    def assert_cleared(self):
        for name in ("_tree_path", "_tree_num_sampled", "_tree_is_tree_row", "_tree_resume_column"):
            self.assertIsNone(getattr(self.state, name))
        self.assertEqual(self.state._treewy_contexts_to_commit, [])

    def test_fused_success_skips_duplicate_eager_and_upstream_writes(self):
        mask = self.state._tree_is_tree_row
        self.run_commit()
        self.assertEqual(self.state._run_treewy_commit_graph.call_args.kwargs["bookkeeping"], (self.idx, mask))
        self.eager.assert_not_called()
        self.state._run_snapshot_commit_graph.assert_not_called()
        self.state._tree_buffers.assert_not_called()
        self.state.upstream.assert_not_called()
        self.assert_cleared()

    def test_capture_failure_preserves_reconstruction_and_history_fallback(self):
        contexts = self.state._treewy_contexts_to_commit[0]
        path, sampled = self.state._tree_path, self.state._tree_num_sampled
        self.state._run_treewy_commit_graph.return_value = False
        self.run_commit()
        self.eager.assert_called_once_with(contexts, path, sampled)
        self.state._run_snapshot_commit_graph.assert_called_once()
        self.state.upstream.assert_called_once_with(self.idx, self.torch.ones_like.return_value, None)
        self.assert_cleared()

    def test_only_first_successful_context_graph_gets_bookkeeping(self):
        self.state._treewy_contexts_to_commit *= 3
        self.state._run_treewy_commit_graph.side_effect = [False, True, True]
        mask = self.state._tree_is_tree_row
        self.run_commit()
        calls = self.state._run_treewy_commit_graph.call_args_list
        self.assertEqual([c.kwargs["bookkeeping"] for c in calls], [(self.idx, mask), (self.idx, mask), None])
        self.eager.assert_called_once()
        self.state.upstream.assert_not_called()

    def test_recovery_alignment_and_pipeline_keep_upstream_processing(self):
        for setting in ("pipeline", "recovery", "alignment"):
            with self.subTest(setting=setting):
                self.setUp()
                if setting == "pipeline":
                    self.state.vllm_config.parallel_config.pipeline_parallel_size = 2
                elif setting == "recovery":
                    self.state.recoverssm = object()
                else:
                    self.state._align_mode = True
                self.run_commit()
                self.assertIsNone(self.state._run_treewy_commit_graph.call_args.kwargs["bookkeeping"])
                self.state.upstream.assert_called_once()
                self.assert_cleared()

    def test_missing_captured_contexts_keep_history_and_upstream_processing(self):
        self.state._treewy_contexts_to_commit = []
        self.run_commit()
        self.state._run_treewy_commit_graph.assert_not_called()
        self.state._run_snapshot_commit_graph.assert_called_once()
        self.state.upstream.assert_called_once()

    def test_fusion_rejects_non_tree_and_mismatched_batch_shapes(self):
        self.assertTrue(self.state._can_fuse_treewy_bookkeeping(self.idx))
        for idx in (TensorStub((0,)), TensorStub((1,)), TensorStub((2, 1)), TensorStub((2,), "cpu")):
            self.assertFalse(self.state._can_fuse_treewy_bookkeeping(idx))
        self.state._tree_is_tree_row = TensorStub((1,))
        self.assertFalse(self.state._can_fuse_treewy_bookkeeping(self.idx))
        self.state._tree_is_tree_row = None
        self.assertTrue(self.state._can_fuse_treewy_bookkeeping(self.idx))
        self.state._treewy_state_layout = False
        self.assertFalse(self.state._can_fuse_treewy_bookkeeping(self.idx))


if __name__ == "__main__":
    unittest.main()
