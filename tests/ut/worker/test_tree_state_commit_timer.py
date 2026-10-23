# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only timing-boundary tests; runnable without PyTorch or an NPU."""

import ast
import io
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
TREE = ROOT / "vllm_ascend/worker/v2/spec_decode/tree"
STATE = ROOT / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py"
RUNNER = ROOT / "vllm_ascend/worker/v2/model_runner.py"


def load_commit_wrapper(timer):
    module = ast.parse(STATE.read_text(encoding="utf-8"))
    cls = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendMambaHybridModelState"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "postprocess_state"
    )
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {name: timer[name] for name in ("tree_time", "tree_timer_enabled")}
    exec(compile(ast.fix_missing_locations(isolated), str(STATE), "exec"), namespace)
    return namespace["postprocess_state"]


class TestTreeStateCommitTimer(unittest.TestCase):
    def setUp(self):
        self.sync = Mock()
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=self.sync))
        with patch.dict(sys.modules, {"torch": torch}):
            self.timer = runpy.run_path(str(TREE / "timer.py"))
        self.commit = load_commit_wrapper(self.timer)
        self.state = SimpleNamespace(
            _tree_path=object(), _tree_prev_path=None, _postprocess_state_impl=Mock()
        )
        self.args = (object(), object(), object())

    def enable(self, warmup=0):
        # Do not install exit hooks for these isolated test modules.
        with patch("atexit.register"):
            self.timer["configure_tree_timer"](enabled=True, warmup_steps=warmup)
        self.timer["tree_timer_begin_step"]()

    def call_commit(self):
        self.commit(self.state, *self.args)
        self.state._postprocess_state_impl.assert_called_once_with(*self.args)

    def test_disabled_preserves_call_without_synchronization(self):
        self.call_commit()
        self.sync.assert_not_called()
        self.assertEqual(dict(self.timer["_SAMPLES"]), {})

    def test_enabled_times_complete_commit_once(self):
        self.enable()
        self.call_commit()
        self.assertEqual(self.sync.call_count, 2)
        self.assertEqual(len(self.timer["_SAMPLES"]["gdn_state_commit"]), 1)

    def test_stale_path_cleanup_is_also_timed(self):
        self.enable()
        self.state._tree_path = None
        self.state._tree_prev_path = object()
        self.call_commit()
        self.assertEqual(len(self.timer["_SAMPLES"]["gdn_state_commit"]), 1)

    def test_non_tree_commit_is_not_timed(self):
        self.enable()
        self.state._tree_path = None
        self.call_commit()
        self.sync.assert_not_called()

    def test_warmup_does_not_add_samples(self):
        self.enable(warmup=1)
        self.call_commit()
        self.assertEqual(dict(self.timer["_SAMPLES"]), {})

    def test_commit_failure_is_not_swallowed_or_retried(self):
        self.enable()
        self.state._postprocess_state_impl.side_effect = RuntimeError("commit failed")
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            self.commit(self.state, *self.args)
        self.state._postprocess_state_impl.assert_called_once_with(*self.args)

    def test_report_contains_three_comparable_segments(self):
        self.enable()
        names = ("gdn_state_resume_prepare", "gdn_state_commit", "compact_query_path")
        for name in names:
            with self.timer["tree_time"](name):
                pass
        output = io.StringIO()
        with redirect_stdout(output):
            self.timer["print_tree_timer_report"]()
        for name in names:
            self.assertIn(f"{name}: n=1 mean_ms=", output.getvalue())

    def test_resume_timer_covers_calculation_but_not_diagnostics(self):
        module = ast.parse(RUNNER.read_text(encoding="utf-8"))
        method = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "_stage_tree_state_resume_column"
        )
        span = next(
            node for node in ast.walk(method)
            if isinstance(node, ast.With)
            and "gdn_state_resume_prepare" in ast.unparse(node.items[0].context_expr)
        )
        calls = [ast.unparse(node.func) for node in ast.walk(span) if isinstance(node, ast.Call)]
        for name in ("compute_tree_resume_column", "tree_row_mask", "model_state.stage_tree_resume_column"):
            self.assertIn(name, calls)
        self.assertNotIn("tree_diag_once", calls)


if __name__ == "__main__":
    unittest.main()
