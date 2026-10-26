# SPDX-License-Identifier: Apache-2.0
"""Exercise the real validator with cached warning semantics, without NPU imports."""

import ast
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


class TestTreeConfigWarning(unittest.TestCase):
    def test_unknown_params_are_preserved_and_warning_is_hashable(self):
        path = Path(__file__).resolve().parents[2] / "vllm_ascend/ascend_config.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TreeSpecConfig")
        validator = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_validate")
        validator.decorator_list = []
        messages = []

        @lru_cache(maxsize=None)
        def warning_once(message, *args):
            messages.append(message % args)

        namespace = {"logger": SimpleNamespace(warning_once=warning_once)}
        exec(compile(ast.Module(body=[validator], type_ignores=[]), str(path), "exec"), namespace)
        config = SimpleNamespace(
            method="beam", SUPPORTED_METHODS=("beam",), rejection_sampler="greedy",
            SUPPORTED_REJECTION_SAMPLERS=("greedy",), enabled=True, budget=15, topk=4,
            params={"unknown_param": "/tmp/params"}, PREFIX_PARAM_KEYS=frozenset({"candidate_size"}),
        )
        for _ in range(2):
            self.assertIs(namespace["_validate"](config), config)
        self.assertEqual(len(messages), 1)
        self.assertIn("unknown_param", messages[0])
        self.assertEqual(config.params, {"unknown_param": "/tmp/params"})


if __name__ == "__main__":
    unittest.main()
