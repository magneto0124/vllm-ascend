# SPDX-License-Identifier: Apache-2.0
"""Pairing rules between ``tree_spec_config.method`` and the draft backend."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from vllm_ascend.ascend_config import TreeSpecConfig
from vllm_ascend.worker.v2.spec_decode.tree.builder import (
    METHOD_REQUIRED_BACKEND,
    create_tree_builder,
    validate_tree_method_backend,
)
from vllm_ascend.worker.v2.spec_decode.tree.beam import BeamTreeBuilder

TREE_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/tree"


class TestTreeMethodBackend(unittest.TestCase):
    def test_beam_lists_dspark_and_dflash(self):
        self.assertEqual(METHOD_REQUIRED_BACKEND["beam"], ("dspark", "dflash"))

    def test_beam_accepts_both_backends(self):
        for backend in ("dspark", "dflash"):
            validate_tree_method_backend("beam", backend)

    def test_beam_rejects_other_backends(self):
        with self.assertRaisesRegex(ValueError, r"'dspark' or 'dflash', got 'eagle'"):
            validate_tree_method_backend("beam", "eagle")

    def test_priority_and_prefix_still_require_dflash(self):
        for method in ("priority", "prefix"):
            validate_tree_method_backend(method, "dflash")
            with self.assertRaisesRegex(ValueError, r"requires speculative_config method 'dflash'"):
                validate_tree_method_backend(method, "dspark")

    def test_unknown_method_rejected(self):
        with self.assertRaisesRegex(ValueError, r"method must be one of"):
            validate_tree_method_backend("bogus", "dflash")

    def test_create_tree_builder_beam_on_dflash_has_no_markov_head(self):
        builder = create_tree_builder(
            method="beam",
            budget=15,
            topk=2,
            draft_backend="dflash",
        )
        self.assertIsInstance(builder, BeamTreeBuilder)
        self.assertIsNone(builder.draft_model)

    def test_ascend_config_gate_accepts_beam_for_a_dflash_draft(self):
        tree_cfg = TreeSpecConfig(enabled=True, method="beam", budget=15, topk=2)
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(use_dspark=lambda: False)
        )
        # A DFlash / DFlash2 draft is now a valid beam host.
        tree_cfg._validate_method_backend(vllm_config)
        vllm_config.speculative_config = SimpleNamespace(use_dspark=lambda: True)
        tree_cfg._validate_method_backend(vllm_config)


def load_bind_correction_heads():
    """Extract ``_bind_correction_heads`` so the test needs no NPU/PyTorch."""
    path = TREE_DIR / "speculator.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendTreeSpeculator"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_bind_correction_heads"
    )
    isolated = ast.Module(
        body=[
            ast.ClassDef(
                name="Host", bases=[], keywords=[], body=[method], decorator_list=[]
            )
        ],
        type_ignores=[],
    )
    namespace = {
        "logger": Mock(),
        "create_tree_builder": Mock(return_value="builder"),
        # Only used by the ``target_model: nn.Module`` annotation on the method.
        "nn": SimpleNamespace(Module=object),
    }
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace["Host"], namespace["create_tree_builder"]


class TestBeamMarkovBinding(unittest.TestCase):
    def setUp(self):
        self.Host, self.create_tree_builder = load_bind_correction_heads()
        self.host = self.Host()
        self.host.method = "beam"
        self.host.budget = 15
        self.host.topk = 2
        self.host.params = {}
        self.host.draft_backend = "dflash"
        self.host._domino_scorer = None
        self.host._domino_prefix_len = 0
        self.host._domino_shift_label = False

    def test_draft_without_markov_head_gets_no_correction_model(self):
        # DFlash / DFlash2: no markov_embed / markov_bias on the draft model.
        self.host.model = SimpleNamespace()
        self.host._bind_correction_heads(None)
        kwargs = self.create_tree_builder.call_args.kwargs
        self.assertIsNone(kwargs["draft_model"])
        self.assertEqual(self.host.tree_builder, "builder")

    def test_dspark_markov_head_is_still_bound(self):
        model = SimpleNamespace(markov_embed=Mock(), markov_bias=Mock())
        self.host.model = model
        self.host._bind_correction_heads(None)
        kwargs = self.create_tree_builder.call_args.kwargs
        self.assertIs(kwargs["draft_model"], model)


if __name__ == "__main__":
    unittest.main()
