# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path


def load_diag():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/tree_graph_diag.py"
    spec = importlib.util.spec_from_file_location("tree_graph_diag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    shape = (1, 16)

    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


@dataclass
class Metadata:
    num_actual_tokens: int = 16


class TestTreeGraphDiag(unittest.TestCase):
    def test_dynamic_metadata_fields_are_included(self):
        diag = load_diag()
        meta = Metadata()
        meta.spec_decode_metadata = {"init_state_indices": FakeTensor(1)}
        result = diag.tensor_fields(meta)
        self.assertIn("spec_decode_metadata.init_state_indices", result)

    def test_address_shape_and_missing_changes(self):
        diag = load_diag()
        before = {"same": FakeTensor(1), "moved": FakeTensor(2), "missing": FakeTensor(3)}
        after = {"same": FakeTensor(1), "moved": FakeTensor(4)}
        self.assertEqual(diag.pointer_changes(before, after), ["moved:address", "missing:missing"])
        after["same"].shape = (2, 16)
        self.assertIn("same:shape (1, 16)->(2, 16)", diag.pointer_changes(before, after))


if __name__ == "__main__":
    unittest.main()
