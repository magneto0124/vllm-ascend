# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the bounded diagnostic recorder."""

import ast
import tempfile
import unittest
from inspect import signature
from pathlib import Path
from types import SimpleNamespace


class HostTensor:
    def __getitem__(self, key):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return HostTensor()


class TestTreeModelRecorder(unittest.TestCase):
    def test_post_forward_recording_does_not_query_forward_context_eligibility(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/tree_model_diag.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(module)
                      if isinstance(node, ast.FunctionDef) and node.name == "_record_first_attention")
        calls = [node.func.attr for node in ast.walk(method)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        # execute_model has already exited set_forward_context at this point.
        self.assertNotIn("_tree_decode_fia_eligible", calls)

    def test_first_attention_metadata_copies_are_reused_and_refreshed(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/tree_model_diag.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef))

        class Tensor:
            def __init__(self, values):
                self.values = list(values)
                self.shape = (len(values),)

            def copy_(self, other):
                self.values = list(other.values)

        namespace = {
            "Path": Path, "signature": signature,
            "torch": SimpleNamespace(Tensor=Tensor, empty_like=lambda value: Tensor(value.values)),
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        recorder = namespace["TreeModelRecorder"]("unused")
        recorder.first_attention = "layer"
        recorder.tap_width = 2
        recorder._tap = lambda *args: None
        metadata = SimpleNamespace(attn_mask=Tensor([1, 0]), slot_mapping=Tensor([4, 5]),
                                   block_tables=Tensor([1]))

        def original(query, attn_metadata, output):
            # Metadata must be copied before the attention body, without aliasing.
            attn_metadata.attn_mask.values[0] = 9
            return output

        wrapped = recorder._wrap_attention(original, "layer/ATTN", ("query",), "output")
        wrapped(Tensor([0, 0]), metadata, Tensor([0, 0]))
        buffer = recorder.attention_taps["attn_mask"]
        self.assertEqual(buffer.values, [1, 0])
        metadata.attn_mask.values = [0, 1]
        wrapped(Tensor([0, 0]), metadata, Tensor([0, 0]))
        self.assertIs(buffer, recorder.attention_taps["attn_mask"])
        self.assertEqual(buffer.values, [0, 1])
        metadata.attn_mask.values = [1, 1]
        wrapped(Tensor([0]), metadata, Tensor([0]))
        self.assertEqual(buffer.values, [0, 1])  # Ignore non-verify widths.

    def test_taps_surround_inplace_attention_and_preserve_return(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/tree_model_diag.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef))
        namespace = {"Path": Path, "signature": signature}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        recorder = namespace["TreeModelRecorder"]("unused")
        events = []
        recorder._tap = lambda name, tensor: events.append((name, list(tensor)))

        def original(query, output):
            output[:] = [query[0] * 2]
            query[0] = 9
            return output

        wrapped = recorder._wrap_attention(original, "layer", ("query",), "output")
        output = [0]
        self.assertIs(wrapped([3], output=output), output)
        self.assertEqual(events, [("layer/in/query", [3]), ("layer/out", [6])])

    def test_records_prefill_and_only_three_verifications(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/tree_model_diag.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef))
        saved = []
        namespace = {
            "Path": Path,
            "torch": SimpleNamespace(save=lambda data, path: saved.append((data, path))),
            "get_world_group": lambda: SimpleNamespace(rank_in_group=1),
            "logger": SimpleNamespace(warning=lambda *args: None),
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
        tensor = HostTensor()
        batch = SimpleNamespace(num_tokens=16, num_reqs=1, input_ids=tensor, positions=tensor,
                                query_start_loc=tensor, seq_lens=tensor)
        state = SimpleNamespace(input_batch=batch, hidden_states=tensor, aux_hidden_states=[])
        runner = SimpleNamespace(execute_model_state=state)
        scheduler = SimpleNamespace(scheduled_spec_decode_tokens={})
        with tempfile.TemporaryDirectory() as directory:
            recorder = namespace["TreeModelRecorder"](directory)
            recorder.record(runner, scheduler)
            scheduler.scheduled_spec_decode_tokens = {"request": [1]}
            for _ in range(6):
                recorder.record(runner, scheduler)
        self.assertEqual(len(saved), 4)
        self.assertFalse(saved[0][0]["verify"])
        self.assertEqual(saved[-1][1].name, "rank1_step003.pt")
        self.assertIsNot(saved[0][0]["hidden"], tensor)


if __name__ == "__main__":
    unittest.main()
