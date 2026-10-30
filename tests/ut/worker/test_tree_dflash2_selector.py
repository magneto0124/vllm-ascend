# SPDX-License-Identifier: Apache-2.0
"""DFlash2 candidate-selector rows for the draft-tree host."""

import unittest
from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.spec_decode.tree.beam import _dflash2_parent_logits
from vllm_ascend.worker.v2.spec_decode.tree.dflash2_selector import (
    DFlash2TreeSelector,
    is_dflash2_architecture,
)

VOCAB_SIZE = 32
SELECTOR_TOP_K = 4
DEPTH = 3
NUM_REQS = 2
# ``compute_candidates`` returns ids offset by this much to prove the scatter
# uses the candidate ids rather than the shortlist slot indices.
ID_OFFSET = 5


class _FakeCandidateSelector(torch.nn.Module):
    """Scores candidates by slot, with predecessor 1 winning after depth 0."""

    def forward(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        num_reqs, depth, top_k = candidate_ids.shape
        child = torch.arange(top_k, dtype=torch.float32)
        scores = child.view(1, 1, 1, -1).expand(num_reqs, depth, top_k, top_k).clone()
        if depth > 1:
            # Only predecessor slot 1 scores high, so rows must pick that one.
            scores[:, 1:, 1, :] += 5.0
        # The anchor is the depth-0 predecessor, so an anchor shift shows up in
        # that request's rows only.
        return scores + anchor_token_ids.view(-1, 1, 1, 1).to(scores.dtype)


class _FakeDFlash2Model:
    def __init__(self):
        self.model = SimpleNamespace(candidate_selector=_FakeCandidateSelector())

    def compute_candidates(self, hidden_states):
        rows = hidden_states.shape[0]
        ids = torch.arange(SELECTOR_TOP_K).expand(rows, -1) + ID_OFFSET
        return ids.to(torch.int64), torch.zeros(rows, SELECTOR_TOP_K)


def _draft_model_config(top_k=SELECTOR_TOP_K):
    return SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        hf_config=SimpleNamespace(
            dflash_config={
                "selector_top_k": top_k,
                "selector_rank": 8,
                "conv_kernel_size": 3,
                "conv_group_size": 2,
            }
        ),
    )


class TestDFlash2TreeSelector(unittest.TestCase):
    def setUp(self):
        self.selector = DFlash2TreeSelector(
            _draft_model_config(),
            num_speculative_steps=DEPTH,
            vocab_size=VOCAB_SIZE,
        )
        self.model = _FakeDFlash2Model()

    def _scores(self, anchors=None):
        hidden = torch.zeros(NUM_REQS * DEPTH, 8)
        return self.selector.score(
            self.model,
            hidden,
            torch.zeros(NUM_REQS, dtype=torch.int64) if anchors is None else anchors,
            NUM_REQS,
        )

    def test_preserves_candidate_ids_and_predecessor_scores(self):
        candidate_ids, scores = self._scores()
        self.assertEqual(tuple(candidate_ids.shape), (NUM_REQS, DEPTH, SELECTOR_TOP_K))
        self.assertEqual(tuple(scores.shape), (NUM_REQS, DEPTH, SELECTOR_TOP_K, SELECTOR_TOP_K))
        self.assertEqual(scores.dtype, torch.float32)

    def test_each_beam_parent_uses_its_own_predecessor_row(self):
        candidate_ids, scores = self._scores()
        parents = torch.tensor([[ID_OFFSET, ID_OFFSET + 1]])
        logits, child_ids = _dflash2_parent_logits(candidate_ids, scores, 1, parents)
        self.assertEqual(child_ids[0, 0].tolist(), list(range(ID_OFFSET, ID_OFFSET + SELECTOR_TOP_K)))
        self.assertEqual(child_ids[0, 1].tolist(), list(range(ID_OFFSET, ID_OFFSET + SELECTOR_TOP_K)))
        self.assertEqual(logits[0, 0].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(logits[0, 1].tolist(), [5.0, 6.0, 7.0, 8.0])

    def test_anchor_selects_depth_zero_scores(self):
        candidate_ids, scores = self._scores()
        logits, _ = _dflash2_parent_logits(
            candidate_ids, scores, 0, torch.tensor([[10], [20]])
        )
        self.assertEqual(logits[0, 0].tolist(), [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(logits[1, 0].tolist(), [20.0, 21.0, 22.0, 23.0])


class TestDFlash2DraftDetection(unittest.TestCase):
    def test_dflash2_architecture_is_detected(self):
        self.assertTrue(is_dflash2_architecture(_draft_model_config()))

    def test_plain_dflash_is_not_dflash2(self):
        config = SimpleNamespace(
            architectures=["DFlashDraftModel"], hf_config=SimpleNamespace()
        )
        self.assertFalse(is_dflash2_architecture(config))

    def test_missing_selector_top_k_is_rejected(self):
        config = SimpleNamespace(
            architectures=["DFlash2DraftModel"],
            hf_config=SimpleNamespace(dflash_config={"selector_rank": 8}),
        )
        with self.assertRaisesRegex(ValueError, "selector_top_k"):
            DFlash2TreeSelector(
                config,
                num_speculative_steps=DEPTH,
                vocab_size=VOCAB_SIZE,
            )


if __name__ == "__main__":
    unittest.main()
