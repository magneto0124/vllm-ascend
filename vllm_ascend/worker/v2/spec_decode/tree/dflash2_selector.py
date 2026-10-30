# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DFlash2 candidate-selector scores for beam tree construction."""

import torch

# Architecture of the DFlash2 draft entry registered in ``vllm_ascend.models``.
DFLASH2_ARCHITECTURE = "DFlash2DraftModel"


def is_dflash2_architecture(draft_model_config) -> bool:
    """True when the draft checkpoint is a DFlash2 draft model."""
    architectures = getattr(draft_model_config, "architectures", None) or ()
    return DFLASH2_ARCHITECTURE in architectures


class DFlash2TreeSelector:
    """Return candidate shortlists and their predecessor-conditioned scores."""

    def __init__(
        self,
        draft_model_config,
        num_speculative_steps: int,
        vocab_size: int,
    ):
        draft_config = getattr(draft_model_config.hf_config, "dflash_config", None) or {}
        if "selector_top_k" not in draft_config:
            raise ValueError(
                "DFlash2 draft tree scoring requires dflash_config.selector_top_k "
                "in the draft checkpoint config."
            )
        self.top_k = int(draft_config["selector_top_k"])
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size

    def score(
        self,
        model,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``candidate_ids [R,D,K]`` and ``scores [R,D,K,K]``.

        ``hidden_states`` is ``[num_reqs * depth, hidden]`` -- the per-mask-token
        draft hidden; ``anchor_token_ids`` is each request's accepted root token,
        the predecessor that scores the first depth.
        """
        depth = self.num_speculative_steps
        hidden = hidden_states.reshape(num_reqs, depth, -1)
        candidate_ids, unary_logits = model.compute_candidates(
            hidden.reshape(num_reqs * depth, -1)
        )
        candidate_ids = candidate_ids.reshape(num_reqs, depth, -1)
        unary_logits = unary_logits.view_as(candidate_ids)
        scores = model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )
        return candidate_ids, scores
