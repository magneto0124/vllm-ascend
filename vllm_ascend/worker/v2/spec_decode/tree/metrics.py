# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree graph and GDN cache statistics from host metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from math import prod
from typing import TYPE_CHECKING, Any

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass
class _GraphPhaseStats:
    captures: int = 0
    replays: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    workspaces: dict[Hashable, int] = field(default_factory=dict)


class TreeGraphStats:
    """One runner-owned counter set, updated only outside graph execution.

    Callers use a fixed phase name and one workspace key per retained gear.
    No per-step samples or tensor values are retained. Workspace bytes describe
    explicitly owned tensor storage, not allocator peaks or graph-pool usage.
    Capture/replay counts record successful submissions, not synchronized
    device completions. Performance timing belongs to an external profiler.
    """

    def __init__(self) -> None:
        self._phases: dict[str, _GraphPhaseStats] = defaultdict(_GraphPhaseStats)

    def record_lookup(self, phase: str, hit: bool) -> None:
        stats = self._phases[phase]
        if hit:
            stats.cache_hits += 1
        else:
            stats.cache_misses += 1

    def record_capture(self, phase: str) -> None:
        self._phases[phase].captures += 1

    def record_replay(self, phase: str) -> None:
        self._phases[phase].replays += 1

    def set_workspace_bytes(self, phase: str, key: Hashable, nbytes: int) -> None:
        """Replace one gear's retained-storage accounting using host metadata."""
        if nbytes < 0:
            raise ValueError("Workspace size cannot be negative")
        self._phases[phase].workspaces[key] = nbytes

    @staticmethod
    def tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
        """Count allocated storage once for explicitly provided tensor aliases."""
        storage_bytes = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            storage_bytes[key] = storage.nbytes()
        return sum(storage_bytes.values())

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return host counters and aggregate retained bytes, without graph keys."""
        result = {}
        for phase, stats in self._phases.items():
            result[phase] = {
                "captures": stats.captures,
                "replays": stats.replays,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "workspace_gears": len(stats.workspaces),
                "workspace_bytes": sum(stats.workspaces.values()),
            }
        return result


def collect_gdn_tree_cache_stats(
    kv_cache_config: KVCacheConfig, forward_context: dict[str, Any]
) -> dict[str, int]:
    """Measure this worker's persistent GDN cache and per-request page usage."""
    components = ("conv", "ssm", "replay")
    pool_bytes = dict.fromkeys(components, 0)
    request_bytes = dict.fromkeys(components, 0)
    pages_per_request = 0
    request_page_bytes = 0
    layer_count = 0
    tensors = []
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        for name in group.layer_names:
            spec = group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            if not isinstance(spec, MambaSpec) or spec.mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
                continue
            states = forward_context[name].kv_cache
            layer_count += 1
            pages = 1 + spec.num_speculative_blocks
            pages_per_request = max(pages_per_request, pages)
            request_page_bytes += pages * spec.page_size_bytes
            for component, state in zip(components, states):
                pool_bytes[component] += state.nbytes
                request_bytes[component] += pages * prod(state.shape[1:]) * state.element_size()
                tensors.append(state)

    return {
        "gdn_layers": layer_count,
        "cache_blocks": kv_cache_config.num_blocks,
        "max_pages_per_request_per_layer": pages_per_request,
        "request_page_bytes": request_page_bytes,
        "request_payload_bytes": sum(request_bytes.values()),
        **{f"request_{component}_bytes": count for component, count in request_bytes.items()},
        "pool_payload_bytes": sum(pool_bytes.values()),
        **{f"pool_{component}_bytes": count for component, count in pool_bytes.items()},
        "shared_backing_storage_bytes": TreeGraphStats.tensor_bytes(tensors),
    }
