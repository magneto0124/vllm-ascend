"""Accounting of the Ascend Mamba/GDN state cache: what is *really* allocated.

Chain MTP and draft-tree verify (``tree_spec_config``) lay the same state out
with different widths, and the question this module answers is how much device
memory the Mamba group really consumes in each mode. Every number reported is
read off a live object, never re-derived from config:

* the raw per-layer state buffer the allocator handed out and the state views
  carved out of it (``attn_utils._reshape_mamba_kv_cache``) give the real block
  count, the real per-block stride, and the conv / ssm / padding split;
* the real pool capacity is ``KVCacheConfig.num_blocks`` as assigned by the
  engine, plus the byte delta of the KV allocation itself;
* the block tables of a live step (``model_runner.prepare_attn``) give how many
  Mamba state pages requests really hold, hence the bytes really in use.

Nothing here depends on a draft tree, so a chain run and a tree run exercise the
same recorders and the counters can be compared directly; the values differ only
because the widened ``MambaSpec`` has more speculative blocks and wider conv
state pages.

Enable with ``additional_config.mamba_mem_stats_config`` (see
:class:`vllm_ascend.ascend_config.MambaMemStatsConfig`); the worker configures
the collector once, from the real KV cache groups, and the report is printed once
at process exit (per rank). The block-table part samples one step out of
``sample_interval`` because reading it back to the host synchronizes the device;
the allocation part runs once, at cache init. With no Mamba group in the KV cache
nothing is ever recorded and no report is printed, so the option is inert on
every other model.

A note on what the totals mean: an Ascend hybrid pool overlays its groups, i.e.
one block id is owned by one group at a time. The buffer total is therefore what
the Mamba group is *sized* for (``pool_blocks`` pages per layer, which is what
the memory accounting reserves), while the sampled block tables say how much of
that live requests really hold. Both are reported, so the two cannot be
confused.
"""

from __future__ import annotations

import atexit
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from vllm.logger import logger

try:  # vLLM main and 0.28 both expose the kind helper (see tree/kv_project.py)
    from vllm.v1.kv_cache_interface import KVCacheSpecKind, get_kv_cache_spec_kind
except ImportError:  # pragma: no cover - older vLLM without the helper
    KVCacheSpecKind = None  # type: ignore[assignment]
    get_kv_cache_spec_kind = None  # type: ignore[assignment]


# Layout names of a GDN/Mamba1/Mamba2/KDA state buffer; every one of them stores
# (conv sliding window, temporal/recurrent state) in this order.
_STATE_NAMES = ("conv", "ssm")


def _state_name(index: int, num_states: int, shape: tuple[int, ...]) -> str:
    """Name one state view: GDN/Mamba store (conv window, temporal state)."""
    if num_states == len(_STATE_NAMES):
        return _STATE_NAMES[index]
    if num_states == 1:
        # A pure-SSM spec has no conv state; a short-conv-only spec has nothing
        # else. The shape rank tells the two apart.
        return "ssm" if len(shape) == 3 else "conv"
    return f"state{index}"


def _is_mamba_spec(spec: Any) -> bool:
    if get_kv_cache_spec_kind is not None:
        try:
            return get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA
        except Exception:
            # An unknown spec type is not a Mamba state cache; the name check
            # below still covers an older vLLM without the kind helper.
            pass
    return type(spec).__name__ == "MambaSpec"


def _allocated_bytes() -> int:
    """Device memory currently allocated, or -1 when it cannot be read."""
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "memory_allocated"):
        try:
            return int(npu.memory_allocated())
        except Exception:  # pragma: no cover - device not ready
            return -1
    return -1


def _reserved_bytes() -> int:
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "memory_reserved"):
        try:
            return int(npu.memory_reserved())
        except Exception:  # pragma: no cover - device not ready
            return -1
    return -1


def _fmt_bytes(value: int) -> str:
    if value < 0:
        return "n/a"
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if value >= scale:
            return f"{value / scale:.3f} {unit}"
    return f"{value} B"


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole > 0 else "n/a"


@dataclass(frozen=True)
class MambaStateView:
    """One really-created state view (``conv`` or ``ssm``) of a layer buffer."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    bytes_per_block: int

    def describe(self) -> str:
        dims = "x".join(str(dim) for dim in self.shape)
        return f"{self.name}={dims} {str(self.dtype).replace('torch.', '')}"


@dataclass(frozen=True)
class MambaLayerAllocation:
    """The state buffer one Mamba layer really holds, and its internal split."""

    layer_name: str
    blocks: int
    # Real per-block stride of the buffer: this is the padded page the allocator
    # used, not a number recomputed from the spec.
    page_bytes: int
    buffer_bytes: int
    views: tuple[MambaStateView, ...]
    spec_blocks: int
    block_size: int
    # Identity of the buffer region, so layers that alias one allocation are not
    # counted twice in the totals.
    storage_ptr: int = 0

    @property
    def view_bytes_per_block(self) -> int:
        return sum(view.bytes_per_block for view in self.views)

    @property
    def padding_bytes_per_block(self) -> int:
        return self.page_bytes - self.view_bytes_per_block

    def view_bytes_per_block_named(self, name: str) -> int:
        return sum(view.bytes_per_block for view in self.views if view.name == name)

    def layout_key(self) -> tuple:
        return (
            self.page_bytes,
            tuple((view.name, view.shape, str(view.dtype), view.bytes_per_block) for view in self.views),
            self.spec_blocks,
            self.block_size,
        )


@dataclass
class _RuntimeTotals:
    samples: int = 0  # sampled steps
    rows: int = 0  # request rows observed, i.e. one per request per sample
    held_sum: int = 0
    held_max: int = 0
    # Block ids really held by the mamba groups at a sampled step, summed over
    # all requests of that step.
    held_step_sum: int = 0
    held_step_max: int = 0
    bytes_in_use_sum: int = 0
    bytes_in_use_max: int = 0
    # One entry per mamba group: (row width of its block table, bytes one block id
    # of that group really spans across its layers). Filled by the first sample.
    group_shapes: tuple[tuple[int, int], ...] = ()

    def observe(self, held: int, nbytes: int) -> None:
        self.rows += 1
        self.held_sum += held
        self.held_max = max(self.held_max, held)
        self.bytes_in_use_sum += nbytes
        self.bytes_in_use_max = max(self.bytes_in_use_max, nbytes)

    def bytes_per_request(self) -> int:
        return sum(width * per_block for width, per_block in self.group_shapes)


_LAYERS: dict[str, MambaLayerAllocation] = {}
_LAYER_ORDER: list[str] = []
_POOL_BLOCKS: int = 0
_ALLOC_DELTA: tuple[int, int] = (-1, -1)
_RUNTIME = _RuntimeTotals()
_STEP = 0
_SPEC_LABEL: str = ""
_ENABLED = False
_INTERVAL = 50
# Mamba KV cache groups of this engine: group id -> its layer names. Empty when
# the model has no Mamba/GDN state, which is what makes the option inert there.
_GROUPS: dict[int, tuple[str, ...]] = {}
_REGISTERED = False


def mamba_state_groups(kv_cache_config: Any) -> dict[int, tuple[str, ...]]:
    """The Mamba state groups of the KV cache, as ``group id -> layer names``."""
    groups: dict[int, tuple[str, ...]] = {}
    for group_id, group in enumerate(getattr(kv_cache_config, "kv_cache_groups", ())):
        if _is_mamba_spec(group.kv_cache_spec):
            groups[group_id] = tuple(group.layer_names)
    return groups


def configure_mamba_mem_stats(
    *,
    enabled: bool,
    sample_interval: int = 50,
    mamba_groups: dict[int, tuple[str, ...]] | None = None,
) -> None:
    """Turn the collector on/off from ``additional_config.mamba_mem_stats_config``.

    Called by the v2 worker with the real KV cache groups, so a model without
    Mamba/GDN state never collects anything: the request is logged as inert and
    the switch stays off.
    """
    global _ENABLED, _INTERVAL, _GROUPS, _REGISTERED
    _GROUPS = dict(mamba_groups or {})
    _INTERVAL = max(1, int(sample_interval))
    _ENABLED = bool(enabled) and bool(_GROUPS)
    if enabled and not _ENABLED:
        logger.info(
            "mamba_mem_stats_config.enabled is set but this model has no Mamba/GDN "
            "state cache, so no state memory is collected."
        )
        return
    if _ENABLED and not _REGISTERED:
        atexit.register(print_mamba_mem_report)
        _REGISTERED = True


def mamba_mem_stats_enabled() -> bool:
    return _ENABLED


def reset_mamba_mem_stats() -> None:
    """Drop everything collected so far (tests, and a second engine in-process)."""
    global _POOL_BLOCKS, _ALLOC_DELTA, _RUNTIME, _STEP, _SPEC_LABEL, _ENABLED, _GROUPS, _INTERVAL
    _LAYERS.clear()
    _LAYER_ORDER.clear()
    _POOL_BLOCKS = 0
    _ALLOC_DELTA = (-1, -1)
    _RUNTIME = _RuntimeTotals()
    _STEP = 0
    _SPEC_LABEL = ""
    _ENABLED = False
    _INTERVAL = 50
    _GROUPS.clear()


@contextmanager
def kv_allocation_region() -> Iterator[None]:
    """Measure the device-memory delta of the KV cache allocation itself."""
    if not _ENABLED:
        yield
        return
    global _ALLOC_DELTA
    before_alloc, before_reserved = _allocated_bytes(), _reserved_bytes()
    try:
        yield
    finally:
        after_alloc, after_reserved = _allocated_bytes(), _reserved_bytes()
        if before_alloc >= 0 and after_alloc >= 0:
            _ALLOC_DELTA = (after_alloc - before_alloc, after_reserved - before_reserved)


def record_mamba_allocation(
    layer_name: str,
    raw_cache: torch.Tensor,
    state_views: Sequence[torch.Tensor],
    spec: Any,
    pool_blocks: int,
) -> None:
    """Record one really-allocated state buffer and the views carved out of it.

    ``raw_cache`` is the layer's raw buffer as the allocator produced it and
    ``state_views`` are the conv/ssm views created from it, so every value below
    is read off the live objects. Dimensions that do not describe an accounted
    layout are skipped rather than guessed.
    """
    if not _ENABLED:
        return
    page_bytes = int(getattr(spec, "page_size_bytes", 0))
    buffer_bytes = int(raw_cache.numel()) * raw_cache.element_size()
    if page_bytes <= 0 or buffer_bytes <= 0 or buffer_bytes % page_bytes:
        return
    blocks = buffer_bytes // page_bytes
    if blocks <= 0:
        return
    views = []
    for index, state in enumerate(state_views):
        if state.numel() % blocks:
            return
        shape = tuple(state.shape[1:])
        views.append(
            MambaStateView(
                name=_state_name(index, len(state_views), shape),
                shape=shape,
                dtype=state.dtype,
                bytes_per_block=(state.numel() // blocks) * state.element_size(),
            )
        )
    global _POOL_BLOCKS, _SPEC_LABEL
    _POOL_BLOCKS = max(_POOL_BLOCKS, int(pool_blocks))
    if not _SPEC_LABEL:
        mamba_type = getattr(spec, "mamba_type", None)
        _SPEC_LABEL = (
            f"mamba_type={getattr(mamba_type, 'value', mamba_type)} "
            f"spec_blocks={int(getattr(spec, 'num_speculative_blocks', 0))} "
            f"block_size={int(getattr(spec, 'block_size', 0))} "
            f"page_padded={getattr(spec, 'page_size_padded', None)}"
        )
    if layer_name not in _LAYERS:
        _LAYER_ORDER.append(layer_name)
    _LAYERS[layer_name] = MambaLayerAllocation(
        layer_name=layer_name,
        blocks=blocks,
        page_bytes=page_bytes,
        buffer_bytes=buffer_bytes,
        views=tuple(views),
        spec_blocks=int(getattr(spec, "num_speculative_blocks", 0)),
        block_size=int(getattr(spec, "block_size", 0)),
        storage_ptr=int(raw_cache.data_ptr()),
    )


def _group_bytes_per_block_id(group_id: int) -> int:
    """Bytes one block id of ``group_id`` really spans, summed over its layers."""
    total = 0
    for layer_name in _GROUPS.get(group_id, ()):
        layer = _LAYERS.get(layer_name)
        if layer is not None:
            total += layer.page_bytes
    return total


def record_mamba_step(
    block_tables: Sequence[torch.Tensor],
    num_reqs: int,
) -> None:
    """Record how many Mamba state pages the live requests really hold.

    Reads a copy of the mamba group's block table rows back to the host, so it
    runs one step out of ``configure_mamba_mem_stats(sample_interval=...)`` only.
    Block id ``0`` is vLLM's null block, hence counting non-zero entries counts
    the state pages a request really owns.
    """
    if not _ENABLED or num_reqs <= 0:
        return
    global _STEP
    _STEP += 1
    # Sample the first step as well, so a run shorter than the interval still
    # reports what the live block tables held.
    if (_STEP - 1) % _INTERVAL:
        return
    # Per request, summed over every mamba group: the pages it holds and the
    # bytes those pages really span.
    held_per_row: list[int] | None = None
    bytes_per_row: list[int] | None = None
    group_shapes: list[tuple[int, int]] = []
    for group_id in _GROUPS:
        if group_id >= len(block_tables):
            continue
        table = block_tables[group_id]
        rows = min(num_reqs, table.shape[0])
        if rows <= 0:
            continue
        group_bytes = _group_bytes_per_block_id(group_id)
        group_shapes.append((int(table.shape[1]), group_bytes))
        # One small device-to-host copy per sampled step; `.tolist()` keeps it to
        # a single synchronization for the whole row block.
        held = (table[:rows] != 0).sum(dim=1).to("cpu").tolist()
        if held_per_row is None:
            held_per_row = [0] * rows
            bytes_per_row = [0] * rows
        for row, value in enumerate(held):
            held_per_row[row] += int(value)
            bytes_per_row[row] += int(value) * group_bytes
    if held_per_row is not None and bytes_per_row is not None:
        for held, nbytes in zip(held_per_row, bytes_per_row):
            _RUNTIME.observe(held, nbytes)
        held_at_step = sum(held_per_row)
        _RUNTIME.held_step_sum += held_at_step
        _RUNTIME.held_step_max = max(_RUNTIME.held_step_max, held_at_step)
        _RUNTIME.samples += 1
        _RUNTIME.group_shapes = tuple(group_shapes)


def _layout_groups() -> list[tuple[MambaLayerAllocation, list[str]]]:
    grouped: dict[tuple, list[str]] = {}
    for layer_name in _LAYER_ORDER:
        layer = _LAYERS[layer_name]
        grouped.setdefault(layer.layout_key(), []).append(layer_name)
    return [(_LAYERS[names[0]], names) for names in grouped.values()]


def build_mamba_mem_report() -> str:
    """Render the collected numbers; empty when nothing was recorded."""
    if not _LAYERS:
        return ""
    layers = [_LAYERS[name] for name in _LAYER_ORDER]
    page_total = sum(layer.page_bytes for layer in layers)
    conv_total = sum(layer.view_bytes_per_block_named("conv") for layer in layers)
    ssm_total = sum(layer.view_bytes_per_block_named("ssm") for layer in layers)
    view_total = sum(layer.view_bytes_per_block for layer in layers)
    padding_total = page_total - view_total
    # Layers can alias one allocation; count every distinct region once.
    regions: dict[tuple[int, int], int] = {}
    for layer in layers:
        regions.setdefault((layer.storage_ptr, layer.buffer_bytes), layer.buffer_bytes)
    buffer_total = sum(regions.values())

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    lines = [
        "========== mamba state cache report ==========",
        f"rank={rank} mamba_layers={len(layers)} pool_blocks={_POOL_BLOCKS} "
        f"blocks_per_layer={layers[0].blocks} mamba_groups={len(_GROUPS)} "
        f"layers_per_group={[len(names) for names in _GROUPS.values()]}",
        f"config: {_SPEC_LABEL} sample_interval={_INTERVAL}",
    ]
    alloc, reserved = _ALLOC_DELTA
    if alloc >= 0:
        lines.append(f"kv allocation delta: allocated +{_fmt_bytes(alloc)} reserved +{_fmt_bytes(reserved)}")
    lines.append(f"current device: allocated {_fmt_bytes(_allocated_bytes())} reserved {_fmt_bytes(_reserved_bytes())}")
    lines.append("-- per-block layout (identical layers merged) --")
    for layer, names in _layout_groups():
        views = " ".join(view.describe() for view in layer.views)
        lines.append(
            f"  x{len(names):<3} page={_fmt_bytes(layer.page_bytes)} "
            f"conv={_fmt_bytes(layer.view_bytes_per_block_named('conv'))} "
            f"({_pct(layer.view_bytes_per_block_named('conv'), layer.page_bytes)}) "
            f"ssm={_fmt_bytes(layer.view_bytes_per_block_named('ssm'))} "
            f"({_pct(layer.view_bytes_per_block_named('ssm'), layer.page_bytes)}) "
            f"padding={_fmt_bytes(layer.padding_bytes_per_block)} "
            f"({_pct(layer.padding_bytes_per_block, layer.page_bytes)}) [{views}]"
        )
    lines.append("-- totals over all mamba layers --")
    lines.append(
        f"  per_block: page={_fmt_bytes(page_total)} conv={_fmt_bytes(conv_total)} "
        f"ssm={_fmt_bytes(ssm_total)} padding={_fmt_bytes(padding_total)}"
    )
    lines.append(
        f"  mamba buffers: {_fmt_bytes(buffer_total)} = {len(regions)} distinct state buffers x "
        f"{_POOL_BLOCKS} blocks, {len(layers)} layers ({len(layers) - len(regions)} aliased)"
    )
    lines.append("-- runtime (sampled block tables) --")
    if _RUNTIME.rows == 0:
        lines.append("  no sample taken (no mamba group step recorded)")
    else:
        widths = [width for width, _ in _RUNTIME.group_shapes]
        lines.append(
            f"  steps_sampled={_RUNTIME.samples} reqs_per_step={_RUNTIME.rows / _RUNTIME.samples:.2f} "
            f"row_width_per_group={widths} pages_held mean={_RUNTIME.held_sum / _RUNTIME.rows:.2f} "
            f"max={_RUNTIME.held_max}"
        )
        lines.append(
            f"  state bytes in use (per request, all groups): "
            f"mean={_fmt_bytes(int(_RUNTIME.bytes_in_use_sum / _RUNTIME.rows))} "
            f"peak={_fmt_bytes(_RUNTIME.bytes_in_use_max)}"
        )
        lines.append(f"  per request at capacity: {_fmt_bytes(_RUNTIME.bytes_per_request())}")
        if _POOL_BLOCKS:
            held_mean = _RUNTIME.held_step_sum / _RUNTIME.samples
            lines.append(
                f"  block ids held by mamba state (all requests, per sampled step): "
                f"mean={held_mean:.1f} max={_RUNTIME.held_step_max} of pool_blocks={_POOL_BLOCKS} "
                f"({_pct(int(held_mean), _POOL_BLOCKS)} of the pool)"
            )
        if widths and _POOL_BLOCKS:
            caps = [_POOL_BLOCKS // width for width in widths if width > 0]
            lines.append(
                f"  mamba state can serve at most {caps} concurrent requests "
                f"(pool_blocks // row_width, mamba-side upper bound)"
            )
    lines.append("=============================================")
    return "\n".join(lines)


def print_mamba_mem_report() -> None:
    """Print the report to stdout (no file), like the tree timer report."""
    if not _ENABLED:
        return
    report = build_mamba_mem_report()
    if report:
        print(report, flush=True)
