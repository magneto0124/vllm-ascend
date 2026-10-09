import atexit
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch

TREE_TIMER_SEGMENTS: tuple[str, ...] = (
    "build_precompute",
    "build_expand_score",
    "build_expand_select",
    "build_expand_gru",
    "build_prune",
    "build_finalize_layout",
    "build_draft_tree",  # non-prefix builders (beam / priority)
    "build_attn_mask",
    "target_fia_forward",
    "rejection_sample",
    "compact_kv_path",
    "compact_query_path",
    "draft_model_forward",
)

_ENABLED = False
_BACKEND = "torch"
_WARMUP_STEPS = 2
_STEP = 0
_RECORDING = False
_SAMPLES: dict[str, list[float]] = defaultdict(list)
_ACCUM: dict[str, float] = defaultdict(float)
_META: dict[str, object] = {}
_REGISTERED = False
_STAT_COUNT: dict[str, int] = defaultdict(int)
_STAT_SUM: dict[str, float] = defaultdict(float)


def configure_tree_timer(
    *,
    enabled: bool = False,
    backend: str = "torch",
    warmup_steps: int = 2,
    meta: dict[str, object] | None = None,
) -> None:
    """Enable/disable the tree-spec segment timer.

    ``backend`` is printed as ``torch`` or ``triton`` (implementation path).
    Segments always fence with ``torch.npu.synchronize`` so samples include
    host wall-clock and device compute wait.
    """
    global _ENABLED, _BACKEND, _WARMUP_STEPS, _REGISTERED
    _ENABLED = bool(enabled)
    _BACKEND = backend
    _WARMUP_STEPS = max(0, int(warmup_steps))
    if meta:
        _META.update(meta)
    if _ENABLED and not _REGISTERED:
        atexit.register(print_tree_timer_report)
        _REGISTERED = True


def tree_timer_enabled() -> bool:
    return _ENABLED


def tree_timer_begin_step() -> None:
    """Mark the start of one decode step (verify + propose)."""
    global _STEP, _RECORDING
    if not tree_timer_enabled():
        return
    _STEP += 1
    _RECORDING = _STEP > _WARMUP_STEPS
    _ACCUM.clear()


def tree_stat(name: str, value: float = 1.0) -> None:
    """Count one shape observation for the exit report.

    The report prints ``sum`` and ``mean`` per name, so a boolean observation
    counts how often it held and a per-step count averages over the steps. Free
    when the timer is disabled -- but the *caller* has to keep any device
    synchronization (``.item()``, ``bool(...)``) behind
    :func:`tree_timer_enabled`, because the early return here cannot undo it.
    """
    if not _ENABLED:
        return
    _STAT_COUNT[name] += 1
    _STAT_SUM[name] += float(value)


def _sync() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        try:
            torch.npu.synchronize()
            return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def tree_time(segment: str) -> Iterator[None]:
    """Time one segment with device synchronize; no-op when disabled."""
    if not tree_timer_enabled():
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        if _RECORDING:
            _SAMPLES[segment].append((time.perf_counter() - t0) * 1000.0)


@contextmanager
def tree_time_accum(segment: str) -> Iterator[None]:
    """Accumulate timed spans into one per-step sample (for expand depth loops)."""
    if not tree_timer_enabled():
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        if _RECORDING:
            _ACCUM[segment] += (time.perf_counter() - t0) * 1000.0


def tree_time_accum_flush() -> None:
    """Flush accumulated expand sub-segments as one sample each."""
    if not tree_timer_enabled() or not _RECORDING:
        _ACCUM.clear()
        return
    for name, ms in _ACCUM.items():
        _SAMPLES[name].append(ms)
    _ACCUM.clear()


def print_tree_timer_report() -> None:
    """Print aggregated segment timings to stdout (no file)."""
    if not _ENABLED:
        return
    lines = [
        "========== tree-spec timer report ==========",
        f"backend={_BACKEND} sync=device steps={_STEP} warmup={_WARMUP_STEPS}",
    ]
    if _META:
        meta_str = " ".join(f"{k}={v}" for k, v in sorted(_META.items()))
        lines.append(f"config: {meta_str}")
    any_data = False
    for name in TREE_TIMER_SEGMENTS:
        vals = list(_SAMPLES.get(name, []))
        if not vals:
            lines.append(f"  {name}: n=0")
            continue
        any_data = True
        n = len(vals)
        mean = sum(vals) / n
        lines.append(
            f"  {name}: n={n} mean_ms={mean:.3f} "
            f"min_ms={min(vals):.3f} max_ms={max(vals):.3f}"
        )
    if not any_data:
        lines.append("  (no samples after warmup)")
    if _STAT_COUNT:
        # Shape counters: what the draft tree really looked like. A tree that
        # collapsed to a chain, or whose layout no longer matches the tokens the
        # scheduler put in the batch, is invisible in the timings above.
        lines.append("-- shape counters (sum=total, mean=per step) --")
        for name in sorted(_STAT_COUNT):
            n = _STAT_COUNT[name]
            total = _STAT_SUM[name]
            lines.append(f"  {name}: n={n} sum={total:g} mean={total / n:.3f}")
    lines.append("===========================================")
    print("\n".join(lines), flush=True)
