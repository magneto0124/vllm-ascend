# SPDX-License-Identifier: Apache-2.0
"""Temporary, bounded inspection of GDN metadata retained by a FULL graph."""

from dataclasses import fields, is_dataclass


def tensor_fields(value, prefix="", depth=0):
    """Collect tensor views without reading device data during capture."""
    if depth > 8 or value is None:
        return {}
    if hasattr(value, "data_ptr") and hasattr(value, "shape"):
        return {prefix: value}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    elif is_dataclass(value):
        attributes = {field.name: getattr(value, field.name) for field in fields(value)}
        attributes.update(vars(value) if hasattr(value, "__dict__") else {})
        items = attributes.items()
    elif hasattr(value, "__dict__"):
        items = vars(value).items()
    else:
        return {}
    result = {}
    for name, child in items:
        if name == "treewy_contexts":
            continue
        result.update(tensor_fields(child, f"{prefix}.{name}".lstrip("."), depth + 1))
    return result


def pointer_changes(captured, current):
    changes = []
    for name, old in captured.items():
        new = current.get(name)
        if new is None:
            changes.append(f"{name}:missing")
        elif tuple(old.shape) != tuple(new.shape):
            changes.append(f"{name}:shape {tuple(old.shape)}->{tuple(new.shape)}")
        elif old.data_ptr() != new.data_ptr():
            changes.append(f"{name}:address")
    return changes


class TreeGraphAudit:
    def __init__(self):
        self.captured = {}
        self.steps = 0

    def record(self, metadata):
        if not isinstance(metadata, dict):
            return
        for layer, meta in metadata.items():
            spec = getattr(meta, "spec_decode_metadata", None)
            if spec is not None and getattr(spec, "tree_verify_slots", None) is not None:
                key = (layer, getattr(meta, "num_actual_tokens", None))
                # Preserve the tensor objects used by the last captured forward.
                self.captured[key] = tensor_fields(meta)
                break

    def report(self, metadata, log):
        if self.steps >= 3 or not isinstance(metadata, dict):
            return
        self.steps += 1
        for layer, meta in metadata.items():
            spec = getattr(meta, "spec_decode_metadata", None)
            if spec is None or getattr(spec, "tree_verify_slots", None) is None:
                continue
            key = (layer, getattr(meta, "num_actual_tokens", None))
            old = self.captured.get(key)
            current = tensor_fields(meta)
            if old is None:
                log("[GDN-AUDIT] step=%s capture_missing key=%s available=%s", self.steps, key, list(self.captured))
                return
            changes = pointer_changes(old, current)
            log("[GDN-AUDIT] step=%s layer=%s tensors=%s changed=%s", self.steps, layer, len(old), changes)
            # Deliberate D2H synchronization, outside capture and only three steps.
            # This diagnostic is for correctness, not benchmarking.
            for name, tensor in current.items():
                if any(word in name for word in ("init_state_indices", "num_accepted_tokens", "conv_prefix_src")):
                    previous = old.get(name)
                    captured_values = None if previous is None else previous.detach().flatten()[:16].cpu().tolist()
                    values = tensor.detach().flatten()[:16].cpu().tolist()
                    log("[GDN-AUDIT] %s captured_now=%s runtime=%s", name, captured_values, values)
            return
