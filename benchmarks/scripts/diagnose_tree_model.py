# SPDX-License-Identifier: Apache-2.0
"""Record and compare target outputs in separate eager/FULL processes."""

import argparse
import runpy
import re
import subprocess
import sys
from pathlib import Path


def run_model(args):
    directory = Path(args.output).resolve() / args.mode
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"Diagnostic directory is not empty: {directory}. Use a new --output directory.")
    namespace = runpy.run_path(str(Path(args.test).resolve()))
    config = namespace["main"].__globals__
    config.update(num_prompts=1, enable_profiling=False, enforce_eager=args.mode == "eager")
    tree = config["additional_config"]["tree_spec_config"]
    if not tree.get("enabled") or tree.get("gdn_state_update", "snapshot") != "snapshot":
        raise ValueError("This diagnostic currently expects enabled snapshot tree speculation.")
    if config["max_num_seqs"] != 1:
        raise ValueError("Use max_num_seqs=1 for step alignment.")
    tree.setdefault("params", {})["model_diag_dir"] = str(directory)
    config["results_dir"] = str(directory / "results")
    print(f"[MODEL-COMPARE] running {args.mode}", flush=True)
    config["main"]()


def compare_runs(output):
    # Lazy import keeps the subprocess launcher free of accelerator initialization.
    import torch

    eager_dir, full_dir = Path(output) / "eager", Path(output) / "full"
    for phase in ("after_prefill", "before_verify"):
        state_names = sorted({p.name for p in eager_dir.glob(f"rank*_{phase}.pt")} |
                             {p.name for p in full_dir.glob(f"rank*_{phase}.pt")})
        for name in state_names:
            paths = [directory / name for directory in (eager_dir, full_dir)]
            if not all(path.exists() for path in paths):
                print(f"[STATE-COMPARE] {name} MISSING_RECORD", flush=True)
                continue
            eager_states, full_states = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
            changed = 0
            keys = sorted(set(eager_states) | set(full_states))
            for key in keys:
                left, right = eager_states.get(key), full_states.get(key)
                if left is None or right is None or left.shape != right.shape:
                    print(f"[STATE-COMPARE] {name} {key} MISSING_OR_SHAPE_MISMATCH", flush=True)
                    changed += 1
                elif not torch.equal(left, right):
                    changed += 1
                    if changed <= 3:
                        delta = (left.float() - right.float()).abs()
                        print(f"[STATE-COMPARE] {name} {key} max_abs={delta.max().item():.7g} "
                              f"mean_abs={delta.mean().item():.7g}", flush=True)
            print(f"[STATE-COMPARE] {name} tensors={len(keys)} changed={changed}", flush=True)
    names = sorted({p.name for p in eager_dir.glob("rank*_step*.pt")} | {p.name for p in full_dir.glob("rank*_step*.pt")})
    if not names:
        raise RuntimeError("No recordings found: ensure model_runner.py and tree_model_diag.py were synchronized.")
    for name in names:
        paths = [directory / name for directory in (eager_dir, full_dir)]
        if not all(path.exists() for path in paths):
            print(f"[MODEL-COMPARE] {name} MISSING_RECORD", flush=True)
            continue
        eager, full = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        mismatches = []
        for key in ("input_ids", "positions", "tree_parents", "tree_num_nodes", "query_start_loc", "seq_lens"):
            left, right = eager.get(key), full.get(key)
            if (left is None) != (right is None) or (left is not None and not torch.equal(left, right)):
                mismatches.append(key)
        print(f"[MODEL-COMPARE] {name} verify={full['verify']} input_mismatch={mismatches}", flush=True)
        has_fia = "first_attention" in eager or "first_attention" in full
        if full["verify"] and (has_fia or "layer_taps" in eager or "layer_taps" in full):
            print(f"[FIA-COMPARE] {name} recorded_eager={'first_attention' in eager} "
                  f"recorded_full={'first_attention' in full}", flush=True)
            if not has_fia:
                print("[FIA-COMPARE] MISSING_RECORD: synchronize tree_model_diag.py on the worker "
                      "and rerun into a new directory; existing recordings lack FIA data.", flush=True)
            elif mismatches:
                print("[FIA-COMPARE] SKIPPED_INPUT_MISMATCH", flush=True)
        if mismatches:
            continue
        if "first_attention" in eager or "first_attention" in full:
            compare_first_attention(name, eager.get("first_attention", {}), full.get("first_attention", {}))
        if "layer_taps" in eager or "layer_taps" in full:
            left_taps, right_taps = eager.get("layer_taps", {}), full.get("layer_taps", {})
            def layer_order(key):
                match = re.search(r"layers\.(\d+)\.", key)
                return (int(match.group(1)) if match else -1, key)
            tap_names = sorted(set(left_taps) | set(right_taps), key=layer_order)
            changed = []
            for key in tap_names:
                left, right = left_taps.get(key), right_taps.get(key)
                if left is None or right is None or left.shape != right.shape or not torch.equal(left, right):
                    changed.append(key)
            print(f"[LAYER-COMPARE] {name} tensors={len(tap_names)} changed={len(changed)} "
                  f"first={changed[0] if changed else 'none'}", flush=True)
            # Show all boundaries in the first differing attention layer, plus
            # two following ones, including matching inputs needed to localize it.
            if changed:
                first_layer = layer_order(changed[0])[0]
                for key in tap_names:
                    layer_index = layer_order(key)[0]
                    if layer_index < first_layer or layer_index > first_layer + 2:
                        continue
                    left, right = left_taps.get(key), right_taps.get(key)
                    if left is None or right is None or left.shape != right.shape:
                        print(f"[LAYER-COMPARE] {key} MISSING_OR_SHAPE_MISMATCH", flush=True)
                        continue
                    delta = (left.float() - right.float()).abs()
                    print(f"[LAYER-COMPARE] {key} exact={torch.equal(left, right)} "
                          f"max_abs={delta.max().item():.7g} mean_abs={delta.mean().item():.7g} "
                          f"root_max={delta[0].max().item():.7g}", flush=True)
        for key in sorted(set(eager) | set(full)):
            if key != "hidden" and not key.startswith("aux_"):
                continue
            left, right = eager.get(key), full.get(key)
            if left is None or right is None or left.shape != right.shape:
                print(f"[MODEL-COMPARE] {key} OUTPUT_MISSING_OR_SHAPE_MISMATCH", flush=True)
                continue
            delta = (left.float() - right.float()).abs()
            # The root row is shared even when later draft branches diverge.
            print(
                f"[MODEL-COMPARE] {key} exact={torch.equal(left, right)} "
                f"finite={bool(torch.isfinite(left).all() and torch.isfinite(right).all())} "
                f"max_abs={delta.max().item():.7g} mean_abs={delta.mean().item():.7g} "
                f"root_max={delta[0].max().item():.7g}", flush=True,
            )


def compare_first_attention(name, eager, full):
    import torch

    def compare(label, left, right):
        if left is None or right is None or left.shape != right.shape:
            print(f"[FIA-COMPARE] {label} MISSING_OR_SHAPE_MISMATCH", flush=True)
            return
        delta = (left.float() - right.float()).abs()
        print(f"[FIA-COMPARE] {label} exact={torch.equal(left, right)} "
              f"max_abs={delta.max().item():.7g} mean_abs={delta.mean().item():.7g}", flush=True)

    print(f"[FIA-COMPARE] {name}", flush=True)
    for mode, record in (("eager", eager), ("full", full)):
        summary = {key: value for key, value in record.items() if not isinstance(value, torch.Tensor)}
        print(f"[FIA-COMPARE] {mode} {summary}", flush=True)
        compare(f"{mode}/actual_vs_reference", record.get("actual_out"), record.get("reference_out"))
        compare(f"{mode}/inside_vs_runtime_mask", record.get("inside_attn_mask"), record.get("runtime_mask"))
        compare(f"{mode}/inside_vs_runtime_slots", record.get("inside_slot_mapping"), record.get("runtime_slots"))
        compare(f"{mode}/inside_vs_runtime_table", record.get("inside_block_tables"), record.get("runtime_table"))
        runtime_mask = record.get("runtime_mask")
        if "expected_mask" in record:
            compare(f"{mode}/effective_vs_expected_mask", record.get("effective_mask"), record["expected_mask"])
        if runtime_mask is not None and runtime_mask.ndim == 4:
            compare(f"{mode}/effective_vs_runtime_mask", record.get("effective_mask"), runtime_mask[:, 0])
            if "expected_mask" in record:
                compare(f"{mode}/runtime_vs_expected_mask", runtime_mask[:, 0], record["expected_mask"])
        captured_mask, runtime_mask = record.get("captured_mask"), record.get("runtime_mask")
        if captured_mask is not None and runtime_mask is not None and runtime_mask.ndim == 4:
            compare(f"{mode}/captured_vs_runtime_mask", captured_mask, runtime_mask[:, 0])
    for key in ("key_cache", "value_cache", "runtime_mask", "effective_mask", "reference_out"):
        compare(f"eager_vs_full/{key}", eager.get(key), full.get(key))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="test.py")
    parser.add_argument("--output", required=True, help="New directory for local recordings")
    parser.add_argument("--mode", choices=("all", "eager", "full", "compare"), default="all")
    args = parser.parse_args()
    if args.mode == "all":
        for mode in ("eager", "full"):
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--test", str(Path(args.test).resolve()),
                "--output", str(Path(args.output).resolve()), "--mode", mode,
            ], check=True)
        compare_runs(args.output)
    elif args.mode == "compare":
        compare_runs(args.output)
    else:
        run_model(args)


if __name__ == "__main__":
    main()
