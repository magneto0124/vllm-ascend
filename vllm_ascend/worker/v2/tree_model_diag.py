# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded target-output recording outside compilation/capture."""

from pathlib import Path
from inspect import signature

import torch
from vllm.distributed import get_world_group
from vllm.logger import logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


class TreeModelRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.step = 0
        self.verify_steps = 0
        self.prefill_pages = {}
        self.layer_taps = {}
        self.tap_width = None
        self.first_attention = None
        self.attention_taps = {}

    def _tap(self, name, value):
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != self.tap_width:
            return
        # Executed inside the attention custom-op body, not the Dynamo-traced
        # model forward. copy_ becomes a graph node and runs on every replay.
        buffer = self.layer_taps.get(name)
        if buffer is None:
            buffer = torch.empty_like(value)
            self.layer_taps[name] = buffer
        buffer.copy_(value)

    def _wrap_attention(self, original, name, inputs, output_name):
        call_signature = signature(original)

        def traced(*args, **kwargs):
            bound = call_signature.bind(*args, **kwargs).arguments
            for field in inputs:
                self._tap(f"{name}/in/{field}", bound.get(field))
            if self.first_attention is not None and name == f"{self.first_attention}/ATTN":
                query = bound.get("query")
                metadata = bound.get("attn_metadata")
                if metadata is not None and query.shape[0] == self.tap_width:
                    # Device copies become graph nodes. These record the metadata
                    # seen inside replay, not just the host metadata prepared for it.
                    for field in ("attn_mask", "slot_mapping", "block_tables"):
                        value = getattr(metadata, field, None)
                        if isinstance(value, torch.Tensor):
                            buffer = self.attention_taps.get(field)
                            if buffer is None:
                                buffer = torch.empty_like(value)
                                self.attention_taps[field] = buffer
                            buffer.copy_(value)
            result = original(*args, **kwargs)
            output = bound.get(output_name)
            self._tap(f"{name}/out", output if output is not None else result)
            return result

        return traced

    def install_layer_taps(self, runner):
        self.tap_width = runner.decode_query_len
        target_ids = {id(module) for module in runner.model.modules()}
        layers = runner.vllm_config.compilation_config.static_forward_context
        seen = set()
        gdn_count = attention_count = 0
        for name, layer in layers.items():
            if id(layer) not in target_ids:
                continue
            if hasattr(layer, "_forward_core") and hasattr(layer, "conv1d"):
                layer._forward_core = self._wrap_attention(
                    layer._forward_core, f"{name}/GDN", ("mixed_qkv", "b", "a"), "core_attn_out"
                )
                gdn_count += 1
            elif hasattr(layer, "impl") and callable(getattr(layer.impl, "forward", None)):
                if id(layer.impl) in seen:
                    raise RuntimeError("Layer diagnostic does not support shared attention implementation instances")
                seen.add(id(layer.impl))
                if self.first_attention is None:
                    self.first_attention = name
                layer.impl.forward = self._wrap_attention(
                    layer.impl.forward, f"{name}/ATTN", ("query", "key", "value"), "output"
                )
                attention_count += 1
        logger.warning("[LAYER-DIAG] installed GDN=%s ATTN=%s width=%s", gdn_count, attention_count, self.tap_width)

    def _record_first_attention(self, runner, state):
        # Worker-only imports keep this optional diagnostic out of launcher setup.
        from vllm_ascend.compilation.acl_graph import get_graph_params

        name = self.first_attention
        if name is None or name not in state.attn_metadata:
            return {}
        impl = runner.vllm_config.compilation_config.static_forward_context[name].impl
        metadata = state.attn_metadata[name]
        if metadata.num_decodes != 1 or metadata.num_prefills:
            return {"unsupported": "Expected one tree decode request"}
        prefix = f"{name}/ATTN"
        query = self.layer_taps[f"{prefix}/in/query"].view(self.tap_width, impl.num_heads, impl.head_size)
        current_key = self.layer_taps[f"{prefix}/in/key"]
        current_value = self.layer_taps[f"{prefix}/in/value"]
        key, value, block_size, table, kv_lens = impl._get_fia_params(current_key, current_value, metadata)
        length = int(metadata.seq_lens_list[0])
        blocks = (length + block_size - 1) // block_size
        indices = table[0, :blocks].to(dtype=torch.long)
        data = {
            "layer": name,
            "runtime_q_lens": list(metadata.actual_seq_lengths_q),
            "runtime_kv_lens": list(metadata.seq_lens_list),
            "num_decodes": metadata.num_decodes,
            "num_prefills": metadata.num_prefills,
            "block_size": block_size,
        }
        prepare = runner.model_state.prepare_attn
        prepare = getattr(prepare, "__func__", prepare)
        code = getattr(prepare, "__code__", None)
        visibility = getattr(state.input_batch, "tree_visibility", None)
        data["prepare_file"] = code.co_filename if code is not None else "unknown"
        data["prepare_forwards_visibility"] = bool(
            code is not None and any(
                isinstance(value, tuple) and "tree_visibility" in value
                for value in code.co_consts
            )
        )
        data["visibility_shape"] = tuple(visibility.shape) if visibility is not None else None
        data["runtime_mask_shape"] = tuple(metadata.attn_mask.shape) if metadata.attn_mask is not None else None
        for field, cache in (("key_cache", key), ("value_cache", value)):
            # Compare logical, valid KV rows only; page IDs and unused memory
            # can differ between independent eager and FULL processes.
            data[field] = cache.index_select(0, indices).reshape(-1, cache.shape[-1])[:length].cpu().clone()
        for field, tensor in self.attention_taps.items():
            data[f"inside_{field}"] = tensor.cpu().clone()
        data["runtime_mask"] = metadata.attn_mask.cpu().clone()
        effective_mask = impl._bsnd_tree_attn_mask(metadata.attn_mask, 1, self.tap_width)
        if effective_mask is not None:
            data["effective_mask"] = effective_mask.cpu().clone()
            data["effective_mask_shape"] = tuple(effective_mask.shape)
        # Build an independent host-side oracle from the actual input tree.
        # Do not refresh the singleton mask: that would hide stale-buffer bugs.
        if visibility is not None and effective_mask is not None:
            expected = torch.ones_like(data["effective_mask"], dtype=torch.bool)
            previous = length - self.tap_width
            expected[:, :, :previous + 1] = False
            nodes = visibility.shape[-1]
            expected[:, 1:1 + nodes, previous + 1:previous + 1 + nodes] = ~visibility.cpu().bool()
            data["expected_mask"] = expected
        logger.warning(
            "[MASK-TRACE] file=%s forwarded=%s visibility=%s runtime_mask=%s effective_mask=%s",
            data["prepare_file"], data["prepare_forwards_visibility"], data["visibility_shape"],
            data["runtime_mask_shape"], data.get("effective_mask_shape"),
        )
        data["runtime_slots"] = metadata.slot_mapping.cpu().clone()
        data["runtime_table"] = table.cpu().clone()
        params = get_graph_params()
        captured = params.attn_params.get(self.tap_width, []) if params is not None else []
        # Match cache storage, never assume dictionary order equals capture order.
        matches = [p for p in captured if isinstance(p, tuple) and len(p) == 22
                   and p[1].data_ptr() == key.data_ptr()]
        data["captured_matches"] = len(matches)
        if len(matches) == 1:
            param = matches[0]
            data["capture_layout"] = "BSND" if param[0].ndim == 4 else "TND"
            data["capture_sparse_mode"] = param[13]
            data["capture_window"] = (param[14], param[15])
            if param[4] is not None:
                data["captured_mask"] = param[4].cpu().clone()
        # Recompute only FIA on the SAME query and already-written KV. This
        # neither reruns the model nor changes recurrent/KV state or sampling.
        reference = impl._forward_tree_decode_fia_impl(
            query, key, value, current_value, block_size, table, metadata,
            kv_lens, self.tap_width, metadata.num_decodes,
        )
        data["reference_out"] = reference.reshape(self.tap_width, -1).cpu().clone()
        data["actual_out"] = self.layer_taps[f"{prefix}/out"].reshape(self.tap_width, -1).cpu().clone()
        return data

    def _save_gdn_pages(self, runner, phase):
        """Read only pages written by prefill, excluding uninitialized tree slots."""
        layers = runner.vllm_config.compilation_config.static_forward_context
        data = {}
        for name, pages in self.prefill_pages.items():
            layer = layers.get(name)
            if layer is None or not hasattr(layer, "conv1d"):
                continue
            conv, recurrent = layer.kv_cache
            indices = torch.tensor(pages, device=recurrent.device, dtype=torch.long)
            data[f"{name}/recurrent"] = recurrent.index_select(0, indices).detach().cpu()
            history = conv.index_select(0, indices.to(conv.device))
            width = layer.conv1d.weight.shape[-1] - 1
            history = history[:, :, :width] if is_conv_state_dim_first() else history[:, :width, :]
            data[f"{name}/conv_prefix"] = history.detach().cpu()
        self.directory.mkdir(parents=True, exist_ok=True)
        rank = get_world_group().rank_in_group
        torch.save(data, self.directory / f"rank{rank}_{phase}.pt")
        logger.warning("[STATE-DIAG] rank=%s phase=%s tensors=%s", rank, phase, len(data))

    def before_forward(self, runner, scheduler_output):
        if self.verify_steps == 0 and scheduler_output.scheduled_spec_decode_tokens:
            self._save_gdn_pages(runner, "before_verify")

    def record(self, runner, scheduler_output):
        state = runner.execute_model_state
        if state is None or self.verify_steps >= 3:
            return
        batch = state.input_batch
        is_verify = bool(scheduler_output.scheduled_spec_decode_tokens)
        if not is_verify:
            metadata = getattr(state, "attn_metadata", None)
            if isinstance(metadata, dict):
                for name, meta in metadata.items():
                    indices = getattr(meta, "prefill_state_indices", None)
                    if indices is not None:
                        self.prefill_pages[name] = indices.detach().flatten().cpu().tolist()
                self._save_gdn_pages(runner, "after_prefill")
        self.verify_steps += int(is_verify)
        rank = get_world_group().rank_in_group
        data = {"verify": is_verify, "step": self.step, "rank": rank}
        # Only real tokens: ignore differing FULL padding between runs.
        count = batch.num_tokens
        for name in ("input_ids", "positions"):
            tensor = getattr(batch, name, None)
            if tensor is None and name == "input_ids":
                tensor = runner.input_buffers.input_ids
            if tensor is not None:
                data[name] = tensor[..., :count].detach().cpu().clone()
        for name in ("tree_parents", "tree_num_nodes"):
            tensor = getattr(batch, name, None)
            if tensor is not None:
                data[name] = tensor.detach().cpu().clone()
        data["query_start_loc"] = batch.query_start_loc[:batch.num_reqs + 1].detach().cpu().clone()
        data["seq_lens"] = batch.seq_lens[:batch.num_reqs].detach().cpu().clone()
        if state.hidden_states is not None:
            data["hidden"] = state.hidden_states[:count].detach().cpu().clone()
        for index, tensor in enumerate(state.aux_hidden_states or []):
            data[f"aux_{index}"] = tensor[:count].detach().cpu().clone()
        if is_verify and self.verify_steps == 1:
            data["layer_taps"] = {name: tensor.detach().cpu().clone() for name, tensor in self.layer_taps.items()}
            if self.first_attention is not None:
                data["first_attention"] = self._record_first_attention(runner, state)
            logger.warning("[LAYER-DIAG] first verify tensors=%s", len(self.layer_taps))
        # Sampling may mutate these views, so save synchronously before returning.
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"rank{rank}_step{self.step:03d}.pt"
        torch.save(data, path)
        logger.warning("[MODEL-DIAG] rank=%s step=%s verify=%s saved=%s", rank, self.step, is_verify, path.name)
        self.step += 1
