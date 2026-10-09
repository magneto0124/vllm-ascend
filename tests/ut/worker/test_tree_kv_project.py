import math
from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.worker.v2.spec_decode.tree.kv_project import TreeKvCompact

_BLOCK_SIZE = 16
_NUM_BLOCKS = 16
_CONV_DIM = 8192
# Short-conv ring buffer of a 4-wide kernel with num_speculative_blocks = 3.
_CONV_STATE_LEN = 3 + 3
# Identity markers: the test only asserts that the state stays untouched.
_SSM_STATE_SHAPE = (2, 4, 32, 32)


class _Layer:
    def __init__(self, kv_cache):
        self.kv_cache = kv_cache


def _runner(attn_layer, mamba_layer, spec_len=4):
    block_table = torch.arange(_NUM_BLOCKS, dtype=torch.int32).reshape(2, _NUM_BLOCKS // 2)
    return SimpleNamespace(
        device=torch.device("cpu"),
        num_speculative_steps=spec_len,
        max_num_reqs=2,
        compilation_config=SimpleNamespace(
            static_forward_context={"attn": attn_layer, "mamba": mamba_layer}
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["attn"],
                    kv_cache_spec=FullAttentionSpec(
                        block_size=_BLOCK_SIZE,
                        num_kv_heads=2,
                        head_size=128,
                        dtype=torch.bfloat16,
                    ),
                ),
                KVCacheGroupSpec(
                    layer_names=["mamba"],
                    kv_cache_spec=MambaSpec(
                        shapes=((_CONV_DIM, _CONV_STATE_LEN), _SSM_STATE_SHAPE),
                        dtypes=(torch.bfloat16, torch.bfloat16),
                        block_size=1024,
                        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                        num_speculative_blocks=3,
                    ),
                ),
            ]
        ),
        block_tables=SimpleNamespace(
            block_tables=(SimpleNamespace(gpu=block_table), SimpleNamespace(gpu=block_table)),
            kernel_block_sizes=(_BLOCK_SIZE, _BLOCK_SIZE),
        ),
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=torch.tensor([16, 32], dtype=torch.int32))),
    )


def _uniform_mamba_runner(attn_layer, mamba_layer, spec_len=4):
    runner = _runner(attn_layer, mamba_layer, spec_len)
    mamba_spec = runner.kv_cache_config.kv_cache_groups[1].kv_cache_spec
    runner.kv_cache_config.kv_cache_groups[1] = KVCacheGroupSpec(
        layer_names=["mamba"],
        kv_cache_spec=UniformTypeKVCacheSpecs.from_specs({"mamba": mamba_spec}),
    )
    return runner


def _fixture():
    attn_cache = torch.arange(
        _NUM_BLOCKS * _BLOCK_SIZE * 2 * 128, dtype=torch.float32
    ).reshape(_NUM_BLOCKS, _BLOCK_SIZE, 2, 128)
    conv_state = torch.arange(
        _NUM_BLOCKS * _CONV_STATE_LEN * _CONV_DIM, dtype=torch.float32
    ).reshape(_NUM_BLOCKS, _CONV_STATE_LEN, _CONV_DIM)
    ssm_state = torch.arange(
        _NUM_BLOCKS * math.prod(_SSM_STATE_SHAPE), dtype=torch.float32
    ).reshape(_NUM_BLOCKS, *_SSM_STATE_SHAPE)
    return attn_cache, conv_state, ssm_state


def test_bind_groups_skips_mamba_state_groups():
    attn_cache, conv_state, ssm_state = _fixture()
    compact = TreeKvCompact(_runner(_Layer(attn_cache), _Layer((conv_state, ssm_state))))

    compact._bind_groups()

    # Only the token-major attention group is compacted; mamba state pages are
    # addressed by request state slot and must be left alone.
    assert len(compact._groups) == 1


def test_bind_groups_skips_uniform_mamba_state_groups():
    attn_cache, conv_state, ssm_state = _fixture()
    compact = TreeKvCompact(
        _uniform_mamba_runner(_Layer(attn_cache), _Layer((conv_state, ssm_state)))
    )

    compact._bind_groups()

    assert len(compact._groups) == 1


def test_compact_moves_accepted_path_and_keeps_mamba_state():
    attn_cache, conv_state, ssm_state = _fixture()
    compact = TreeKvCompact(_runner(_Layer(attn_cache), _Layer((conv_state, ssm_state))))
    attn_before = attn_cache.clone()
    conv_before = conv_state.clone()
    ssm_before = ssm_state.clone()

    compact.run(
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        path_node_ids=torch.tensor([[2, 5, 6, -1], [4, -1, -1, -1]]),
    )

    assert torch.equal(conv_state, conv_before)
    assert torch.equal(ssm_state, ssm_before)
    # Request 0 has num_computed = 16, so accepted nodes 2/5/6 sit at token
    # positions 18/21/22 and land on the linear slots 17/18/19.
    assert torch.equal(attn_cache[1, 1], attn_before[1, 2])
    assert torch.equal(attn_cache[1, 2], attn_before[1, 5])
    assert torch.equal(attn_cache[1, 3], attn_before[1, 6])


def test_compact_moves_both_k_and_v_views_of_shared_allocation():
    attn_cache, conv_state, ssm_state = _fixture()
    # This is the hybrid allocation layout: K and V share a storage pointer,
    # but their views start at different offsets and must both be compacted.
    kv_backing = torch.stack((attn_cache, attn_cache + 1_000_000))
    k_cache, v_cache = kv_backing[0], kv_backing[1]
    assert k_cache.untyped_storage().data_ptr() == v_cache.untyped_storage().data_ptr()
    k_before, v_before = k_cache.clone(), v_cache.clone()
    compact = TreeKvCompact(_runner(_Layer((k_cache, v_cache)), _Layer((conv_state, ssm_state))))

    compact.run(
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        path_node_ids=torch.tensor([[2, 5, 6, -1], [4, -1, -1, -1]]),
    )

    assert len(compact._groups[0][0]) == 2
    for cache, before in ((k_cache, k_before), (v_cache, v_before)):
        assert torch.equal(cache[1, 1], before[1, 2])
        assert torch.equal(cache[1, 2], before[1, 5])
        assert torch.equal(cache[1, 3], before[1, 6])


def test_compact_skips_unallocated_tail_and_masked_request():
    attn_cache, conv_state, ssm_state = _fixture()
    # Only two logical blocks are available per request. The accepted node
    # at position 31 must move to position 30, but the unused path columns
    # would otherwise look up logical block 2 (past the table's end).
    runner = _runner(_Layer(attn_cache), _Layer((conv_state, ssm_state)))
    runner.req_states.num_computed_tokens.gpu = torch.tensor([29, 29], dtype=torch.int32)
    runner.block_tables.block_tables = (
        SimpleNamespace(gpu=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)),
        runner.block_tables.block_tables[1],
    )
    before = attn_cache.clone()

    TreeKvCompact(runner).run(
        idx_mapping=torch.tensor([0, -1], dtype=torch.int32),
        path_node_ids=torch.tensor([[2, -1, -1, -1], [1, -1, -1, -1]]),
    )

    assert torch.equal(attn_cache[2, 14], before[2, 15])
    assert torch.equal(attn_cache[0], before[0])  # reserved null block
    assert torch.equal(attn_cache[3:], before[3:])  # masked request


def test_compact_all_rejected_at_block_table_limit():
    attn_cache, conv_state, ssm_state = _fixture()
    runner = _runner(_Layer(attn_cache), _Layer((conv_state, ssm_state)))
    runner.req_states.num_computed_tokens.gpu = torch.tensor([32, 32], dtype=torch.int32)
    runner.block_tables.block_tables = (
        SimpleNamespace(gpu=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)),
        runner.block_tables.block_tables[1],
    )
    before = attn_cache.clone()

    TreeKvCompact(runner).run(
        idx_mapping=torch.tensor([0, -1], dtype=torch.int32),
        path_node_ids=torch.full((2, 4), -1),
    )

    assert torch.equal(attn_cache, before)


def test_compact_does_not_gather_stale_half_cache_block_for_unused_path():
    _, conv_state, ssm_state = _fixture()
    attn_cache = torch.arange(_NUM_BLOCKS * _BLOCK_SIZE, dtype=torch.float16).reshape(
        _NUM_BLOCKS, _BLOCK_SIZE, 1, 1
    )
    runner = _runner(_Layer(attn_cache), _Layer((conv_state, ssm_state)))
    runner.req_states.num_computed_tokens.gpu = torch.tensor([29, 29], dtype=torch.int32)
    runner.block_tables.block_tables = (
        SimpleNamespace(
            gpu=torch.tensor([[1, 2, _NUM_BLOCKS + 1], [3, 4, 5]], dtype=torch.int32)
        ),
        runner.block_tables.block_tables[1],
    )
    before = attn_cache.clone()

    TreeKvCompact(runner).run(
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        path_node_ids=torch.tensor([[2, -1, -1, -1], [-1, -1, -1, -1]]),
    )

    assert torch.equal(attn_cache[2, 14], before[2, 15])
    assert torch.equal(attn_cache[0], before[0])
    assert torch.equal(attn_cache[3:], before[3:])


def test_bind_groups_keeps_distinct_layer_views_and_deduplicates_aliases():
    attn_cache, conv_state, ssm_state = _fixture()
    backing = torch.stack([attn_cache] * 4)
    first_layer = _Layer((backing[0], backing[1]))
    second_layer = _Layer((backing[2], backing[3]))
    runner = _runner(first_layer, _Layer((conv_state, ssm_state)))
    runner.compilation_config.static_forward_context.update(
        attn_second=second_layer, attn_alias=first_layer
    )
    runner.kv_cache_config.kv_cache_groups[0].layer_names.extend(["attn_second", "attn_alias"])
    compact = TreeKvCompact(runner)

    compact._bind_groups()

    caches = compact._groups[0][0]
    assert [cache.data_ptr() for cache in caches] == [backing[i].data_ptr() for i in range(4)]
