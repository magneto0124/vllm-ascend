from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.worker.v2 import mamba_mem_stats as stats
from vllm_ascend.worker.v2.attn_utils import _reshape_mamba_kv_cache

_NUM_SPEC = 3
_BUDGET = 15  # tree width: slots == 1 + budget == 16
_CONV_DIM = 8192
_CONV_WIDTH_MINUS_1 = 3  # conv_kernel - 1, i.e. chain width == 4 at num_spec == 1
_SSM_SHAPE = (32, 128, 128)
_SSM_BYTES = 32 * 128 * 128 * 2
_CONV_COLUMN_BYTES = _CONV_DIM * 2
_NUM_BLOCKS = 8
# patch_mamba_config pads the mamba page up to the attention page, K + V.
_POOL_PAGE_BYTES = 2 * 1024 * 1024
_CHAIN_COLUMNS = _CONV_WIDTH_MINUS_1 + _NUM_SPEC
_TREE_COLUMNS = _CONV_WIDTH_MINUS_1 + _BUDGET + 1  # tree conv width == (kernel - 1) + slots
_EXTRA_COLUMNS = _TREE_COLUMNS - _CHAIN_COLUMNS


def _spec(conv_columns, num_spec, page_size_padded):
    """A GDN MambaSpec with the conv width and page a real run would produce."""
    return MambaSpec(
        shapes=((_CONV_DIM, conv_columns), _SSM_SHAPE),
        dtypes=(torch.bfloat16, torch.bfloat16),
        block_size=1024,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        num_speculative_blocks=num_spec,
        page_size_padded=page_size_padded,
    )


def _chain_spec(page_size_padded=_POOL_PAGE_BYTES):
    return _spec(_CHAIN_COLUMNS, _NUM_SPEC, page_size_padded)


def _tree_spec(page_size_padded=_POOL_PAGE_BYTES):
    # ``_widen_mamba_spec_for_tree`` keeps num_speculative_blocks == slots - 1 and
    # rewrites the conv shape to (kernel - 1) + slots columns.
    return _spec(_TREE_COLUMNS, _BUDGET, page_size_padded)


def _group(spec, layer_names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(layer_names))


def _allocate(spec, num_blocks=_NUM_BLOCKS):
    """Create the buffer ``_reshape_mamba_kv_cache`` would really carve up."""
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    return raw, _reshape_mamba_kv_cache(raw, spec)


@pytest.fixture(autouse=True)
def _enabled():
    stats.reset_mamba_mem_stats()
    stats.configure_mamba_mem_stats(
        enabled=True,
        sample_interval=1,
        mamba_groups={0: ("layer0",)},
    )
    yield
    stats.reset_mamba_mem_stats()


def test_allocation_split_is_read_off_the_real_views():
    spec = _chain_spec()
    raw, views = _allocate(spec)
    stats.record_mamba_allocation("layer0", raw, views, spec, pool_blocks=_NUM_BLOCKS)

    layer = stats._LAYERS["layer0"]
    assert layer.blocks == _NUM_BLOCKS
    assert layer.page_bytes == spec.page_size_bytes == _POOL_PAGE_BYTES
    assert layer.buffer_bytes == _NUM_BLOCKS * spec.page_size_bytes
    assert layer.view_bytes_per_block_named("conv") == _CHAIN_COLUMNS * _CONV_COLUMN_BYTES
    assert layer.view_bytes_per_block_named("ssm") == _SSM_BYTES
    # The rest of the page stride is padding no state view uses.
    assert layer.padding_bytes_per_block == _POOL_PAGE_BYTES - _CHAIN_COLUMNS * _CONV_COLUMN_BYTES - _SSM_BYTES
    assert [view.name for view in layer.views] == ["conv", "ssm"]
    assert layer.views[0].shape == (_CONV_DIM, _CHAIN_COLUMNS)
    assert layer.views[1].shape == _SSM_SHAPE


def test_tree_columns_are_absorbed_by_the_page_padding():
    # The pool already pads the mamba page up to the attention page, so a wider
    # conv state eats padding instead of moving the block count.
    chain, tree = _chain_spec(), _tree_spec()
    raw, views = _allocate(chain)
    stats.record_mamba_allocation("chain", raw, views, chain, pool_blocks=_NUM_BLOCKS)
    raw, views = _allocate(tree)
    stats.record_mamba_allocation("tree", raw, views, tree, pool_blocks=_NUM_BLOCKS)

    chain_layer, tree_layer = stats._LAYERS["chain"], stats._LAYERS["tree"]
    extra = _EXTRA_COLUMNS * _CONV_COLUMN_BYTES
    assert tree_layer.view_bytes_per_block_named("conv") == chain_layer.view_bytes_per_block_named("conv") + extra
    assert tree_layer.view_bytes_per_block_named("ssm") == chain_layer.view_bytes_per_block_named("ssm")
    assert tree_layer.page_bytes == chain_layer.page_bytes == _POOL_PAGE_BYTES
    assert chain_layer.padding_bytes_per_block - tree_layer.padding_bytes_per_block == extra
    assert tree_layer.spec_blocks == _BUDGET


def test_tree_columns_grow_the_page_when_nothing_pads_it():
    # No attention page to hide behind: the extra conv columns widen the real
    # stride, i.e. they cost device memory per block.
    chain, tree = _chain_spec(None), _tree_spec(None)
    raw, views = _allocate(chain)
    stats.record_mamba_allocation("chain", raw, views, chain, pool_blocks=_NUM_BLOCKS)
    raw, views = _allocate(tree)
    stats.record_mamba_allocation("tree", raw, views, tree, pool_blocks=_NUM_BLOCKS)

    chain_layer, tree_layer = stats._LAYERS["chain"], stats._LAYERS["tree"]
    extra = _EXTRA_COLUMNS * _CONV_COLUMN_BYTES
    assert chain_layer.page_bytes == _SSM_BYTES + _CHAIN_COLUMNS * _CONV_COLUMN_BYTES
    assert tree_layer.page_bytes == chain_layer.page_bytes + extra
    assert chain_layer.padding_bytes_per_block == tree_layer.padding_bytes_per_block == 0
    assert tree_layer.buffer_bytes - chain_layer.buffer_bytes == _NUM_BLOCKS * extra


def test_step_counts_only_mamba_groups_and_nonzero_blocks():
    spec = _chain_spec()
    raw, views = _allocate(spec)
    stats.record_mamba_allocation("layer0", raw, views, spec, pool_blocks=_NUM_BLOCKS)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=(
            _group(spec, ["layer0"]),
            _group(SimpleNamespace(), ["attn0"]),  # not a Mamba group: skipped
        )
    )
    # This is what the worker passes in: only the Mamba group is configured.
    stats.configure_mamba_mem_stats(
        enabled=True,
        sample_interval=1,
        mamba_groups=stats.mamba_state_groups(kv_cache_config),
    )
    assert stats.mamba_state_groups(kv_cache_config) == {0: ("layer0",)}
    mamba_table = torch.tensor([[7, 8, 9, 10], [11, 0, 0, 0]], dtype=torch.int32)
    block_tables = (mamba_table, torch.ones(2, 32, dtype=torch.int32))

    stats.record_mamba_step(block_tables, num_reqs=2)

    assert stats._RUNTIME.samples == 1
    assert stats._RUNTIME.rows == 2
    assert stats._RUNTIME.held_sum == 5  # 4 + 1 really-owned state pages
    assert stats._RUNTIME.held_max == 4
    assert stats._RUNTIME.held_step_sum == 5  # block ids the whole batch holds
    assert stats._RUNTIME.held_step_max == 5
    assert stats._RUNTIME.group_shapes == ((4, spec.page_size_bytes),)
    assert stats._RUNTIME.bytes_per_request() == 4 * spec.page_size_bytes
    assert stats._RUNTIME.bytes_in_use_max == 4 * spec.page_size_bytes


def test_sampling_interval_skips_steps():
    stats.reset_mamba_mem_stats()
    stats.configure_mamba_mem_stats(enabled=True, sample_interval=3, mamba_groups={0: ("layer0",)})
    block_tables = (torch.ones(1, 4, dtype=torch.int32),)

    for _ in range(7):
        stats.record_mamba_step(block_tables, num_reqs=1)

    assert stats._RUNTIME.samples == 3  # the first step, then 4 and 7


def test_report_carries_the_measured_numbers():
    spec = _chain_spec()
    raw, views = _allocate(spec)
    stats.record_mamba_allocation("layer0", raw, views, spec, pool_blocks=_NUM_BLOCKS)
    stats.record_mamba_step((torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),), num_reqs=1)

    report = stats.build_mamba_mem_report()

    assert f"pool_blocks={_NUM_BLOCKS}" in report
    assert f"spec_blocks={_NUM_SPEC}" in report
    assert "row_width_per_group=[4]" in report
    assert "x1 " in report  # one layer carries this layout
    # Real block ids of the pool that the mamba state held at the sampled step.
    assert f"of pool_blocks={_NUM_BLOCKS}" in report
    assert "concurrent requests" in report


def test_enabling_without_mamba_state_is_inert():
    stats.reset_mamba_mem_stats()
    stats.configure_mamba_mem_stats(enabled=True, sample_interval=1, mamba_groups={})

    assert not stats.mamba_mem_stats_enabled()
    stats.record_mamba_step((torch.ones(1, 4, dtype=torch.int32),), num_reqs=1)
    assert stats._RUNTIME.samples == 0
    assert stats.build_mamba_mem_report() == ""


def test_disabled_collector_records_nothing():
    stats.reset_mamba_mem_stats()
    stats.configure_mamba_mem_stats(enabled=False, sample_interval=1, mamba_groups={0: ("layer0",)})
    spec = _chain_spec()
    raw, views = _allocate(spec)

    stats.record_mamba_allocation("layer0", raw, views, spec, pool_blocks=_NUM_BLOCKS)
    stats.record_mamba_step((torch.ones(1, 4),), num_reqs=1)

    assert stats._LAYERS == {}
    assert stats._RUNTIME.samples == 0
    assert stats.build_mamba_mem_report() == ""
