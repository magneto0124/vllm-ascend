# GDN 状态矩阵在树投机解码中的回滚适配（设计说明）

> 对应提交：`6d2c1f7af14f8bde0361a27febc4c0efa3e8c7b5`
> `[Feature] gdn state snapshots for tree sd`
>
> 本文从"为什么这么设计"讲到"每一行代码在干什么"，读完应当可以独立把这个功能重新实现一遍。

---

## 目录

- [1. 背景与问题定义](#1-背景与问题定义)
- [2. 术语与变量约定](#2-术语与变量约定)
- [3. 总体设计思路](#3-总体设计思路)
- [4. 实现细节](#4-实现细节)
- [5. 注意事项与踩坑清单](#5-注意事项与踩坑清单)
- [6. 测试覆盖](#6-测试覆盖)
- [7. 动手实现的最小步骤清单](#7-动手实现的最小步骤清单)
- [8. 已知限制](#8-已知限制)

---

## 1. 背景与问题定义

### 1.1 树投机解码（draft tree）的验证批

`tree_spec_config` 打开后，投机解码不再验证一条线性链，而是一棵草稿树。目标模型每一步为
每个请求验证 **`1 + budget` 个 token**：

- 第 0 个 token 是 **root**，即上一步已经接受、已经提交的那个 token（重新验证一次）；
- 其后 `budget` 个是树上的 **非 root 节点**，节点 id 从 1 开始编号；
- 节点按拓扑序（**保证 `parent_id < node_id`**）排在验证批里，接受判定会走出一条
  **root → leaf 的路径**（而不是"前缀"）。

采样器（`worker/v2/spec_decode/tree/rejection_sampler.py`）在这一步结束后产出两个关键量：

| 变量 | 形状 | 含义 |
| --- | --- | --- |
| `path_node_ids` | `[R, spec_len]` | `path_node_ids[r, k]` = 该请求被接受路径上**深度为 `k+1` 的节点 id**；未用到的槽位是 `-1`。节点 0 是 root，所以它永远不会出现在这里 |
| `num_sampled` | `[R]` | 这一步该请求采出的 token 数，**含末尾那次重采样（resample）**，因此"最后一个被接受的 draft 节点"落在索引 `num_sampled - 2` |

### 1.2 GDN 有两份状态要跟着走

Qwen GDN（`MambaAttentionBackendEnum.GDN_ATTN`）的线性注意力层缓存两份状态：

1. **recurrent state（SSM state）**：递推状态矩阵，一个 page（block）存一份，形状
   `[blocks, NV, Dv, Dk]`，由 `block_table` 的列来寻址。线性链 MTP 里，验证批中
   **第 `k` 个 token 的状态写在 `block_table` 的第 `k` 列**，并且第 `k` 个 token 用
   **第 `k-1` 个 token 的列**作为自己的初始状态（相邻 token 在寄存器里直接 carry）。
2. **short conv state**：短卷积的滑动窗口，形状 `(width - 1 + num_spec) × conv_dim`
   （或 dim-first 转置），`width` 是卷积核宽度（`conv1d.weight` 的 tap 数）。

### 1.3 线性链的机制为什么在树上失效

- **recurrent**：内核默认"token `seq_i` 的初始状态 = 前一个 token 写下的状态"。在树上，
  节点 `j` 的父节点**不是** `j-1`（兄弟节点、无关节点会插在中间），寄存器里的 carry 会把
  兄弟分支的状态污染进当前节点。
- **short conv**：`npu_causal_conv1d_custom` 是**沿 batch 顺序滑一个窗口**，等价于"前一个
  token 的历史"。树上每个节点的窗口必须是**它自己的 root→node 祖先链**，否则非链上的节点
  会拿别的分支的激活做卷积，验证结果就是错的。
- **回滚**：线性链用 `num_accepted_tokens`（被接受 token 数 = 前缀长度）就能表达"从第几列
  续跑"。树上被接受的是**一条路径**而不是前缀，"路径长度 + 1"根本表达不了"最后接受的是哪
  个节点"，因此拒绝回滚无法用索引表达，只能拷贝状态或重算。

本提交要做的，就是让这两份状态**跟随树的形状**，并且让"丢弃被拒绝的兄弟"退化成
**纯粹的下标变化**（无状态拷贝、无重算）。

---

## 2. 术语与变量约定

记 `R` = 投机行数（spec batch rows）、`budget` = `tree_spec_config.budget`（每请求草稿节点数）、
**`slots = 1 + budget`**（每请求验证 token 数、也是每请求的状态槽数）、
`width` = 短卷积核宽度。

| 变量 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `slots` | `int` | 状态槽数，`1 + budget`。所有"状态宽度"都由它决定 |
| `chunk_width` | `int` | 分块验证时**单次算子调用**覆盖的节点列数 = `GDN_RECURRENT_MAX_VERIFY_TOKENS`（16）；见 §4.13 |
| `state_indices`（`spec_state_indices_tensor`） | `[R, slots]` int32 | **写**表：`state_indices[r, j]` = 节点 `j` 的递推状态要写进哪个 page；`j = 0` 是 root。来源是 `block_table_tensor[:, :slots]` |
| `num_accepted_tokens` | `[R]` int32 | **续跑游标**：上一步最后被接受节点所在的列 `+1`。线性链语义是"接受 token 数"，树上被复用成"接受节点列号 + 1" |
| `init_state_indices` / `spec_init_state_indices` | `[R, slots]` int32 | **读**表：第 `p` 个 token（token-major 打包后的全局序号）的初始状态从哪个 page 读 |
| `tree_parents` | `[R, budget]` int32 | `tree_parents[r, t]` = 节点 `t+1` 的父节点 id（`0` = root） |
| `tree_num_nodes` | `[R]` int32 | 非 root 节点数；**`0` 表示该行是从 prompt chunk 折进来的线性序列，不是一棵树** |
| `prev_path_node_ids` / `prev_num_sampled` | `[R, n]` / `[R]` | **上一步**的被接受路径与采样数，用于重建短卷积的 committed history |
| `spec_row_widths` | `[R]` | 每行本步真正验证多少 token（`1 ≤ w ≤ slots`；接近 `max_model_len` 时调度器会截断草稿，树搜索也可能候选不足） |
| `window_rows` | `[T, width]` int32 | 每个 token 的 `width` 个卷积 tap 行号，索引 `cat([committed_history, 本步激活])` |
| `conv_prefix_src` | `[R, width - 1]` int32 | 新的 committed history 取 page 的哪几列 |
| `_tree_resume_column` | `[R]` int32 | 模型状态里暂存的续跑列，由 model runner 在采样后写入 |
| `_tree_path` / `_tree_prev_path` | `[R, n]` / `[max_num_reqs, budget]` int32 | 本步被接受路径（宽度 = 采样器路径宽度）/ 上一步被接受路径（按**请求槽位**持久化，宽度按 `budget` 左对齐） |

---

## 3. 总体设计思路

### 3.1 核心思想：**一个树节点一份状态快照（snapshot）**

把"状态布局"从"链"改成"树"，规则只有两条：

1. **写**：节点 `j`（`j = 0` 是 root）把自己的递推状态写进 `state_indices[r, j]`。
2. **读**：节点 `j` 的初始状态从 `state_indices[r, parent(j)]` 读；root 的父节点是"上一步最后
   被接受的节点"，所以 root 从 `state_indices[r, num_accepted_tokens - 1]` 读。

于是：

- **拒绝兄弟 = 改下标**。下一步的 root 直接去读"被接受的那个节点"的 page，其余节点的 page
  留在那里没人再引用，不需要拷贝、也不需要重算。
- **不需要跨页拷贝状态**。传统做法是把接受路径的状态"搬"到线性槽位上，这需要 D2D 拷贝和同步；
  快照法把这件事彻底消掉。
- **续跑语义可以完全复用线性链的 `num_accepted_tokens`**：只要把它的值从"前缀长度"换成
  "最后被接受节点的列号 + 1"，下游（`block_table` 取列、kernel 取 page）一行都不用改。

为了让内核能按"每个 token 各自去读父节点"的方式跑起来，recurrent 算子新增了一个**可选输入
`init_state_indices`**；这是整条链路上唯一需要动 C++/AscendC 的地方。

### 3.2 状态布局的两处加宽

| 状态 | 线性链 | 树 | 加宽位置 |
| --- | --- | --- | --- |
| recurrent | `num_speculative_blocks = num_spec` | `budget` | `attn_utils._widen_mamba_spec_for_tree` |
| short conv | `(width - 1 + num_spec) × conv_dim` | `(width - 1 + slots) × conv_dim` | 同上 |

短卷积 page 为什么也要放宽：**它是"每请求一份"，不是"每节点一份"**。为了让下一步能从"被接受
的那个节点"重建 committed history，本步必须把**所有 `slots` 个节点的激活**都留在 page 里
（存在第 `width - 1 + node` 列），下一步再按 `conv_prefix_src` 挑出需要的那 `width - 1` 列，
原地覆盖到 page 的前 `width - 1` 列（即 committed history 的位置）。所以 page 需要
`width - 1 + slots` 列。

### 3.3 一步（step）的完整数据流

```text
① 投机器生成树：req_states.tree_parents / tree_num_nodes（parent_id < node_id）
        ↓  input_batch.tree_parents = req_states.tree_parents[idx_mapping]
② prepare_attn：AscendMambaHybridAttnMetadata 携带 tree_parents / tree_num_nodes /
        tree_prev_path / tree_prev_num_sampled（batch 顺序）
        ↓  get_extra_attn_kwargs → builder.build(..., tree_parents=..., tree_num_nodes=...,
                                                prev_path_node_ids=..., prev_num_sampled=...)
③ builder：
     · state_indices = block_table[:, :slots]（[R, slots]）
     · init_state_indices = compute_tree_init_state_indices(...)   ← 读表
     · window_rows / prefix_src = compute_tree_conv_rows / compute_tree_prefix_source
     · state_indices 与 init_state_indices 一起 _pack_tree_rows_to_tokens（token-major）
④ 前向：
     · 短卷积  tree_causal_conv1d(x, ..., window_rows, prefix_src)
     · 递推    npu_recurrent_gated_delta_rule(..., init_state_indices=...)
⑤ 采样：rejection_sampler 产出 path_node_ids / num_sampled
⑥ postprocess_sampled：
     · _stage_tree_state_resume_column → compute_tree_resume_column → 暂存 _tree_resume_column
     · super().postprocess_sampled → model_state.postprocess_state
         - num_sampled 被替换成 _tree_resume_column（写 num_accepted_tokens_gpu）
         - 把本步 path 记进 _tree_prev_path / _tree_prev_num_sampled（按请求槽位）
⑦ 下一步的 ② 再把 _tree_prev_path 读回 batch 顺序 —— 闭环
```

### 3.4 必须成立的不变量

1. `parent_id < node_id`（所有树的 builder 都保证）。因为父节点的状态是**同一步内**先算出来
   写进 page 的，只有父节点排在前面，节点 `j` 才能在读表时拿到已写好的状态。
2. 列 `j` 既是"节点 `j` 状态的写入位置"，也是"节点 `j` 状态的读取位置"。两者必须用同一张
   `[R, slots]` 表。
3. **单次算子调用的宽度** `≤ 16`（`GDN_RECURRENT_MAX_VERIFY_TOKENS`，等于内核的 `MAX_MTP`）。
   `slots` 本身可以更宽：超过 16 时按节点列切块（§4.13），块与块之间仍然只靠"父节点先算、
   父节点的 page 先写好"这一条顺序保证串起来。
4. `budget ≥ num_speculative_tokens`：采样器给出的路径只有 draft chain 那么宽，而缓冲区按
   `budget` 宽分配（左对齐，`-1` 补齐），窄于 `budget` 才装得下。
5. `tree_num_nodes == 0` 的行必须退化成**线性链**行为（父节点 = 前一个 token），这样从 prompt
   chunk 折进来的行才能继续用链语义。

---

## 4. 实现细节

### 4.1 公共入口：`state_rollback.py`

新增 `vllm_ascend/worker/v2/spec_decode/tree/state_rollback.py`，它是整个功能的"算法层"，
只做纯张量计算（无副作用、可单测），把结果交给 builder 和 `ops/gdn.py` 使用。

#### 4.1.1 配置项与注册表

```python
gdn_state_update: str = "snapshot"      # TreeSpecConfig（ascend_config.py）

@dataclass(frozen=True)
class GdnStateUpdateMethod:
    state_init: bool   # 递推状态是否跨树传递（靠 init_state_indices）
    conv: bool         # 短卷积是否端到端跟树（tree_causal_conv1d + 加宽 page）

GDN_STATE_UPDATE_METHODS = {"snapshot": GdnStateUpdateMethod(True, True, ...)}
DEFAULT_GDN_STATE_UPDATE = "snapshot"
```

设计意图：**配置只在 GDN 路径上被解析和校验**。

- `TreeSpecConfig` 里不做合法性检查（`gdn_state_update="magic"` 也能构造成功），所以不用
  GDN 的模型永远不会因为这个值报错；
- `selected_gdn_state_update_method()` 在真正需要时（`validate_tree_state_slots` 中，且
  `mamba_type is GDN_ATTN`）才查表并抛 `ValueError`。测试
  `test_config_defers_the_state_update_to_the_gdn_path` / `test_validate_ignores_the_state_update_for_other_backends`
  覆盖的就是这个语义。
- 如果将来加的新方法只实现了半套（`state_init=False` 或 `conv=False`），
  `validate_tree_state_slots` 会明确拒绝并解释后果，而不是静默跑出错误结果。

#### 4.1.2 开关与校验

```python
def tree_state_slot_count(vllm_config) -> int | None
```

返回 `None`（= 保持线性链布局）的三个条件，以及一个硬错误：

| 情况 | 行为 |
| --- | --- |
| `not vllm_config.use_v2_model_runner` | 返回 `None`（只有 v2 runner 承载树） |
| `not dflash_tree_spec_enabled(vllm_config)` | 返回 `None` |
| `tree_spec_config.budget is None` | 返回 `None` |
| `cache_config.mamba_cache_mode == "align"` | 抛 `NotImplementedError` |
| 其它 | 返回 `1 + budget` |

`mamba_cache_mode="align"` 被拒绝的原因写得很清楚：align 的 pre-copy kernel 会把 running
state page 搬走，并把短卷积窗口按 `num_accepted_tokens - 1` 平移；而树上"游标是节点 id、
窗口是 root→node 链"，**没有对应的平移语义**，硬做会静默破坏 conv history。recurrent 状态
本身不在乎（每个节点有自己的 page），坏的只是 conv。

```python
def validate_tree_state_slots(slots, mamba_type) -> None
```

- `mamba_type is not GDN_ATTN` → `NotImplementedError`（KDA / Mamba1/2 / short-conv / 纯线性
  注意力都还没接）；
- `slots < 1` → `ValueError`（至少验证 root 一个 token）；
- 选中方法的 `state_init` / `conv` 必须都为真，否则 `NotImplementedError`。

> **注意**：这里**不再**用内核的 `MAX_MTP = 16` 去卡 `slots`。16 限制的是"**单次算子调用**每请求
> 验证多少 token"（超过它内核直接 `return`、状态静默变脏），而比 16 宽的树由 §4.13 的分块验证
> 拆成多次调用，每次仍满足这个上限。所以 `slots` 只受状态显存（每请求 `1 + budget` 个 page）
> 约束，配置校验里也就只剩"方法是否实现"这一类检查。

```python
def tree_conv_width(mamba_type, conv_state_shape, num_spec) -> int | None
    width = min(conv_state_shape) - num_spec + 1
```

`conv_state` 形状是 `(width - 1 + num_spec) × conv_dim`，而 `conv_dim`（`2*key_dim + value_dim`）
比 state 长度大好几个数量级，所以用 2 维形状的**较小边**反推长度。非 GDN 或形状不符返回 `None`；
`width < 2` 抛错。

> **注意**：`num_spec` 参数必须是"这个 shape 实际承载的投机列数"。
>
> - 在 `attn_utils._widen_mamba_spec_for_tree` 里传的是配置的 `num_speculative_tokens`
>   （此时 spec 还没被加宽）；
> - 在 builder 里必须传 `max(self.num_spec, slots)`（此时 `kv_cache_spec` 已经被 `attn_utils`
>   加宽过，shape 里已经含 `slots` 列）。传错就会算出错的 `width`，卷积 tap 数直接错。

### 4.2 KV cache spec 加宽：`attn_utils._widen_mamba_spec_for_tree`

位置：`vllm_ascend/worker/v2/attn_utils.py`，在 `get_kv_cache_spec` 里对每个 `MambaSpec` 调用。

```python
slots = tree_state_slot_count(vllm_config)
if slots is None: return spec                       # 无树 → 原样返回
validate_tree_state_slots(slots, spec.mamba_type)

num_spec = vllm_config.speculative_config.num_speculative_tokens
if slots > spec.num_speculative_blocks + 1:
    spec = replace(spec, num_speculative_blocks=slots - 1)     # 每节点一份递推状态

width = tree_conv_width(spec.mamba_type, tuple(spec.shapes[0]), num_spec)
extra_columns = slots - num_spec
if width is None or extra_columns <= 0:
    return spec

conv_dim = max(conv_state_shape)
expected_columns = width - 1 + slots
conv_state_shape = (conv_dim, expected_columns) if is_conv_state_dim_first() else (expected_columns, conv_dim)
return replace(
    spec,
    shapes=(conv_state_shape, *spec.shapes[1:]),
    page_size_padded=spec.page_size_padded + extra_columns * conv_dim * get_dtype_size(spec.dtypes[0]),
)
```

要点：

- `MambaBase.get_kv_cache_spec` 从 `num_speculative_tokens` 推出 `num_speculative_blocks`
  （每请求保留几份递推状态 page），那是线性链的语义，所以树要覆盖它。
- **即使 `budget == num_spec`，conv page 也要 +1 列**：链上 page 存 `num_spec` 个节点的激活，
  树要存 `slots = num_spec + 1` 个。所以判断条件是 `extra_columns = slots - num_spec > 0`，
  与递推状态的条件（`slots > num_speculative_blocks + 1`）**不是同一个**。测试
  `test_widen_mamba_spec_widens_the_conv_page_at_chain_width` 专门锁死这一点。
- `shapes` / `dtypes` 是位置对应的，下标 0 是短卷积状态，所以只改 entry 0。
- `page_size_padded` 要跟着加（多出来的 `extra_columns * conv_dim * dtype_size` 字节），否则
  后续按 padded size 分配的 KV cache 会越界。
- 每个 GDN 层都会被同样加宽，从而保证 mamba 各层仍落在**同一个 uniform KV cache group**。

### 4.3 builder 侧的缓冲加宽：`AscendGDNAttentionMetadataBuilder._resize_spec_buffers_for_tree`

位置：`vllm_ascend/ops/gdn_attn_builder.py` 的 `__init__` 里，`super().__init__()` 之后立刻调用。

上游 `GDNAttentionMetadataBuilder.__init__` 用 `1 + num_speculative_tokens` 给所有 spec 尺寸的
buffer 定容量（`spec_state_indices_tensor`、`spec_conv1d` 相关、`num_accepted_tokens`、
`spec_query_start_loc` …），树要的是 `1 + budget`，所以这里**镜像上游的分配逻辑**重分配一遍：

```python
slots = tree_state_slot_count(self.vllm_config)
if slots is None: return
validate_tree_state_slots(slots, mamba_type)

self.spec_tree_slots = slots
self.spec_init_state_indices = torch.empty((self.decode_cudagraph_max_bs, slots), torch.int32, device)

state_columns = max(self.num_spec, slots)                     # 见 4.1.2 的注意点
self.spec_conv_width = _tree_conv_width(kv_cache_spec, mamba_type, state_columns)
if conv_width is not None:
    self.spec_conv_window_rows = torch.empty((max_bs * slots, conv_width), torch.int32, device)
    self.spec_conv_prefix_src  = torch.empty((max_bs, conv_width - 1), torch.int32, device)

if slots == self.num_spec + 1: return        # 宽度没变，state 表等 buffer 不需要重建

self.num_spec = slots - 1
self.decode_cudagraph_max_bs = max_num_seqs * slots          # 一个请求现在占 slots 个 token
if max_cudagraph_capture_size is not None:
    self.decode_cudagraph_max_bs = min(self.decode_cudagraph_max_bs, max_cudagraph_capture_size)

self.spec_state_indices_tensor   = empty((max_bs, slots))
self.non_spec_state_indices_tensor = empty((max_bs,))
self.spec_token_indx / non_spec_token_indx = empty((max_bs * slots,))
self.spec_query_start_loc / non_spec_query_start_loc = empty((max_bs + 1,))
self.num_accepted_tokens = empty((max_bs,))

if self.use_spec_decode and self.reorder_batch_threshold != 1:
    self.reorder_batch_threshold = slots
```

要点：

- **`self.num_spec = slots - 1` 是整个 builder 的关键一步**：后续 `block_table[:, :num_spec+1]`
  自动变成 `slots` 列，无需再改业务代码。
- `decode_cudagraph_max_bs` 按 `max_num_seqs * slots` 放大（保证最坏情况下所有请求同一步都在
  验证树），再被 `max_cudagraph_capture_size` 夹住。
- 新增的五个字段的含义：
    - `spec_tree_slots`：`None` = 线性链（默认），非 `None` = 树模式，**所有树分支的入口判断**；
    - `spec_init_state_indices`：`[max_bs, slots]` int32 稳定 buffer（全图重放要按地址读）；
    - `spec_conv_width`：短卷积核宽度；
    - `spec_conv_window_rows`：`[max_bs * slots, width]` 稳定 buffer；
    - `spec_conv_prefix_src`：`[max_bs, width - 1]` 稳定 buffer。
- `reorder_batch_threshold = slots`：树的验证批恒定是 root + 全部节点，剔除/重排阈值必须按
  整棵树算，否则 batch 重排会打乱 `[R, slots]` 的行内布局。

### 4.4 读表（per-token 初始状态）：`compute_tree_init_state_indices`

```python
def compute_tree_init_state_indices(state_indices, num_accepted_tokens, tree_parents, tree_num_nodes):
    rows, slots = state_indices.shape
    if slots - 1 > tree_parents.shape[1]:
        raise ValueError(...)                       # budget 与状态槽数不匹配

    parents = tree_parents[:rows, : slots - 1].long()
    chain_parents = arange(1, slots) - 1            # 节点 t 续 t-1
    parents = where(tree_num_nodes[:rows, None] > 0, parents, chain_parents)

    accepted_slot = (num_accepted_tokens[:rows].long() - 1).clamp(min=0).unsqueeze(1)
    root_init = gather(state_indices, 1, accepted_slot)   # root 读"上一个被接受节点"的列
    node_init = gather(state_indices, 1, parents)         # 节点 j 读父节点的列
    return cat((root_init, node_init), dim=1).int()
```

- 输出 `[R, slots]`，**列 0 = root、列 `j` = 节点 `j` 的初始 page**，与 `state_indices` 的行列顺序
  完全对齐。
- `tree_num_nodes == 0` 的行（prompt chunk 折入）用链式父节点，于是
  `init[p] == state_indices[p-1]`：内核里那个"父 page == 上一个 token 的 page"的判定会命中，
  直接走寄存器 carry，**与线性链行为逐位一致**。
- `accepted_slot = num_accepted - 1`：`num_accepted` 就是上一步写进 `num_accepted_tokens_gpu`
  的续跑列（见 4.5），所以"续跑列 + 1"这个约定在这里被翻译成"取列 `num_accepted - 1`"。

### 4.5 token-major 打包：`_pack_tree_rows_to_tokens`

recurrent 算子按**全局 token 偏移**读 `ssm_state_indices` 与 `init_state_indices`
（`ssmStateIndicesGm_.GetValue(seq_i)`），而树的两张表是 `[row, node]` 布局。绝大多数情况下
每行都验证完整的 `1 + budget` 个 token（此时 `[R, slots]` 展平后天然就是 token-major），但：

- 接近 `max_model_len` 时调度器会截断草稿；
- 树搜索可能候选不足（`num_nodes < budget`，甚至整行候选为空）；
- 全图 padding 行长度为 0。

所以必须把每行真正验证的 token **背靠背**排开：

```python
widths = row_widths.to(device)                      # [R]，每行验证的 token 数
starts = cumsum(widths) - widths                    # 本行第一个 token 的全局偏移
ends   = starts + widths
positions = arange(rows * slots)                    # 固定容量，代价与 widths 无关
token_rows = (positions[:, None] >= ends[None, :]).sum(dim=1).clamp(max=rows - 1)
token_cols = (positions - starts[token_rows]).clamp(max=slots - 1)
packed = table.reshape(-1).index_select(0, token_rows * slots + token_cols)
packed = packed.masked_fill(positions >= ends[-1], NULL_BLOCK_ID)   # NULL_BLOCK_ID == 0
return packed.view(rows, slots)
```

要点：

- 结果**保持 `[rows, slots]` 形状**（算子收到的是 `reshape(-1)`），只是内容被重排；
- 打包后**全局第 `p` 个 token 取"它所属行的第 `p - start(row)` 个节点"**；
- 产出零长度行会被 0（`NULL_BLOCK_ID`）填充，内核永远不会访问（它们 `seqLen == 0` 被 `continue`）；
- 打包同时服务两个消费者：递推算子（按 token 偏移读）**和**短卷积
  （`cache_indices.reshape(-1)[starts[:rows]]` 恰好取到每行 token 起始处的**第 0 列**，也就是该
  请求的 running state page）。这也是为什么 `build()` 里把 `spec_state_indices_tensor` 一并打包。
- 触发打包的条件是 `spec_init_state_indices is not None and self.spec_conv_window_rows is not None`。
  GDN 路径下两者必然都非 `None`，条件只是防御。

### 4.6 续跑列：`compute_tree_resume_column` 与 `postprocess_state`

```python
def compute_tree_resume_column(path_node_ids, num_sampled):
    sampled = num_sampled.long()
    last_index = (sampled - 2).clamp(min=0).unsqueeze(1)     # 最后一个被接受节点
    last_node = path_node_ids.long().gather(1, last_index).squeeze(1)
    last_node = where(sampled >= 2, last_node, zeros_like(last_node))
    return (last_node + 1).int()
```

- `num_sampled == 1`（没接受任何 draft，例如 chunked prefill）→ 钳到 0 → 返回 `1` →
  取列 0（root page = 已提交前缀状态），**与线性链的中性值 `num_accepted_tokens = 1` 完全一致**；
- `num_sampled >= 2` → 返回"最后被接受节点 id + 1"，正好是"续跑列 + 1"的约定；
- 返回值恒 `>= 1`，所以下游 `max(resume, 1)` 是空操作。

`model_runner._stage_tree_state_resume_column`：

```python
model_state = self.model_state
if not hasattr(model_state, "stage_tree_resume_column"):
    return
model_state.stage_tree_resume_column(
    compute_tree_resume_column(path_node_ids, num_sampled),
    path_node_ids,          # 本步接受路径，供短卷积重建 history
    num_sampled,            # 伴随 num_sampled，用于判断"本行有没有新路径"
)
```

它在 `postprocess_sampled` 的**最前面**调用（先于 `super().postprocess_sampled`，后者内部会调用
`model_state.postprocess_state`）。顺序保证：暂存值在提交时一定已经就位。

`AscendMambaHybridModelState.postprocess_state`（`worker/v2/model_states/mamba_hybrid.py`）：

```python
def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
    if self._tree_resume_column is not None:
        num_sampled = self._tree_resume_column        # ① 只换值，下游机制全复用
        self._tree_resume_column = None

    path = self._tree_path
    if path is not None:
        prev_path, prev_num_sampled = self._tree_buffers(idx_mapping.device)
        prev_num_sampled[idx_mapping] = 0             # ② 先失效，防陈旧路径把 history 拉歪
        if self._tree_num_sampled is not None:
            assert path.shape[1] <= prev_path.shape[1]
            prev_path[idx_mapping, : path.shape[1]] = path.int()
            prev_num_sampled[idx_mapping] = self._tree_num_sampled.int()
        self._tree_path = None
        self._tree_num_sampled = None

    super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
```

为什么"只换 `num_sampled`"就够了：换值之后走的还是上游那套 `_scatter_num_accepted_kernel`,
落点、拷贝偏置、align postprocess 全都不变，**只有语义从"前缀长度"变成"节点列 + 1"**。

`_tree_buffers` 的两个细节：

```python
slots = tree_state_slot_count(self.vllm_config)     # assert slots is not None
path_width = slots - 1                              # = budget
num_reqs = self.num_accepted_tokens_gpu.shape[0]    # = max_num_reqs，按请求槽位持久化
path = self._tree_prev_path
if path is None or path.shape != (num_reqs, path_width) or path.device != device:
    path = zeros((num_reqs, path_width), int32, device)
    self._tree_prev_num_sampled = zeros((num_reqs,), int32, device)
    self._tree_prev_path = path
return path, self._tree_prev_num_sampled
```

1. **按 `budget` 而不是按采样器路径宽度分配**。采样器的 `path_node_ids` 只有
   `num_speculative_tokens` 宽，如果缓冲区按那个宽度分配，`path.shape != (num_reqs, path_width)`
   会在**每一步**都命中，于是缓冲区每步被重新创建并清零 —— 短卷积的 re-anchor 会**静默失效**。
   测试 `test_tree_path_buffers_keep_the_tree_width` 就是钉这个坑的。
2. 用**请求槽位（`idx_mapping`）**而不是 batch 下标寻址，因为 `postprocess_state`（写）与
   `prepare_attn`（下一步读）处在不同的 batch 组成里。

### 4.7 把路径读回 batch 顺序：`prepare_attn` + `AscendMambaHybridAttnMetadata`

`worker/v2/model_states/mamba_hybrid.py`：

- 新增 dataclass `AscendMambaHybridAttnMetadata(MambaHybridAttnMetadata)`，多带四个字段
  `tree_parents` / `tree_num_nodes` / `tree_prev_path` / `tree_prev_num_sampled`；
- `get_extra_attn_kwargs` 重写：

```python
kwargs = super().get_extra_attn_kwargs(builder, num_reqs)
if not kwargs or self.tree_parents is None:
    return kwargs
if not isinstance(builder, AscendGDNAttentionMetadataBuilder):
    return kwargs                     # Mamba2 / short-conv 没有树状态回滚
kwargs["tree_parents"] = self.tree_parents[:num_reqs]
kwargs["tree_num_nodes"] = self.tree_num_nodes[:num_reqs]
if self.tree_prev_path is not None and self.tree_prev_num_sampled is not None:
    kwargs["prev_path_node_ids"] = self.tree_prev_path[:num_reqs]
    kwargs["prev_num_sampled"] = self.tree_prev_num_sampled[:num_reqs]
return kwargs
```

`if not kwargs` 这个早退很关键：上游 `get_extra_attn_kwargs` 对不在
`(Mamba2AttentionMetadataBuilder, GDNAttentionMetadataBuilder)` 里的 builder 返回 `{}`，
所以用 `kwargs` 真假就等价于"这是不是一个 mamba builder"，不需要再复制一遍类型判断。

`prepare_attn` 里构造 batch 顺序视图：

```python
tree_slots = tree_state_slot_count(self.vllm_config)
if tree_slots is not None:
    path_buffer, num_sampled_buffer = self._tree_buffers(self.num_accepted_tokens_gpu.device)
    tree_prev_path = zeros((num_reqs, tree_slots - 1), int32)       # 全 0 起手
    tree_prev_path[:input_batch.num_reqs] = path_buffer[input_batch.idx_mapping]
    tree_prev_num_sampled = zeros((num_reqs,), int32)
    tree_prev_num_sampled[:input_batch.num_reqs] = num_sampled_buffer[input_batch.idx_mapping]
```

- 只有真实请求填内容，padding 行保持 `0`；`prev_num_sampled == 0` 在
  `compute_tree_prefix_source` 里被翻译成"恒等映射"，也就是**保持上一步算子留下的 committed
  history 不动**。

同一文件里还有一处全图相关的修正：

```python
expected_query_len = (
    tree_state_slot_count(self.vllm_config) or self.vllm_config.num_speculative_tokens + 1
)
```

全图 capture 会把每个 padding 请求表示成一个投机请求，padding 行的 query 长度必须是
`1 + budget`；用配置里的 `num_speculative_tokens + 1` 去比对会不匹配，从而错误地把全图重放
降级成非全图路径。

### 4.8 短卷积的树化

短卷积比递推状态多一层麻烦：page 是"每请求一份"的滑动窗口，而树上每个节点需要**自己的
root→node 窗口**。方案：

1. page 预留 `width - 1 + slots` 列：前 `width - 1` 列是 committed history（与线性链约定相同），
   后面 `slots` 列存**本步所有节点的激活**（节点 `n` → 第 `width - 1 + n` 列）；
2. 本步开始时，按"上一步被接受的节点"把 committed history **原地重锚**；
3. 每个节点的卷积 tap 由 `window_rows` 显式给出，从
   `cat([committed_history, 本步激活])` 里 gather。

#### 4.8.1 `tree_node_parents_depths`

```python
parents = full((rows, slots), -1); parents[:, 1:] = tree_parents[:, :slots-1]
parents = where(tree_num_nodes[:, None] > 0, parents, arange(slots) - 1)   # 折入行 → 链
depths = zeros((rows, slots))
for node in range(1, slots):
    depths[:, node] = depths.gather(1, parents[:, node].clamp(min=0).unsqueeze(1)).squeeze(1) + 1
```

`parents[:, 0] = -1`（root 没有父节点）。深度用一次**前向扫描**就能算完，前提正是
`parent_id < node_id` 这个不变量。

#### 4.8.2 `_tree_conv_tap_rows`：每个 target 的 `width` 个 tap

输出行号索引 `cat([committed, activations])`：committed 是**上一步的** `width - 1` 段激活，
按请求排在最前面；激活是本步的，排在后面。

```python
for tap in range(width - 1):
    ancestor  = node_row_table[request_ids, chain[:, width - 2 - tap].clamp(min=0)]
    committed = history_base[request_ids] + (depths + tap).clamp(max=width - 2)
    taps.append(where(depths >= (width - 1 - tap), ancestor, committed))
taps.append(own_rows)
```

- tap `width-2` 是父节点，`width-3` 是祖父，……（`chain[:, k]` = 第 `k+1` 级祖先）；
- 一个 tap 落在 root 之前（深度不够）时，就回落到 committed history。**这条规则让浅节点的
  窗口与线性链窗口逐位相同**，因此树不会改变 deep chain 的行为；
- `taps.append(own_rows)`：最后一个 tap 是自身激活。

#### 4.8.3 `compute_tree_conv_rows`

```python
compute_tree_conv_rows(tree_parents, tree_num_nodes, width, node_row_offset, row_widths=None)
```

- `node_row_offset = rows * (width - 1)`，即 committed 段之后本步激活的起始行；
- 先按 4.5 同样的规则算 token-major 的 `(token_rows, token_cols)`，再把每个"打包后的 token"当成
  一个 target，交给 `_tree_conv_tap_rows` 求窗口；
- 输出 `[rows * slots, width]` int32，**容量固定**（代价与 `row_widths` 无关），只有前
  `row_widths.sum()` 行有意义 —— 层里会按真实 token 数切片；
- `row_widths` 是 CPU 张量，先做范围校验再搬到 device：`bool(...)` 在 device 上会同步流。
  校验 `1 <= w <= slots`。

#### 4.8.4 `compute_tree_prefix_source`

上一步被接受的节点决定了下一步的 committed history：

```python
last_depth = (num_sampled - 1).clamp(min=-1).unsqueeze(1)      # 最后被接受节点的深度
depth = last_depth - (width - 2) + arange(width - 1)
node_ids = path_node_ids.gather(1, (depth - 1).clamp(min=0))
return where(depth >= 1, width - 1 + node_ids, width - 1 + depth)
```

深度到 page 列的映射（`width - 1` 就是节点 0 所在的列）：

| 深度 `d` | 含义 | page 列 |
| --- | --- | --- |
| `d >= 1` | 该深度的被接受节点 | `width - 1 + node_id` |
| `d == 0` | root 自身（本步第一个 token，**不在 committed 列里**） | `width - 1` |
| `d < 0` | root 之前第 `-d` 个激活，已在 committed 列中 | `width - 1 + d` |

- `num_sampled == 0`（采样器跳过的行，如折入的 prompt chunk）→ `last_depth = -1` → 所有
  `depth < 0` → **恒等映射**，committed history 原地不动；
- 不变量：`prefix_src[:, -1] - (width - 2) == compute_tree_resume_column(path, num_sampled)`，
  即**短卷积的 history 末端与递推状态的续跑列指向同一个节点**。测试
  `test_tree_prefix_source_follows_the_accepted_node` 断言了这条一致性。

#### 4.8.5 `ops/gdn.tree_causal_conv1d`

位置：`vllm_ascend/ops/gdn.py`。替代 `npu_causal_conv1d_custom` 的树分支。

```python
pages  = cache_indices.reshape(-1)[starts[:rows]].long().clamp(min=0)  # 每行 token 起点的第 0 列
valid  = diff(starts) > 0
page_ids = pages[valid]

# ① 原地重锚 committed history
state = conv_state[page_ids]
columns = prefix_src[valid].long()
if dim_first: conv_state[page_ids, :, :width-1] = state.gather(2, columns.expand(...))
else:         conv_state[page_ids, :width-1, :] = state.gather(1, columns.expand(...))

# ② gather tap 并卷积
state   = conv_state[pages]
history = state[:, :, :width-1].transpose(1,2) if dim_first else state[:, :width-1, :]
source  = cat((history.reshape(-1, conv_dim), x), dim=0)
taps    = source.index_select(0, window_rows.reshape(-1).long()).view(-1, width, conv_dim)
out     = (taps * weight.transpose(0,1)).sum(dim=1) + bias(可选) → silu(可选)

# ③ 把本步所有节点激活写回 page 的节点列
positions   = arange(x.shape[0])
row_of_token = (positions[:, None] >= starts[1:][None, :]).sum(1).clamp(max=rows-1)
token_columns = width - 1 + positions - starts[:rows][row_of_token]
conv_state[pages[row_of_token], token_columns, :] 或 [..., :, token_columns] = x
return out
```

实现要点：

- **顺序不能颠倒**：必须先把新的 committed history 覆盖到 page 的 `[0, width-1)` 列，再
  gather tap；否则回落到 committed 列的 tap 会读到旧 history。而 page 的**节点列此时还是上一步
  的激活**，`prefix_src` 正好要从那里取数 —— 所以重锚必须在写本步激活**之前**。
- `conv_state[page_ids]` 是**高级索引 → 拷贝**，`gather` 读到的是更新前的 page；随后
  `state = conv_state[pages]` 重新取一次，才拿到更新后的 history。
- 权重布局：算子侧期望 `[width, conv_dim]`，而 `self.conv1d.weight` 是 `[conv_dim, width]`
  （oldest tap first），所以调用点是 `tree_causal_conv1d(..., conv_weights, ...)` —— 传**未转置**的
  `conv_weights`，内部做 `weight.transpose(0, 1)`。上游算子调用点传的是 `conv_weights_T`，两者
  注意区分。
- `valid = diff(starts) > 0`：全图 padding 行 query 长度为 0，它们**不拥有 state page**，所以
  读一个无害的 page（`clamp` 到合法下标）并在第 ② 步跳过它们的 state 更新。只有真实行贡献
  token，所以 `x` 的行数就是 `valid` 行的 token 数之和。
- `row_of_token` 用"有多少行的 end ≤ p"求和得到：零长度行因为 `start == end` 会被自然跳过，
  正好落在正确的行上。
- 调用点（`AscendGatedDeltaNetAttention`）：`conv_window_rows is not None` 走 `tree_causal_conv1d`，
  否则保持原来的 `npu_causal_conv1d_custom`（`None` = 线性链，默认路径零开销）。窗口表按
  `window_rows[: mixed_qkv_spec.shape[0]]` 切片，padding 行的窗口永远不会被 gather 到。

### 4.9 递推算子的改造（`csrc/attention/recurrent_gated_delta_rule`）

这是唯一需要动 C++ 的改动，改动本身很薄：**加一个可选输入 `init_state_indices`**。

#### 4.9.1 接口层

| 文件 | 改动 |
| --- | --- |
| `op_host/recurrent_gated_delta_rule_def.cpp` | 新增 `Input("init_state_indices").ParamType(OPTIONAL)`，int32 / ND |
| `op_host/recurrent_gated_delta_rule.h` / `op_host/op_api/*` | l0 与 aclnn 签名加参数，`params.init_state_indices` 透传，`Contiguous` 化 |
| `op_host/recurrent_gated_delta_rule_tiling.cpp` | `INIT_STATE_INDEX = 10`；`AnalyzeDtype` 要求 int32；`AnalyzeFormat` 拒绝 NZ；`GetOptionalInput` 写 `hasInitStateIndices`，并校验**元素个数与 `ssm_state_indices` 相同**（因为两者都按 token 偏移寻址） |
| `op_kernel/recurrent_gated_delta_rule_tiling_data.h` | tiling 结构体加 `uint32_t hasInitStateIndices` |
| `op_kernel/recurrent_gated_delta_rule.cpp` | kernel 入口多一个 `GM_ADDR initStateIndices`，塞进 `RGDRInitParams` |
| `csrc/torch_binding.cpp` / `torch_binding_meta.cpp` / `recurrent_gated_delta_rule_torch_adpt.h` | torch 侧签名尾部追加 `Tensor? init_state_indices=None`（**可选且带默认值，老调用方不受影响**） |

> **注意**：`init_state_indices` 是**尾部追加**的可选参数，所以线性链路径不用改一行调用代码，
> 二进制/图缓存兼容性也最好。

#### 4.9.2 kernel 计算逻辑（`op_kernel/recurrent_gated_delta_rule.h`，arch35 版本同构）

三处改动：

**(a) 行首的初始状态来源**

```cpp
if (copyFlag == 1) {
    if (hasInitStateIndices_) {
        int32_t initStateIdx = initStateIndicesGm_.GetValue(seq0);   // root 的初始 page
        if (initStateIdx < 0) { return; }
        stateOffset = static_cast<uint64_t>(initStateIdx);
    } else {
        // 原线性链：stateTokenIdx = seq0 + numAcceptedTokens[batch_i] - 1
        ...
        stateOffset = ssmStateIndicesGm_.GetValue(stateTokenIdx);
    }
    CopyInGamaBeta(seq0, seq1);
}
```

**(b) token 循环里的"父 page 不同就重载"**

```cpp
for (seq_i = seq0; seq_i < seq1; ++seq_i) {
    ...
    if (hasInitStateIndices_ && seq_i > seq0) {
        int32_t parentStateIdx = initStateIndicesGm_.GetValue(seq_i);
        if (parentStateIdx < 0) { return; }
        if (parentStateIdx != ssmStateIndicesGm_.GetValue(seq_i - 1)) {
            if (hasPendingState) { CopyOutState(pendingStateOffset, curSingleV); hasPendingState = false; }
            SyncStateWriteVisible();
            LoadStateTile(parentStateIdx, head_i, v_i, curSingleV);
        }
    }
    Compute(curSingleV, curQKOffset, curVOffset);
    ...
}
```

这是整段 kernel 逻辑的核心，三层含义：

1. **寄存器 carry 复用**：`stateInUb` 里此刻正是"上一个 token 写完的状态"。如果 `init[seq_i]`
   等于"上一个 token 自己的 page"（`ssmStateIndices[seq_i-1]`），说明节点 `i` 就是延续 `i-1`，
   直接接着算，**不产生任何 MTE2** —— 线性链与"树上的链式段"都走这条零开销路径；
2. **分支才重载**：只有一个 token 的父 page 不等于前一个 token 自己的 page 时（真正的分叉、
   或 root 之后回到主的兄弟分支顺序）才需要重新从 page 读状态；
3. **写后读必须同步**：`initStateGm_` 与 `finalStateGm_` 是**同一片 page**。重载前必须先把挂起
   的状态写（`pendingState`，当 `stateOutBufferNum_ == 1` 时会延迟一个 token）刷出去，并用
   `HardEvent::MTE3_MTE2` 事件让写可见，否则 MTE2 读到的可能是旧值。

**(c) 预取流水线的调整**

```cpp
// 循环前
LoadPrefetchedState(curSingleV);
if (!hasInitStateIndices_) {
    PrefetchNextStateTile(stateOffset, head_i, v_i, nextVOffset, nextSingleV);   // 原逻辑：与 token 循环重叠
}
...
// v_i 循环末尾
if (hasInitStateIndices_) {
    PrefetchNextStateTile(stateOffset, head_i, v_i, nextVOffset, nextSingleV);   // 树：改到循环后预取
}
```

原因：`stateInQueue_` 深度为 1，树路径在里面会**多次 `LoadStateTile`**（各自
`PrefetchState` + `LoadPrefetchedState`）。如果还在循环前预取下一块 v，队列里就会留下一个
未被消费的 tensor，和循环内的重载互相踩。所以树模式把预取挪到 token 循环**之后**（此时队列
是空的），顺序变成"重载 → 计算 → 预取下一块 v"。

辅助函数 `PrefetchNextStateTile` / `SyncStateWriteVisible` / `LoadStateTile` 是为了让普通版与
arch35 版共用同一段逻辑、保持两处代码同构。

### 4.10 树布局进入 builder：`build()` 与 `build_for_cudagraph_capture`

`AscendGDNAttentionMetadataBuilder.build()` 新增四个关键字参数
（`tree_parents` / `tree_num_nodes` / `prev_path_node_ids` / `prev_num_sampled`），并在
`spec_sequence_masks is not None` 分支里：

```python
spec_row_widths = query_lens_cpu[spec_sequence_masks_cpu]          # 每行验证多少 token
spec_init_state_indices = self._build_tree_init_state_indices(...)  # 含 token-major 打包
spec_conv_window_rows, spec_conv_prefix_src = self._build_tree_conv_rows(...)
if spec_init_state_indices is not None and self.spec_conv_window_rows is not None:
    spec_state_indices_tensor = _pack_tree_rows_to_tokens(spec_state_indices_tensor, spec_row_widths, ...)
    self._count_tree_shape_stats(...)
```

然后 `_attach_spec_decode_metadata` 把三张表塞进新字段
（`GDNSpecDecodeMetadata.init_state_indices` / `conv_window_rows` / `conv_prefix_src`，**默认
`None` = 线性链**），全图分支里再把它们拷进稳定 buffer：

```python
if spec_init_state_indices is not None:
    self.spec_init_state_indices[:num_spec_decodes].copy_(spec_init_state_indices, non_blocking=True)
    spec_init_state_indices = self.spec_init_state_indices[:spec_batch_size]
    spec_init_state_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)     # padding 行

if spec_conv_window_rows is not None:
    self.spec_conv_window_rows[:n].copy_(spec_conv_window_rows, non_blocking=True)
    self.spec_conv_prefix_src[:m].copy_(spec_conv_prefix_src, non_blocking=True)
    spec_conv_window_rows = self.spec_conv_window_rows[: spec_batch_size * self.spec_tree_slots]
    spec_conv_prefix_src = self.spec_conv_prefix_src[:spec_batch_size]
```

`_reset_spec_decode_graph_inputs` 也要把 `spec_init_state_indices` 填成 `NULL_BLOCK_ID`，这样
"全图在跑但没有投机请求"的那次重放不会拿上一次请求的 page 去改别人的状态。

新增 `build_for_cudagraph_capture`：

```python
def build_for_cudagraph_capture(self, common_attn_metadata):
    if self.spec_tree_slots is None:
        return super().build_for_cudagraph_capture(common_attn_metadata)
    # tree_num_nodes == 0 → 链式父节点；prev_num_sampled == 0 → 恒等 prefix_src
    tree_parents = zeros((m.num_reqs, slots - 1), int32)
    tree_num_nodes = zeros((m.num_reqs,), int32)
    prev_path_node_ids = zeros((m.num_reqs, slots - 1), int32)
    prev_num_sampled = zeros((m.num_reqs,), int32)
    num_accepted_tokens = diff(m.query_start_loc)
    num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
    return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu,
                      tree_parents=tree_parents, tree_num_nodes=tree_num_nodes,
                      prev_path_node_ids=prev_path_node_ids, prev_num_sampled=prev_num_sampled)
```

**为什么必须重写**：全图重放的是 capture 时录下的 kernel。如果 capture 时 `init_state_indices`
是 `None`，算子会以 `hasInitStateIndices = 0` 分支被录进去，重放时树就永远走寄存器 carry，
**状态静默错位**。所以 capture 时必须把这几张表**填上并传下去**，让算子录成"带
`init_state_indices`"的版本；capture 用的具体 page id 无所谓（capture 的输出会被丢弃），只要
形状对、布局是链式（`tree_num_nodes = 0`）、history 恒等（`prev_num_sampled = 0`）即可。

### 4.11 树 KV compact 跳过 Mamba：`kv_project.py`

```python
for group_id, group in enumerate(self.runner.kv_cache_config.kv_cache_groups):
    if isinstance(group.kv_cache_spec, MambaSpec):
        # Mamba 状态 page 按"请求状态槽位"寻址，不是按 token 位置，
        # 所以"把接受路径搬成线性位置"这个动作必须跳过它们。
        continue
    ...
```

`TreeKvCompact` 的职责是把被接受路径上的 KV 搬到线性位置（让下一步看起来像普通连续解码）。
但 Mamba 的状态 page 由状态槽位寻址，而树路径本身**已经**通过快照 + 续跑列完成了"回滚"，
二者语义冲突：如果再按 token 位置搬一遍，就会把快照布局打乱。所以这里直接跳过整个 Mamba
group（`_bind_groups` 解决；测试 `test_bind_groups_skips_mamba_state_groups`、
`test_compact_moves_accepted_path_and_keeps_mamba_state` 都验证 state 未被触碰）。

### 4.12 可观测性：shape counters

`worker/v2/spec_decode/tree/timer.py` 新增：

```python
def tree_stat(name: str, value: float = 1.0) -> None:
    if not _ENABLED:
        return
    _STAT_COUNT[name] += 1
    _STAT_SUM[name] += float(value)
```

退出报告里按 `n / sum / mean` 打印。`_count_tree_shape_stats` 在 `enable_timer` 打开时记录：

| 计数器 | 含义 / 为什么需要 |
| --- | --- |
| `tree.rows` / `tree.width` | 本步投机行数 / 总 token 数 |
| `tree.short_rows` | 验证宽度不足 `slots` 的行数（调度器截断或候选不足） |
| `tree.branch_rows` | `parents != chain` 且 `num_nodes > 0` 的行数 —— **这一步到底有没有在验证树**（全塌成链时树状态路径根本没被触发） |
| `tree.nodes` / `tree.drafts` | 树给出的节点数 / 调度器实际下发的草稿数 |
| `tree.unverified_extra_nodes` | `nodes - drafts`：**没有对应验证 token 的额外节点**。树搜索仍然会走这些节点，于是会"接受目标模型从未检查过的 token" —— 这是接受率曲线看不出来的正确性隐患 |
| `tree.init_ne_own_nodes` | `init_state_indices[p] != state_indices[p]` 的 token 数，即**真正触发 kernel 重载路径**的节点数 |

> **注意**：这些计数会做 `.item()` / `bool()`，是**设备同步**操作，所以调用方必须把同步放在
> `tree_timer_enabled()` 之后（`tree_stat` 里的提前 return 救不回来已经发生的同步）。

### 4.13 超宽树的分块验证（`slots > 16`）

内核的 `MAX_MTP = 16` 只约束**单次调用**：`Process` 里 `seqLen > MAX_MTP` 就 `return`，该行
（以及这一次调用里剩下的所有行）状态都不再更新。所以 `slots > 16` 不是"跑不了"，而是
"一次跑不完"：把每行的 `slots` 个节点列切成 `MAX_MTP` 宽的块，逐块调用算子即可。父子顺序
保证（`parent_id < node_id`，§3.4 第 1 条）让这条链成立 —— 父节点所在的那一块一定先执行，
它写进 page 的递推状态在后面的块里作为子节点的 `init_state_indices` 被读回（§4.9.2 的
"父 page 不同就重载"）。

`state_rollback.compute_tree_verify_plan` 负责切块，产物是：

```python
@dataclass(frozen=True)
class TreeVerifyChunk:            # 一次算子调用的全部输入
    token_indices: Tensor         # [rows * chunk_width] int64：gather 进本步的紧凑 token 维
    state_indices: Tensor         # [rows * chunk_width] int32：本块拥有的节点的写 page；块外 -1
    init_state_indices: Tensor    # 同上，读 page（父节点的 page，由更早的块写好）
    out_indices: Tensor           # [rows * chunk_width] int64：算子输出落到 out 的哪一行
    cu_seqlens: Tensor            # [rows + 1] int32：每行本块验证多少 token
    width: int                    # 本块覆盖的节点列数 = 16

@dataclass(frozen=True)
class TreeVerifyPlan:
    rows: int; slots: int; width: int; chunks: tuple[TreeVerifyChunk, ...]
```

切块规则（共 `ceil(slots / chunk_width)` 次调用，第 `c` 块管节点列 `[c*16, (c+1)*16)`）：

- 第 `c` 块里第 `r` 行拥有的**节点列**恰好是 `[c*16, min(width_r, (c+1)*16))`（即本行自己的前缀），
  摆在 `rows * 16` 网格的 `[r*16, r*16 + len)` 单元格里，其余单元格是 padding。
  `cu_seqlens` 用**长度**表达每行的前缀（第 0 项是内核约定的占位 0，第 `row+1` 项是该行长度，
  内核自己累加成全局 token 偏移），`seqLen <= 0` 的行内核直接 `continue`，所以 padding
  永远不会被读。
- 因此块内的形状只由 `rows` 与 `chunk_width` 决定，与"每行实际验证多少 token"无关 —— 这正是
  全图 capture 需要的静态性（对比 §4.5：**整步**的紧凑 token 维仍然由每行宽度决定，
  分块的网格只存在于单次调用的输入里）。
- padding 单元格的 `state/init` 一律写 `-1`（内核见到负数直接 `return` 不更新状态），
  `out_indices` 指向 `num_actual_tokens` 之后的 scratch 行。这两条都是防呆：padding 本来
  就落在每行 `cu_seqlens` 前缀之外、不会被读；万一将来切块规则改坏，结果是**响亮地什么都不
  更新**，而不是悄悄把上一行的状态串到下一行。
- `token_indices` 是**本步紧凑 token 维**（`_pack_tree_rows_to_tokens` 之后的顺序）的下标，
  用 `index_select` 把 q/k/v/g/beta 取成 `rows * 16` 的块输入 —— 与 §4.5 的打包共用一套坐标，
  所以 `state_indices[p]`、`init_state_indices[p]` 直接就是第 `p` 个 token 的 page。
- `out_indices` 把算子输出的 `rows * 16` 行按 token 序号散回输出 buffer，最后 `out[:tokens]`
  与单次调用的输出**同序同形状**。

`ops/gdn._chunked_tree_verify` 就是这段循环（每个 GDN 层每步执行 `chunks` 次算子调用）；
`spec_decode_metadata.verify_plan is None` 时走的仍是原来的单次调用，**`slots ≤ 16` 的路径
逐字节不变**。

builder 侧（`gdn_attn_builder`）：

- `_allocate_tree_verify_plan_buffers`（在 `_resize_spec_buffers_for_tree` 末尾调用，含
  `slots == num_spec + 1` 的提前返回分支）按图的**行容量** `decode_cudagraph_max_bs` 一次性
  分配每块的稳定张量（`TreeVerifyPlanBuffers`）；`slots ≤ 16` 或 eager 模式不分配任何东西；
- `_build_tree_verify_plan` 每步由 `actual_seq_lengths[1:]`（每行验证 token 数，**长度**语义）、
  `spec_state_indices_tensor`、`init_state_indices` 生成 plan；全图模式下
  `plan.copy_into(buffers)` 把值原地写进稳定张量，再把**视图**交给算子（capture 录下的就是
  这些地址，与 §4.10 的 `spec_init_state_indices` 等是同一套"稳定 buffer + 每步原地刷新"模式）；
- 只有"树 packing 跑过"（`conv_window_rows is not None`，见 §4.10 的 gating）并且
  `spec_tree_slots is not None` 时才建 plan，因为 plan 按 token-major 打包后的坐标索引两张表；
- `_reset_spec_decode_graph_inputs` 把每块的 `cu_seqlens` 清零：没有投机请求的那次重放里，
  录下来的分块调用退化成"所有行长度 0"的空转（与 `spec_actual_seq_lengths` 清零同理）。

> **注意**：是否分块只看 `slots > GDN_RECURRENT_MAX_VERIFY_TOKENS`，不看
> `num_speculative_tokens`；每行**实际**验证多少 token（`≤ slots`，调度器截断或候选不足时
> 更短）由 `cu_seqlens` 与 padding 表达，不需要第二套分支。

---

## 5. 注意事项与踩坑清单

1. **`num_spec` 一改，全 builder 跟着变**。`self.num_spec = slots - 1` 之后
   `block_table[:, :num_spec+1]`、`_fold_spec_sized_prefill_chunks_into_spec`、
   `spec_token_indx` 等全部自动切到树宽度；反过来，任何**硬编码 `num_speculative_tokens`** 的
   新代码都会和这里不一致。改这个文件时优先用 `self.num_spec` / `self.spec_tree_slots`。
2. **两处加宽不是同一个条件**（4.2）：递推状态按 `budget > num_spec` 加宽，conv page 按
   `slots > num_spec` 加宽，差 1 列。
3. **`tree_conv_width` 的 `num_spec` 参数必须匹配传入 shape 的实际列数**：builder 里传
   `max(num_spec, slots)`，`attn_utils` 里传配置的 `num_speculative_tokens`。
4. **token-major 打包是必需品，不是优化**。递推算子按全局 token 偏移读表；短卷积通过
   `cache_indices.reshape(-1)[starts]` 取"每行第 0 列"，也只有打包后这个下标才等于"每行
   token 起点"。路由上：`_pack_tree_rows_to_tokens` 必须同时作用于 `state_indices`、
   `init_state_indices`、`window_rows`（`window_rows` 在 `compute_tree_conv_rows` 内部按同一规则
   打包）。
5. **短卷积必须"先重锚、再取 tap、最后写节点列"**（4.8.5）。顺序错了会读到旧 history 或新的
   自己的激活。
6. **`_tree_buffers` 的宽度必须按 `budget` 分配**，否则每步重建/清零 → 短卷积 re-anchor
   静默失效（4.6）。
7. **`prev_num_sampled` 先置 0 再写**（4.6）：下一步只会 re-anchor 本步真正在批里的请求，
   陈旧路径必须被主动失效。
8. **kernel 里 `initStateGm_` 与 `finalStateGm_` 是同一片 page**，重载前必须
   `CopyOutState` 清 pending + `MTE3_MTE2` 同步（4.9.2b）。
9. **`stateInQueue_` 深度为 1**：树路径在 token 循环内多次重载，所以预取必须挪到循环后
   （4.9.2c），否则队列里残留 tensor 会和重载互相踩。
10. **`MAX_MTP = 16` 是"单次调用"的硬限制**：超过它内核**直接 `return`、状态不更新**。所以
    `slots > 16` 必须走 §4.13 的分块验证（`compute_tree_verify_plan`），不能把整棵树塞进一次
    调用；`slots ≤ 16` 时 plan 是 `None`，保持原来的单次调用路径。
11. **`mamba_cache_mode="align"` 与树互斥**：align 的 conv 窗口平移在树上没有对应语义。报错是
    刻意的，不要绕过。
12. **`budget >= num_speculative_tokens`**：采样器路径只有 draft chain 那么宽，缓冲区按
    `budget` 分配并左对齐拷贝（`prev_path[idx, :path.shape[1]] = path`）。`postprocess_state`
    里的 `assert` 是这条约定的守门人。
13. **`tree_num_nodes == 0` 必须退化为链**：这是 prompt chunk 折入 spec batch 的唯一通路，
    也是 `tree.init_ne_own_nodes` 解释"这一步有没有真的分支"的基准。
14. **全图必须录到带 `init_state_indices` 的 kernel**（4.10），否则重放时静默退回 carry。
15. **配置项不在 `TreeSpecConfig` 里校验**（4.1.1）：这样非 GDN 模型不会因为一个无意义的键
    报错，但代价是"拼错的值只有在跑 GDN 时才会被发现"。
16. **`reorder_batch_threshold = slots`**：树的验证批恒定是整棵树，阈值不跟着改会导致 batch
    重排打乱行内布局。
17. **NPU 同步开销**：所有 `bool(...)` / `.item()` 都要么在 CPU 张量上做，要么放在
    `tree_timer_enabled()` 门后（`row_widths` 的校验、`_count_tree_shape_stats` 都是这么处理的）。

---

## 6. 测试覆盖

| 文件 | 覆盖内容 |
| --- | --- |
| `tests/ut/worker/test_tree_state_rollback.py` | 续跑列（含 `num_sampled == 1` 回落、满路径）、槽位数开关、`align` 拒绝、`GDN_ATTN`/方法校验、`init_state_indices` 的父子关系与折入行链式回退 + budget 不匹配、`_widen_mamba_spec_for_tree` 的槽位/conv 列/padded size、path buffer 宽度（防"每步重建"回归）、**分块 verify plan**（每个 token 恰好被覆盖一次、块边界按父 page 重载、与"一次宽调用"的伪内核状态等价、`slots ≤ 16` 返回 `None`） |
| `tests/ut/ops/test_gdn_tree_conv.py` | `tree_causal_conv1d` 的祖先窗口（dim-first 与 dim-last 两种布局）、零长度行跳过、短行 token-major 打包、`compute_tree_prefix_source` 的恒等映射与节点列、`prefix_src` 与 `compute_tree_resume_column` 的一致性、metadata 默认 `None` = 链 |
| `tests/ut/ops/test_gdn_attn_builder.py` | builder 侧 `_build_tree_init_state_indices` / 打包 / 越界校验 / 无树布局返回 `None`、`_build_tree_conv_rows` 表形状与 `prefix_src`、`_build_tree_verify_plan`（fits-one-call → `None`、无树布局 → `None`、超宽树切块与父 page / scratch 落点），以及既有的非树路径回归 |
| `tests/ut/worker/test_tree_kv_project.py` | `TreeKvCompact` 跳过 Mamba group、compact 后 mamba state 不变 |

---

## 7. 动手实现的最小步骤清单

如果要重新实现（或新增一种 `gdn_state_update`），按这个顺序推：

1. `state_rollback.py`：定义 `GdnStateUpdateMethod` 与注册表，写 `tree_state_slot_count`
   （含 `align` 拒绝）与 `validate_tree_state_slots`；
2. `attn_utils._widen_mamba_spec_for_tree`：把 recurrent page 数与 conv page 列数按 `slots` 加宽；
3. `gdn_attn_builder`：`_resize_spec_buffers_for_tree` 重建 spec 尺寸 buffer，并设置
   `self.num_spec = slots - 1`、`reorder_batch_threshold = slots`；
4. `compute_tree_init_state_indices` + `_pack_tree_rows_to_tokens`，接进 `build()`；
5. 算子加 `init_state_indices` 可选输入（def → aclnn/l0 → tiling → kernel 三处改动 → torch binding），
   在 `ops/gdn.py` 里按需传参；`1 + budget > 16` 时再补 §4.13 的分块验证
   （`compute_tree_verify_plan` + builder 的稳定 buffer + `_chunked_tree_verify`）；
6. 短卷积半边：`tree_conv_width` / `tree_node_parents_depths` / `_tree_conv_tap_rows` /
   `compute_tree_conv_rows` / `compute_tree_prefix_source` / `tree_causal_conv1d`；
7. `compute_tree_resume_column` + `stage_tree_resume_column` + `postprocess_state` 覆写 +
   `prepare_attn` 的路径回读；
8. 收尾：全图 capture 重写、`TreeKvCompact` 跳过 Mamba、`timer.tree_stat`。

**新方法的三个必改点**（代码注释里也写明了）：
`tree_state_slot_count`（状态布局）、
`AscendGDNAttentionMetadataBuilder._build_tree_init_state_indices`（每节点初始状态）、
`ops/gdn.tree_causal_conv1d`（短卷积）。

---

## 8. 已知限制

- 只接通 **Qwen GDN**（`MambaAttentionBackendEnum.GDN_ATTN`）；KDA、Mamba1/2、short-conv、
  纯线性注意力在 `validate_tree_state_slots` 直接报 `NotImplementedError`（而不是悄悄拿陈旧
  状态去算）。
- 与 `mamba_cache_mode="align"` 互斥。
- 每请求状态槽数从 `1 + num_speculative_tokens` 变成 `1 + budget`，因此
  **mamba 状态显存与每步状态读写流量随树宽线性增长**；同时 conv page 每请求增加
  `(slots - num_spec) * conv_dim * dtype_size` 字节。
- 单次算子调用的宽度上限是内核的 `MAX_MTP = 16`；`1 + budget > 16` 由 §4.13 的分块验证
  支持，代价是每步多出 `ceil((1 + budget) / 16) - 1` 次算子调用，以及每层一次
  `rows * 16` 的 gather / 散回（显存与状态读写流量仍随树宽线性增长，见上一条）。
- `tree_spec_config.gdn_state_update` 目前只有 `"snapshot"` 一个已实现值。
