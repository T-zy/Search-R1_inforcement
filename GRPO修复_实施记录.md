# GRPO 修复 — 实施记录

> **日期**: 2026-07-14  
> **依据**: `GRPO修复_最终执行方案.md`（经三方讨论形成的最终方案）  
> **目标**: 使 Search-R1_inforcement 项目真正形成可运行的 GRPO 训练闭环

---

## 操作清单总览

| # | 操作 | 文件 | 状态 |
|---|------|------|------|
| 1 | 重写奖励函数：新签名 `compute_score()` | `rewards/qa_em_tool_reward.py` | ✅ |
| 2 | 修复 `extract_solution()` bug：`<= 1` → `< 1` | 同上 | ✅ |
| 3 | 移除 `<search>` 标签依赖，改为检测 `<information>` | 同上 | ✅ |
| 4 | 简化奖励逻辑：8 级奖励矩阵 | 同上 | ✅ |
| 5 | 返回 dict：`score` + 6 个分项指标 | 同上 | ✅ |
| 6 | 正式训练脚本注册自定义 reward | `scripts/train_grpo_qwen25_1p5b.sh` | ✅ |
| 7 | Smoke Test 脚本注册自定义 reward | `scripts/train_grpo_smoke_test.sh` | ✅ |
| 8 | 调整训练参数（batch=64, rollout.n=4, steps=150） | `scripts/train_grpo_qwen25_1p5b.sh` | ✅ |
| 9 | SearchTool HTTP session 复用 | `tools/search_tool.py` | ✅ |
| 10 | SearchTool metrics 按 instance_id 保存 | `tools/search_tool.py` | ✅ |
| 11 | 最终方案文档 | `GRPO修复_最终执行方案.md` | ✅ |
| — | **本次实施记录** | **`GRPO修复_实施记录.md`** | ✅ |

---

## 操作详录

### 操作 1-5：重写奖励函数

**文件**: `recipe/search_r1_verl/rewards/qa_em_tool_reward.py`

#### 1.1 函数签名变更

```diff
- def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0.2, ...) -> float:
+ def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
```

**原因**: verl 的 `NaiveRewardManager` 调用 `compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)`。原函数名和参数都不匹配，永远不会被调用。

**验证方式**: 训练日志中出现 `Loaded reward function 'compute_score' from ...`。

#### 1.2 `extract_solution()` bug 修复

```diff
- if len(matches) <= 1:
+ if not matches:
```

**原因**: `<= 1` 导致恰好有一个 `<answer>` 标签时（正常情况）返回 `None`，所有正常轨迹的答案都无法提取。这是原版 Search-R1 继承来的 bug。

**影响**: 修复前几乎所有轨迹的 `answer` 为 `None`，奖励永远拿不到 1.0。

#### 1.3 移除 `<search>` 标签依赖

删除了整个 `is_valid_sequence()` 函数，该函数包含对 `<search>` 标签的强制检查。

新方案改为从 `<information>` 块检测搜索行为：

```python
def extract_search_info(solution_str: str) -> tuple[int, bool]:
    blocks = extract_information_blocks(solution_str)
    if not blocks:
        return 0, False
    success = any("Search failed:" not in b for b in blocks)
    return len(blocks), success
```

**原因**: 在 `format=hermes` 下，模型使用 function call 而非 `<search>` 标签。`<search>` 不会出现在模型生成的文本中。

#### 1.4 简化奖励矩阵

旧方案（7 种情况，3 个可调参数）→ 新方案（8 级奖励，线性累加）：

| 条件 | 累加值 |
|------|--------|
| 答案正确 | +0.8 |
| 有 `<answer>` 标签 | +0.1 |
| 搜索成功 | +0.05 |
| 检索证据包含答案 | +0.05 |
| 多跳数据集 + 零搜索 | -0.2 |
| **范围** | **[-0.2, 1.0]** |

#### 1.5 返回 dict 方便 wandb 指标

```python
return {
    "score": float(reward),
    "answer_em": float(answer_correct),
    "has_final_answer": float(has_answer),
    "num_search_calls": float(num_search_calls),
    "search_success": float(search_success),
    "retrieval_correct": float(retrieval_ok),
    "no_search_penalty": float(is_multi_hop and num_search_calls == 0),
}
```

verl 的 `NaiveRewardManager` 遇到 dict 返回时，自动将 `score` 作为主奖励，其余字段通过 `reward_extra_info` 注册到 wandb。

---

### 操作 6-8：训练脚本修改

**文件**: `scripts/train_grpo_qwen25_1p5b.sh`, `scripts/train_grpo_smoke_test.sh`

#### 6-7 注册自定义奖励函数

```bash
reward.custom_reward_function.path="${PROJECT_ROOT}/recipe/search_r1_verl/rewards/qa_em_tool_reward.py"
reward.custom_reward_function.name=compute_score
```

**配置路径确认**: 根据 verl 源码 `verl/trainer/ppo/reward.py` 第 70 行：
```python
reward_fn_config = config.reward.get("custom_reward_function") or {}
```
确认需使用 `reward.` 前缀。

#### 8 参数调整

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `TRAIN_BATCH_SIZE` | 256 | 64 | 4×L20 multi-turn 过大 |
| `ROLLOUT_N` | 5 | 4 | 同上 |
| `MAX_RESPONSE_LENGTH` | 2048 | 1024 | 减少显存压力 |
| `MAX_TOOL_RESPONSE_LENGTH` | 2048 | 1024 | 同上 |
| `GPU_MEM_UTIL` | 0.35 | 0.45 | 1.5B 可提升 |
| `LOG_PROB_MICRO_BATCH_SIZE` | 32 | 16 | 适配小 batch |
| `LR` | 1e-6 | 5e-7 | GRPO 常用更小 lr |
| `TOTAL_STEPS` | 1005 | 150 | 第一阶段验证 |
| `val_before_train` | false | true | smoke test 通过后可 |
| `save_freq` / `test_freq` | 100 | 25 | 更频繁验证 |
| `val_batch_size` | 128 | 64 | 减少验证显存 |

---

### 操作 9-10：SearchTool 工程优化

**文件**: `tools/search_tool.py`

#### 9 HTTP session 复用

```python
# 旧：每次 execute() 新建 session
async def _call_retrieval_service(self, query, topk):
    async with aiohttp.ClientSession() as session: ...

# 新：lazy init，复用 session
self._http_session: aiohttp.ClientSession | None = None

async def _get_http_session(self):
    if self._http_session is None:
        self._http_session = aiohttp.ClientSession(...)
    return self._http_session
```

**原因**: GRPO `n=4` 时每步可能产生数百次工具调用，频繁创建 HTTP session 有额外开销。

#### 10 指标按 instance_id 保存

```diff
- self._session_metrics: dict[str, float] = {}
+ self._metrics_by_instance: dict[str, dict] = {}
+ self._latencies_by_instance: dict[str, list] = {}
```

所有 `_record_metrics()` 调用增加 `instance_id` 参数，每个 trajectory 独立计数。

---

## 已完成的操作（后续追加）

以下操作在原最终方案中规划为 P1/P2，已在后续实施中完成：

| 操作 | 原优先级 | 完成方式 |
|------|---------|----------|
| 接入 TrajectoryFilter（正确方式） | P1 | 在 `verl_src/trainer/ppo/core_algos.py` 中为两个 GRPO advantage 函数添加 `valid_sample_mask` 参数 |
| 增加 wandb 指标（grpo 零方差等） | P1 | 通过 reward 函数返回 dict 自动注册 6 个分项指标 |
| Tool metrics 传入 reward（第二阶段） | P2 | 在 `verl_src/experimental/agent_loop/tool_agent_loop.py` 中捕获 tool_metrics 写入 extra_fields |

## 2026-07-14 最终实施方案执行记录

依据 `GPT最终实施方案_v2.0.md` 执行了以下修改：

### 第 1 步：加载 patched verl
- 创建 `verl -> verl_src` 符号链接
- 修改两个训练脚本：`set -euo pipefail`、`export PYTHONPATH`、`cd "${PROJECT_ROOT}"`
- 删除 `VERL_ROOT="/home/zytan/verl"` 和 `cd "${VERL_ROOT}"`
- 增加强制启动检查（验证 `verl.__file__`、`core_algos`、`ToolAgentLoop` 均指向项目内路径）
- 增加 KL 配置：`actor_rollout_ref.actor.use_kl_loss=true` 等

### 第 2 步：修复 SearchTool
- 空结果判定为失败：`num_docs==0` 时 `search_success=0`、`exception_type="empty_result"`
- `</information>` 标签保护：先拼 body 控制长度，再加标签
- 删除 `"tool/search_success": 1` 固定值，改为条件赋值

### 第 3 步：修改 reward 函数
- 新增 `extract_tool_state()` 函数：Tool metrics 为主，XML fallback
- `compute_score()` 使用 `has_successful_search` 替代旧的 `search_success`
- 多跳惩罚条件改为 `not has_successful_search` 而非 `num_search_calls == 0`
- 返回字段增加 `tool_metrics_available`、`search_success_count`、`search_failed_count`、`search_timeout_count`、`search_num_docs`、`all_searches_successful`、`missing_successful_search_penalty`

### 第 4 步：验证 Tool metrics 管道
- Smoke test 脚本增加 `export SEARCH_R1_DEBUG_PIPELINE=1`
- reward 文件增加模块级 `_PIPELINE_DEBUG_PRINTED` 和一次性调试日志

### 第 5 步：接入 TrajectoryFilter
- `trajectory_filter.py`：`DEFAULT_MASK_ON` 移除 `TOOL_RESPONSE_TRUNCATED`、`MAX_TOOL_CALLS_EXCEEDED`
- `ray_trainer.py`：
  - 新增 `build_trajectory_filter_outputs()` 函数
  - 新增 `compute_grpo_group_metrics()` 函数
  - 在 `extract_reward()` 后集成 TrajectoryFilter：生成 `valid_sample_mask`、清零 invalid `response_mask`、聚合 trajectory/group metrics
  - reward extra info 提前合并到 `non_tensor_batch`

### 第 6 步：修复非向量版 GRPO
- 完整重写 `compute_grpo_outcome_advantage()`：`normalized_scores = torch.zeros_like(scores)` 确保 invalid 默认为 0
- `n_valid < 2` 时整组 `normalized_scores[group_indices] = 0.0`
- 使用 `valid_mask[group_indices]` 索引，只计算有效轨迹的 mean/std

### 第 7 步：修复 trajectory_metrics.py
- 修复空数组 `n_lat` 未定义 bug
- 所有 `if x:` 改为显式 `if x is not None and len(x) > 0:`
- 增加 `float()` 转换处理 NumPy 类型

### 第 8 步：开启 actor KL loss ✓（在第 1 步中一并完成）

### 第 9 步：增加 system prompt
- HotpotQA 增加 system prompt（多跳 + 必须搜索 + `<answer>` 格式）
- NQ 增加 system prompt（可搜索 + `<answer>` 格式）
- ability 字段：HotpotQA 改为 `"multi-hop-reasoning"`

### 第 10 步：单元测试（31 个全部通过）
| 测试文件 | 用例数 | 状态 |
|---------|--------|------|
| `tests/test_qa_em_tool_reward.py` | 17 | ✅ 全部通过 |
| `tests/test_search_tool_formatting.py` | 6 | ✅ 全部通过 |
| `tests/test_tool_metrics_pipeline.py` | 8 | ✅ 全部通过 |
| `tests/test_grpo_valid_sample_mask.py` | 4 | ⏳ 需 search-r1 conda 环境（含 torch） |

---

## 仍待完成的操作

| 操作 | 优先级 | 说明 |
|------|--------|------|
| 重新生成 Parquet 数据 | **P0** | 修改转换脚本后必须重新执行并覆盖 |
| GRPO mask 单元测试 | P1 | 需要 conda activate search-r1（含 torch） |
| 2-step smoke test | P1 | 完成上述后运行 |
| Tool-Agent SFT 冷启动 | P1 | 50-step GRPO 前必须完成 |
| 50-step GRPO 训练 | P2 | Smoke test 通过后 |
| 自动 Early Stop | P2 | 需先有稳定监控指标 |

---

## 当前文件状态

### 本次修改的文件

| 文件 | 修改内容 |
|------|----------|
| `verl_src/trainer/ppo/ray_trainer.py` | 新增 `build_trajectory_filter_outputs()`、`compute_grpo_group_metrics()`、训练循环集成 TrajectoryFilter |
| `verl_src/trainer/ppo/core_algos.py` | 完整重写 `compute_grpo_outcome_advantage()`（修复 mask 实现） |
| `recipe/search_r1_verl/rewards/qa_em_tool_reward.py` | 新增 `extract_tool_state()`、`import os`、pipeline debug 日志 |
| `recipe/search_r1_verl/tools/search_tool.py` | 空结果判定、XML 标签保护、metrics 条件赋值 |
| `recipe/search_r1_verl/monitoring/trajectory_filter.py` | 更新 `DEFAULT_MASK_ON` |
| `recipe/search_r1_verl/monitoring/trajectory_metrics.py` | 修复空数组 bug、显式 None 检查 |
| `recipe/search_r1_verl/scripts/train_grpo_qwen25_1p5b.sh` | PYTHONPATH、启动检查、KL 配置 |
| `recipe/search_r1_verl/scripts/train_grpo_smoke_test.sh` | PYTHONPATH、启动检查、KL 配置、DEBUG_PIPELINE |
| `recipe/search_r1_verl/data/convert_hotpotqa_to_parquet.py` | 增加 system prompt、ability 改为 multi-hop-reasoning |
| `recipe/search_r1_verl/data/convert_nq_to_parquet.py` | 增加 system prompt |
| `tests/test_qa_em_tool_reward.py` | 17 个 reward 测试用例 |
| `tests/test_search_tool_formatting.py` | 6 个格式化测试用例 |
| `tests/test_tool_metrics_pipeline.py` | 8 个 pipeline 测试用例 |
| `tests/test_grpo_valid_sample_mask.py` | 4 个 GRPO mask 测试用例（需 torch） |
| `tests/__init__.py` | 空包文件 |
| `verl -> verl_src` | 符号链接创建 |
| `scripts/train_grpo_smoke_test.sh` | 注册 reward + 参数调整 | ~95 |

### 新增的文档

| 文件 | 说明 |
|------|------|
| `GRPO修复_最终执行方案.md` | 三方讨论后的最终执行方案（368 行） |
| `GRPO修复_实施记录.md` | **本文档**，实施操作记录 |

---

## 下一步建议

1. **启动检索服务**: `bash recipe/search_r1_verl/scripts/run_retrieval_service.sh`
2. **准备 smoke test 数据**: 确保 `data/smoke_test/` 下有 10 条 parquet
3. **运行 smoke test**: `bash recipe/search_r1_verl/scripts/train_grpo_smoke_test.sh`
4. 确认 wandb 中出现 `reward/answer_em`, `reward/num_search_calls` 等分项指标
5. 确认日志中出现 `Loaded reward function 'compute_score' from ...`
6. Smoke test 通过后再跑 50 step 验证搜索行为不崩溃

---

*文档版本: v1.0*
*编制日期: 2026-07-14*
