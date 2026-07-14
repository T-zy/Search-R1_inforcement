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

## 未完成的操作（下一阶段）

以下操作在最终方案中规划但尚未实施：

| 操作 | 优先级 | 预计工作量 | 依赖 |
|------|--------|-----------|------|
| 接入 TrajectoryFilter（正确方式） | P1 | 2h | 需找到 verl GRPO advantage 计算位置 |
| 增加 wandb 指标（grpo 零方差等） | P1 | 1h | 需要跑通一次训练获取 baseline |
| 确认数据格式（uid, data_source） | P1 | 30min | 需要检查实际 parquet 数据 |
| Tool metrics 传入 reward（第二阶段） | P2 | 2h | 需要修改 ToolAgentLoop |
| 自动 Early Stop | P2 | 1h | 需要先有稳定的监控指标 |

---

## 当前文件状态

### 修改的文件

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `rewards/qa_em_tool_reward.py` | 完全重写（新签名 + 简化逻辑 + dict 返回） | ~150 |
| `tools/search_tool.py` | HTTP session 复用 + 指标按 instance_id | ~260 |
| `scripts/train_grpo_qwen25_1p5b.sh` | 注册 reward + 参数调整 | ~120 |
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
