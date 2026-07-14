# Verl 版本与修改记录

## 版本信息

| 字段 | 值 |
|------|-----|
| 框架 | `verl-project/verl` |
| 语义版本 | `0.9.0.dev`（来自 `verl/version/version` 文件） |
| 源码路径 | `/home/zytan/verl` |
| Git Commit | `30119a253087bff86c12d329d2d8dd43c589705f` |
| Git Tag | `v0.8.0-176-g30119a25`（v0.8.0 之后 176 个 commit） |
| 复制日期 | 2026-07-14 |

## 复制目的

将 verl 源码 vendored 到本项目中，以便：

1. 对 verl 核心代码做定制修改以支持 Search-R1 训练需求
2. 确保训练环境可复现，不依赖外部 verl 版本更新
3. 记录所有对 verl 的修改，便于复盘和升级时追溯

## 修改清单

### 1. `trainer/ppo/core_algos.py` — GRPO advantage 添加 TrajectoryFilter 支持

**修改函数**：

- `compute_grpo_outcome_advantage()`（line 268）
- `compute_grpo_vectorized_outcome_advantage()`（line 335）

**新增参数**：`valid_sample_mask: Optional[list[bool] | torch.Tensor] = None`

**行为变更**：

当 `valid_sample_mask` 不为 None 时：
1. 根据 `valid_sample_mask` 标记系统异常轨迹（tool timeout、HTTP error、truncation 等）
2. 计算 group mean/std 时**只使用有效轨迹**，异常轨迹不参与 baseline 计算
3. 如果某个 uid group 中有效轨迹数 < 2，该 group 所有轨迹的 advantage 置 0
4. 异常轨迹的 advantage 也置 0，`response_mask` 在外部置 0（两者结合实现完整 loss mask）

当 `valid_sample_mask` 为 None 时，行为与原始 verl 完全一致（向后兼容）。

**修改文件**：`/home/zytan/Search-R1_inforcement/verl_src/trainer/ppo/core_algos.py`

---

### 2. `trainer/ppo/ray_trainer.py` — 传递 valid_sample_mask

**修改函数**：`compute_advantage()`（line 186）

**行为变更**：

在 GRPO 分支中读取 `data.non_tensor_batch.get("valid_sample_mask", None)` 并传给 `compute_grpo_outcome_advantage()`。

当 `valid_sample_mask` 不存在时，行为不变。

**修改文件**：`/home/zytan/Search-R1_inforcement/verl_src/trainer/ppo/ray_trainer.py`

---

### 3. `experimental/agent_loop/tool_agent_loop.py` — 工具 metrics 传入 reward

**修改位置**：`ToolAgentLoop._handle_processing_tools_state()` 中工具响应处理循环（line 324）

**行为变更**：

原代码：
```python
for tool_index, (tool_response, tool_reward, _) in enumerate(responses):
```

修改为：
```python
for tool_index, (tool_response, tool_reward, tool_metrics) in enumerate(responses):
    if tool_metrics:
        for k, v in tool_metrics.items():
            # 累加数值指标到 extra_fields
            # 记录第一个非 "none" 的异常类型
```

工具返回的 metrics 字典（包含 `tool/search_called`、`tool/search_success`、`tool/search_latency_ms` 等）被累加到 `agent_data.extra_fields`，通过 `NaiveRewardManager` 的 `tool_extra_fields` 机制传入自定义 reward 函数。

**修改文件**：`/home/zytan/Search-R1_inforcement/verl_src/experimental/agent_loop/tool_agent_loop.py`

---

## 使用说明

项目中的训练脚本通过以下方式引用 vendored verl：

```bash
# 在 train_grpo_qwen25_1p5b.sh 中设置
export PYTHONPATH="${PROJECT_ROOT}/verl_src:$PYTHONPATH"
```

或者在运行训练时：

```bash
cd /home/zytan/Search-R1_inforcement
PYTHONPATH="${PWD}/verl_src:$PYTHONPATH" python3 -m verl.trainer.main_ppo ...
```

## 与原始 Search-R1 vendored verl 的对比

| 维度 | 原始 Search-R1 | 本项目 |
|------|---------------|--------|
| verl 版本 | 旧版（2024，无确切版本号） | `v0.8.0-176-g30119a25`（2025） |
| 修改方式 | 大量修改（手写 agent loop、修改 trainer） | 最小修改（3 个文件，新增参数向后兼容） |
| agent loop | 手写 `LLMGenerationManager`（600+ 行） | 使用 verl 原生 `ToolAgentLoop` |
| GRPO support | 不支持 | 完整支持 + TrajectoryFilter |
| 升级路径 | 需要手动合并新版 verl | 记录所有 diff，可逐一应用到新版 |
