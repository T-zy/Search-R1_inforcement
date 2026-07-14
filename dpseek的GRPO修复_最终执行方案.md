# GRPO 修复 — 最终执行方案

> **执行状态追踪**（2026-07-14）:
> - ✅ P0 全部完成（奖励函数重写、训练脚本注册、SearchTool 优化）
> - ✅ P1 TrajectoryFilter 已完成（verl_src/core_algos.py 添加 valid_sample_mask）
> - ✅ P2 Tool metrics 传入 reward 已完成（verl_src/tool_agent_loop.py 修改）
> - ⬜ P1 数据格式确认 — 待完成
> - ⬜ P2 自动 Early Stop — 待完成
> - 📦 verl 已 vendored 到 `verl_src/`（version `0.9.0.dev`, commit `30119a25`）

## 三方观点汇总

| 来源 | 文档 |
|------|------|
| AI-A（原方案） | `Search-R1_inforcement_GRPO_修复实施方案.md` |
| AI-B（我的评估） | `GRPO修复方案_评估报告.md` |
| AI-C（复核意见） | `对_GRPO修复方案评估报告_的复核意见.md` |

以下是我吸收复核意见后，修正了自己此前评估中的 4 处不准确判断，形成的最中执行方案。

---

## 我对复核意见的回应

复核意见指出了我评估报告中的 4 处不准确之处，**全部接受修正**：

| 原评估（我需要纠正的） | 复核意见 | 修正结论 |
|---|---|---|
| 原 `token_f1` 实现有计算 bug | 数学结果正确，只是效率差 | ❌ 我的判断有误，原实现数学上正确 |
| 4×L20 必须全部 offload | 1.5B 模型大概率不需要 | ❌ 我的判断过于绝对，应实测决定 |
| `top_p=0.95` 一定错误 | `top_p=0.95` 有其适用场景 | ❌ 我的判断过于绝对 |
| `val_before_train=true` 有风险 | 通过 smoke test 后无风险 | ⚠️ 我的判断有道理但前提不同 |

其余 75% 的评估判断得到了复核意见的认可。

此外，复核意见补充了 3 个我遗漏的关键问题：

| 补充的问题 | 说明 |
|---|---|
| CLI 配置路径应为 `reward.custom_reward_function.path` | 我的评估中写的是 `custom_reward_function.path`，少了 `reward.` 前缀 |
| Tool metrics 不会自动进入 reward | `SearchTool.execute()` 返回的 metrics 被 verl ToolAgentLoop 丢弃，无法通过 `extra_info` 在 reward 中获取 |
| 仅设 response_mask=0 不能解决 GRPO group 污染 | 异常轨迹的 reward 仍参与 group mean/std 计算，需要在 advantage 计算前过滤 |

---

## 最终执行方案

### P0：立即完成（Smoke Test 前）

#### 1. 注册自定义奖励函数

修复训练脚本 `train_grpo_qwen25_1p5b.sh`，增加：

```bash
reward.custom_reward_function.path="${PROJECT_ROOT}/recipe/search_r1_verl/rewards/qa_em_tool_reward.py"
reward.custom_reward_function.name=compute_score
```

**验证方式**：训练日志中应出现 `Loaded reward function 'compute_score' from ...`。

#### 2. 修改奖励函数签名

`qa_em_tool_reward.py` 中：

```python
# 旧（不会被调用）
def compute_score_em(solution_str, ground_truth, method='strict', ...)

# 新（verl 标准接口）
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
```

#### 3. 修复 `extract_solution()` bug

```python
# 旧：if len(matches) <= 1: return None   ← 1个匹配时也返回None
# 新：if len(matches) < 1: return None     ← 只有0个匹配才返回None
# 或：if not matches: return None
```

#### 4. 移除对 `<search>` 标签的依赖

奖励函数不再检查 `<search>` 标签。改为检测 `<information>` 块的存在性：

```python
information_blocks = extract_information_blocks(solution_str)
num_search_calls = len(information_blocks)
search_success = any("Search failed:" not in b for b in information_blocks)
```

同时保留 `<think>`、`<information>`、`<answer>` 标签的检查，但 `<search>` 不再作为格式验证的必要条件。

#### 5. 简化奖励函数

采用经过复核确认的简化奖励设计：

| 情况 | 奖励 |
|---|---|
| 搜索后答对（HotpotQA） | 1.0 |
| 搜索后答对（NQ） | 0.95 |
| 不搜索直接答对（NQ） | 0.90 |
| 不搜索直接答对（HotpotQA） | 0.70 |
| 搜索 + 检索含答案 + 答错 | 0.30 |
| 搜索 + 答错 | 0.15 |
| 不搜索 + 答错 + 有 answer 标签 | -0.10 |
| 无 answer 标签 | 0.0 |

核心逻辑：

```python
reward = 0.0
if answer_correct:
    reward += 0.8
if has_final_answer:
    reward += 0.1
if search_success:
    reward += 0.05
if retrieval_contains_answer:
    reward += 0.05
if data_source == "hotpotqa" and num_search_calls == 0:
    reward -= 0.2
reward = max(-0.2, min(1.0, reward))
```

#### 6. 奖励函数返回 dict

```python
return {
    "score": float(reward),
    "answer_em": float(answer_correct),
    "has_final_answer": float(has_final_answer),
    "num_search_calls": float(num_search_calls),
    "search_success": float(search_success),
    "retrieval_correct": float(retrieval_correct),
    "no_search_penalty": float(no_search_penalty),
}
```

`score` 为主奖励，其余字段自动注册为 wandb 指标。

#### 7. 第一阶段搜索判断：基于 `<information>` 块

由于 tool metrics 当前不会自动传入 reward，第一阶段直接解析 `solution_str` 中的 `<information>` 块：

```python
def extract_search_info(solution_str: str) -> tuple[int, bool]:
    """返回 (搜索次数, 是否有成功搜索)"""
    blocks = extract_information_blocks(solution_str)
    if not blocks:
        return 0, False
    success = any("Search failed:" not in b for b in blocks)
    return len(blocks), success
```

第二阶段再改为从 `extra_info` 中读取 tool metrics。

### P1：50~150 Step 训练前完成

#### 8. 接入 TrajectoryFilter（正确方式）

**关键修正**：仅设置 `response_mask=0` **不能**阻止异常轨迹污染 GRPO group mean/std。必须在 advantage 计算前过滤。

正确流程：

```python
# 伪代码：需要在 verl 的 GRPO advantage 计算函数中修改
for uid in unique_uids:
    group_indices = get_group_indices(uid)
    valid_indices = [idx for idx in group_indices if valid_sample_mask[idx]]

    if len(valid_indices) < 2:
        # 整组跳过
        advantage[group_indices] = 0
        loss_mask[group_indices] = 0
        continue

    # 只用有效轨迹计算 mean/std
    valid_rewards = rewards[valid_indices]
    mean = valid_rewards.mean()
    std = valid_rewards.std()

    advantage[valid_indices] = (valid_rewards - mean) / (std + 1e-6)
    advantage[invalid_indices] = 0
    loss_mask[invalid_indices] = 0
```

**接入点**：需要找到 verl 中 GRPO advantage 计算的位置（`verl/trainer/ppo/core_algos.py` 中的 `compute_grpo_outcome_advantage` 或类似函数），在其中加入 `valid_sample_mask` 逻辑。

#### 9. 训练参数调整

```bash
# 第一阶段（Smoke Test + 50 step）
TRAIN_BATCH_SIZE=64
ROLLOUT_N=4
MAX_RESPONSE_LENGTH=1024
MAX_TOOL_RESPONSE_LENGTH=1024
GPU_MEM_UTIL=0.45
PPO_MICRO_BATCH_SIZE=4
LOG_PROB_MICRO_BATCH_SIZE=16
LR=5e-7
KL_COEF=0.001
TOTAL_STEPS=150
```

```bash
# 第二阶段（正式训练，根据显存测试调整 offload）
# 先尝试关闭 actor offload：
actor_rollout_ref.actor.fsdp_config.param_offload=false
actor_rollout_ref.actor.fsdp_config.grad_offload=false
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
# 出现 OOM 时按此顺序逐步启用：optimizer → param → grad
```

```bash
# top_p 保持 1.0（与常见 GRPO 基线一致）
# 若工具调用格式错误率过高，再测试 top_p=0.95
```

#### 10. 增加监控指标

通过 reward 返回的 dict，自动获得以下 wandb 指标：

- `reward/answer_em` — 答案正确率
- `reward/has_final_answer` — 有最终答案的比例
- `reward/num_search_calls` — 平均搜索次数
- `reward/search_success` — 搜索成功率
- `reward/retrieval_correct` — 检索包含答案的比例
- `reward/no_search_penalty` — 无搜索惩罚占比

额外监控：

- `rollout/response_length_mean` — 输出长度
- `grpo/group_reward_std_mean` — 组内奖励标准差均值
- `grpo/zero_variance_group_rate` — 零方差组比例
- `grpo/dropped_group_rate` — 被跳过 group 比例（TrajectoryFilter 生效后）

#### 11. 确认数据格式

确认数据中：

- `data_source` 为 `"nq"` 或 `"hotpotqa"`（自定义 reward 中处理，无需改为 `searchR1_*`）
- `reward_model.ground_truth.target` 格式正确
- 每条数据有稳定的 `uid`（verl 应自动生成，需确认）

### P2：性能优化

#### 12. SearchTool HTTP session 复用

```python
self._http_session = None

async def _get_session(self):
    if self._http_session is None:
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
    return self._http_session
```

#### 13. Tool metrics 按 instance_id 保存

```python
self._metrics_by_instance: dict[str, dict] = {}
self._latencies_by_instance: dict[str, list] = {}
```

#### 14. Tool metrics 传入 reward（第二阶段）

修改 ToolAgentLoop 将工具 metrics 保存到 `extra_info`，使 reward 函数能直接获取：

```python
extra_info["num_search_calls"] = total_search_calls
extra_info["search_success_count"] = success_count
```

#### 15. 自动 Early Stop

当以下条件持续多步时自动停止：

- `no_search_ratio > 0.9` 持续 10 step
- `search_calls_mean < 0.1` 持续 10 step
- `zero_variance_group_rate > 0.8`
- `tool/search_success_rate < 0.8`

---

## 执行顺序总表

| 步骤 | 内容 | 优先级 | 预计时间 | 验证方式 |
|------|------|--------|----------|----------|
| 1 | 注册自定义 reward + 修复签名 | P0 | 15min | 日志出现 `Loaded reward function` |
| 2 | 修复 `extract_solution()` bug | P0 | 5min | 答案正常提取 |
| 3 | 移除 `<search>` 依赖 | P0 | 20min | 格式验证不再依赖 `<search>` |
| 4 | 简化奖励逻辑 + 返回 dict | P0 | 30min | 分项指标出现在 wandb |
| 5 | **Smoke Test**（10 条数据） | P0 | 5min | 训练不报错，tool call 成功 |
| 6 | 接入 TrajectoryFilter（正确方式） | P1 | 2h | 异常轨迹不参与 group mean/std |
| 7 | 调整训练参数 | P1 | 15min | 无 OOM |
| 8 | **50 step 训练** | P1 | 30min | 搜索行为不崩溃 |
| 9 | **150 step 训练验证** | P1 | 1.5h | HotpotQA 搜索率 > NQ |
| 10 | SearchTool 工程优化 | P2 | 1h | 性能提升 |
| 11 | Tool metrics 传入 reward | P2 | 2h | extra_info 中可读 |
| 12 | 正式训练 | P2 | 8h+ | EM 可复现 |

---

## 关键验证检查点

### Smoke Test 通过条件

- [ ] 自定义 reward 被调用（日志或 wandb 确认）
- [ ] 每个 prompt 确实生成了 `ROLLOUT_N` 条轨迹
- [ ] 同组 uid 一致
- [ ] Tool Agent 成功调用检索服务（`<information>` 块出现）
- [ ] reward 能区分搜索与不搜索轨迹（搜索轨迹 reward 更高）
- [ ] `<answer>` 标签被正常提取（`extract_solution` 返回非 None）

### 50 Step 通过条件

- [ ] `no_search_ratio` 不快速接近 1
- [ ] `search_calls_mean` 不快速降到 0
- [ ] `response_length_mean` 不从几百骤降到几十
- [ ] `tool/search_success_rate` > 95%
- [ ] `zero_variance_group_rate` 不长期 > 50%

### 150 Step 通过条件

- [ ] HotpotQA 搜索率明显高于 NQ
- [ ] HotpotQA EM/F1 有提升趋势
- [ ] 搜索后答对率 > 不搜索答对率
- [ ] reward 与搜索次数不呈强负相关
- [ ] 不搜索答对不成为主要提升来源

---

## 附录：我对复核意见的详细回应

### 1. `token_f1` 实现争议

**复核指正**：原实现数学上正确，只是效率低。

**我接受**。我此前评估中的"bug"判断有误。`min(pred_tokens.count(token), gold_tokens.count(token))` 对同一 token 重复计算时每次结果相同，最终 `sum(common.values())` 正确。但代码仍建议改为 `Counter` 实现，原因从"修复 bug"改为"提升效率和代码可读性"。

### 2. offload 争议

**复核指正**：Qwen2.5-1.5B（~3GB）在 4×L20（184GB 总和）上很可能不需要全部 offload。

**我接受**。我此前的判断过于保守。FSDP offload 的本质是用速度换显存，对于 1.5B 模型和 4×46GB 的配置，确实可以先关闭全部 offload 实测，仅在 OOM 时按需启用。

### 3. `top_p=0.95` 争议

**复核指正**：`top_p=0.95` 在工具调用格式不稳定时有正面作用，并非一定错误。

**我接受**。我此前的"RL 必须用 top_p=1.0"过于绝对。修正为：初始保持 top_p=1.0 以对齐常见基线，若工具调用格式错误率过高，可测试 top_p=0.95。

### 4. `val_before_train=true` 争议

**复核指正**：只要 smoke test 验证通过，`val_before_train=true` 本身没有风险。

**我接受**。我此前的"有风险"判断是在假设用户直接跑正式训练的前提下。正确的流程是先 smoke test 再正式训练，在此前提下 val_before_train=true 是推荐配置。

### 5. `reward.custom_reward_function.path` 配置路径

**复核指正**：最新版 verl 的配置路径是 `reward.custom_reward_function.path` 而非 `custom_reward_function.path`。

**我接受**。我此前的评估写的是 `custom_reward_function.path`，缺少 `reward.` 前缀。verl 源码中 `get_custom_reward_fn()` 读取的是 `config.reward.get("custom_reward_function")`，所以 CLI 配置必须使用 `reward.custom_reward_function.path=...`。

### 6. TrajectoryFilter 的 GRPO group 污染问题

**复核指正**：仅设置 `response_mask=0` 和 `reward=0` 不能阻止异常轨迹污染 group mean/std。

**我接受**。这是我评估报告中最大的遗漏。GRPO advantage 计算顺序是：先算所有 reward → 再算 group mean/std → 再算 advantage → 最后应用 mask。异常轨迹即使 reward=0 也会拉低 group mean，从而抬高其他轨迹的 advantage。解决方案必须是在 mean/std 计算前过滤异常轨迹。

---

*文档版本: v1.0（最终版）*
*编制日期: 2026-07-14*
