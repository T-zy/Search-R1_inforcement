# 基于最新版 verl 的 Search-R1 检索增强推理复现与优化实施规划

> 项目名称：基于 verl 框架的 Search-R1 检索增强推理复现与优化
> 目标模型：Qwen2.5-3B
> 目标框架：最新版 `verl-project/verl`
> 原始项目参考：[T-zy/Search-R1
> 新文件](https://github.com/T-zy/Search-R1)路径：/home/zytan/Search-R1_inforcement
> 本文用途：作为交给另一个 AI 或开发者执行的详细改造任务书

---

## 1. 总体判断

当前 `T-zy/Search-R1` 仓库本质上是基于旧版 veRL / verl 的 Search-R1 fork，仓库中直接 vendor 了旧版 `verl/` 代码，并通过 `search_r1/llm_agent/generation.py` 手写了 agent loop：

- 手动解析 `<search>...</search>` 与 `<answer>...</answer>`；
- 手动调用 HTTP 检索服务；
- 手动拼接 `<information>...</information>`；
- 手动维护 rolling state、attention mask、info mask；
- 手动实现多轮生成与搜索交替。

新版 verl 已经提供了更原生的：

- Tool Agent Loop；
- `BaseTool` 工具接口；
- multi-turn rollout；
- async rollout；
- rollout trace；
- response mask / loss mask 相关训练入口。

因此，本项目不建议继续魔改旧仓库中的 vendor `verl/`，而应当：

1. 使用最新版 `verl-project/verl` 作为训练主框架；
2. 保留 Search-R1 中已经验证有效的数据处理、检索服务、奖励设计思想；
3. 将本地检索服务封装为新版 verl 原生 Tool；
4. 用新版 Tool Agent Loop 替代旧的手写 agent loop；
5. 在新版 trainer 中加入异常 trajectory 监控与 loss mask。

一句话：旧 Search-R1 作为业务逻辑参考，新版 verl 作为真正训练底座。

---

## 2. 目标拆解

项目目标拆成 6 个可交付模块：

| 模块               | 目标                                      | 产物                            |
| ------------------ | ----------------------------------------- | ------------------------------- |
| 新版 verl 工程迁移 | 从旧`main_ppo_format` 迁到新版 verl     | 新训练脚本与配置                |
| 检索服务工程化     | 保留 E5 + FAISS IVF + FastAPI，增强稳定性 | `/retrieve`、`/health` 服务 |
| Search Tool 标准化 | 将检索服务封装为 verl`BaseTool`         | `SearchTool` 与 tool config   |
| SFT 冷启动         | 用教师轨迹训练 Qwen2.5-1.5B               | LLaMA-Factory SFT 数据与权重    |
| GRPO 训练          | EM 主奖励 + 格式辅助奖励                  | custom reward + GRPO 脚本       |
| 异常轨迹监控与过滤 | 追踪异常 trajectory 并 loss mask          | filter、metrics、wandb 指标     |

---

## 4. 阶段 0：统一实验口径

当前实验总结中主要使用的是 Qwen2.5-3B-Instruct，但项目目标写的是 Qwen2.5-1.5B。因此正式实现前需要统一实验设定。

推荐实验矩阵：

| 阶段       | 模型                        | 目的                                      |
| ---------- | --------------------------- | ----------------------------------------- |
| smoke test | Qwen2.5-3B-Instruct         | 快速验证 tool loop、reward、mask 能否跑通 |
| 主实验     | Qwen2.5-3B-Base + SFT       | 对齐项目简介                              |
| 对照实验   | 原 Search-R1 或旧版 3B 配置 | 作为论文复现/提升对照                     |

注意：

- 不建议直接用 Qwen2.5-3B-Base 跑 RL；
- Base 模型通常不能稳定产生工具调用格式；
- 应先通过教师轨迹 SFT 冷启动，再进入 GRPO。

---

## 5. 阶段 1：检索服务工程化

当前已经验证稳定的方案是：

- E5 向量模型；
- FAISS `IndexIVFFlat`；
- `e5_IVF4096_Flat.index`；
- CPU FAISS 检索；
- wiki-18.jsonl 常驻内存；
- FastAPI 提供 HTTP 服务；
- 4×L20 全部留给训练。

这一路线应继续保留。不要优先尝试 GPU Flat FAISS。已有实验表明：

- 61GB Flat index 单 L20 放不下；
- 多卡 FAISS 有 cuBLAS error / segfault 风险；
- GPU 检索会与训练争抢显存；
- CPU IVF 已能达到约 4.7ms/次，稳定性更重要。

### 5.1 增加 `/health` 接口

新增健康检查接口：

```json
{
  "status": "ok",
  "index_loaded": true,
  "corpus_loaded": true,
  "retriever": "e5",
  "index_type": "IVF4096_Flat",
  "topk_default": 3
}
```

### 5.2 增强 `/retrieve` 请求参数

建议支持：

```json
{
  "queries": ["What is reinforcement learning?"],
  "topk": 3,
  "return_scores": true,
  "max_doc_chars": 1200
}
```

### 5.3 服务端保护

必须加入：

- `topk` 上限，例如最多 5；
- 空 query 直接返回空结果；
- 单篇文档按 `max_doc_chars` 截断；
- HTTP 请求超时；
- LRU query cache；
- 平均检索耗时统计；
- 异常日志记录。

---

## 6. 阶段 2：实现 verl 原生 Search Tool

最新版 verl 的工具接口核心是：

```python
async def execute(
    self,
    instance_id: str,
    parameters: dict[str, Any],
    **kwargs
) -> tuple[ToolResponse, float, dict]:
    ...
```

需要新增：

```text
recipe/search_r1_verl/tools/search_tool.py
recipe/search_r1_verl/tools/search_tool_config.yaml
```

### 6.1 Tool 行为

Tool 名称：`search`

参数：

- `query: str`
- `topk: int | None`

行为：

1. 接收模型发出的 tool call；
2. 读取 `query` 和 `topk`；
3. 调用本地检索服务：

```text
POST http://127.0.0.1:8000/retrieve
```

4. 将检索结果格式化为：

```text
<information>
Doc 1(Title: ...)
...
Doc 2(Title: ...)
...
</information>
```

5. 返回：

```python
ToolResponse(text=formatted_information), 0.0, tool_metrics
```

### 6.2 Tool metrics

`execute()` 返回的第三项 metrics 必须包含：

```python
{
    "tool/search_called": 1,
    "tool/search_success": 1,
    "tool/search_failed": 0,
    "tool/search_timeout": 0,
    "tool/search_empty_query": 0,
    "tool/search_latency_ms": 12.3,
    "tool/search_num_docs": 3,
    "tool/search_response_truncated": 0,
    "tool/search_exception_type": "none"
}
```

异常类型建议：

```text
none
timeout
http_error
parse_error
empty_query
retriever_unavailable
unknown_error
```

### 6.3 Tool 配置示例

```yaml
tools:
  - class_name: recipe.search_r1_verl.tools.search_tool.SearchTool
    config:
      type: native
      endpoint: http://127.0.0.1:8000/retrieve
      timeout: 5
      default_topk: 3
      max_topk: 5
      max_doc_chars: 1200
      max_tool_response_chars: 6000
    tool_schema:
      type: function
      function:
        name: search
        description: Search Wikipedia passages relevant to a factual question.
        parameters:
          type: object
          properties:
            query:
              type: string
              description: Search query.
            topk:
              type: integer
              description: Number of passages to retrieve.
          required:
            - query
```

---

## 7. 阶段 3：数据格式迁移

新版 multi-turn/tool rollout 推荐使用 raw chat 数据。

训练 parquet 每条样本至少包含：

```python
{
    "data_source": "nq",
    "prompt": [
        {
            "role": "user",
            "content": "Question ..."
        }
    ],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "target": ["answer1", "answer2"]
        }
    },
    "extra_info": {
        "split": "train",
        "index": idx
    }
}
```

verl 配置中必须打开：

```bash
data.return_raw_chat=True
```

建议同时设置：

```bash
data.truncation=error
data.filter_overlong_prompts=True
```

这样可以在数据阶段暴露过长 prompt，而不是在训练中静默截断。

---

## 8. 阶段 4：教师轨迹蒸馏与 SFT 冷启动

正式目标是 Qwen2.5-3B，因此建议先做 SFT 冷启动。

### 8.1 教师轨迹格式

建议统一成 tool-call 对话格式，而不是旧 Search-R1 的纯文本 `<search>` 标签格式。

每条轨迹逻辑如下：

```text
user: question
assistant: reasoning + call search(...)
tool: <information>Doc 1 ... Doc 2 ...</information>
assistant: continue reasoning, optionally call search(...)
tool: ...
assistant: final answer
```

### 8.2 SFT 数据筛选规则

保留样本：

- 最终答案 EM 正确；
- 至少 1 次有效 search；
- tool response 未严重截断；
- 总长度不超过训练上限；
- 工具调用 JSON 合法；
- 有明确 final answer。

丢弃样本：

- 教师答案错误；
- 工具调用格式不合法；
- 检索失败；
- 超过最大轮次；
- 无最终答案；
- 序列过长。

### 8.3 建议数据规模

第一版：

- NQ：10k 条 SFT；
- HotpotQA：10k 条 SFT；
- 验证集：各 500 条。

SFT 完成后导出合并权重：

```text
checkpoints/qwen2.5-1.5b-searchr1-sft/
```

这个权重作为 GRPO 初始模型。

---

## 9. 阶段 5：新版 verl GRPO 训练

第一版不要直接上 fully async trainer，而是先使用：

- `verl.trainer.main_ppo`
- `rollout.mode=async`
- `multi_turn.enable=True`
- `agent.default_agent_loop=tool_agent`

### 9.1 初版训练脚本

新增：

```text
recipe/search_r1_verl/scripts/train_grpo_qwen25_1p5b.sh
```

配置示例：

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$TRAIN_PARQUET \
  data.val_files=$VAL_PARQUET \
  data.return_raw_chat=True \
  data.train_batch_size=256 \
  data.max_prompt_length=4096 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=$SFT_MODEL_PATH \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/search_r1_verl/tools/search_tool_config.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  trainer.n_gpus_per_node=4 \
  trainer.val_before_train=False
```

注意：

- 如果 `vllm + tool_agent` 在当前 verl 版本中 parser 不稳定，可以切换到 `sglang` rollout；
- `format=hermes` 只应在 SFT 数据也按 Hermes tool-call 格式构造时使用；
- 先用 `rollout.n=2` 做 smoke test，再升到 5；
- Qwen2.5-1.5B 比 3B 省显存，可以逐步把 `gpu_memory_utilization` 从 0.35 调到 0.45，但不要一开始激进。

---

## 10. 阶段 6：奖励函数迁移

新增：

```text
recipe/search_r1_verl/rewards/qa_em_tool_reward.py
```

奖励设计：

| 奖励项           |       分数建议 | 说明          |
| ---------------- | -------------: | ------------- |
| EM 正确          |        `1.0` | 主奖励        |
| 最终答案格式正确 |       `+0.1` | 辅助          |
| 工具调用格式正确 | `+0.1 ~ 0.2` | 辅助          |
| 检索证据包含答案 | `+0.0 ~ 0.1` | 可做 ablation |
| 无答案 / 格式坏  | `0` 或小惩罚 | 不要压过 EM   |

重要原则：

- 格式错误、答案错误属于模型策略错误，应给低奖励或惩罚；
- 系统异常才应该 loss mask；
- 不要把所有坏轨迹都 mask，否则模型学不到格式约束。

---

## 11. 阶段 7：异常 trajectory 监控与过滤

新增：

```text
recipe/search_r1_verl/monitoring/trajectory_filter.py
recipe/search_r1_verl/monitoring/trajectory_metrics.py
```

### 11.1 异常类型划分

系统/环境异常，建议 mask：

```text
tool_timeout
tool_http_error
tool_parse_error
retriever_crash
empty_retrieval_due_to_system
tool_response_truncated
sequence_truncated
rollout_engine_error
max_tool_calls_exceeded
```

模型策略错误，默认不 mask，而是奖励惩罚：

```text
invalid_tool_arguments
invalid_answer_format
no_final_answer
unnecessary_search
wrong_answer
```

由于项目简介明确写了“超出调用上限”要过滤，因此 `max_tool_calls_exceeded` 可以放入 mask。

### 11.2 可配置过滤策略

```yaml
trajectory_filter:
  mask_on:
    - tool_timeout
    - tool_http_error
    - tool_response_truncated
    - sequence_truncated
    - max_tool_calls_exceeded
  penalize_only:
    - invalid_tool_arguments
    - no_final_answer
```

### 11.3 GRPO 下的关键点

对于 GRPO，不能只把异常 trajectory 的 reward 置 0。因为 GRPO 会在同一个 prompt 的 group 内计算 baseline，异常 trajectory 的 0 分会污染 group mean/std。

更合理的策略：

1. 按 `uid` 分组；
2. 只用 valid trajectory 计算 GRPO mean/std；
3. 异常 trajectory 的 `response_mask` 清零；
4. 如果某个 `uid` 下 valid trajectory 少于 2 条，则整组跳过或重采样；
5. 记录 dropped/masked group 数量。

### 11.4 必须记录的监控指标

wandb / console 指标：

```text
trajectory/valid_rate
trajectory/masked_rate
trajectory/tool_timeout_rate
trajectory/tool_response_truncated_rate
trajectory/sequence_truncated_rate
trajectory/max_tool_calls_exceeded_rate
trajectory/dropped_group_rate
tool/search_latency_ms_mean
tool/search_latency_ms_p95
tool/search_success_rate
rollout/turns_mean
rollout/search_calls_mean
```

---

## 12. 阶段 8：异步 rollout 与 fully async 优化

异步分两层，不要混在一起做。

### 12.1 必做：rollout async

这是 tool-agent 多轮交互的基础：

```bash
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.multi_turn.enable=True
```

这一步已经可以改善多轮工具调用的吞吐和长尾问题。

### 12.2 进阶：fully async trainer

最新版 verl 还有：

```bash
python3 -m verl.experimental.fully_async_policy.fully_async_main
```

它将 Trainer 和 Rollouter 解耦，适合更大规模训练。

但当前硬件是 4×L20，建议谨慎。

| 配置                | Rollout GPU | Train GPU | 建议           |
| ------------------- | ----------: | --------: | -------------- |
| colocate async      |        共享 |      共享 | 第一阶段使用   |
| fully async 4卡分离 |           1 |         3 | 可测试         |
| fully async 4卡分离 |           2 |         2 | 可能训练慢     |
| fully async 大规模  |         ≥4 |       ≥4 | 资源充足后使用 |

推荐路线：

1. 先跑普通 `main_ppo + rollout.mode=async`；
2. 指标稳定后迁到 `fully_async_main`；
3. 对比吞吐、EM、异常率；
4. 最后再写“异步 Rollout 提升训练吞吐”的实验结论。

---

## 13. 训练参数建议

基于已有 4×L20 经验，初始配置建议保守：

| 参数                          |                    建议值 |
| ----------------------------- | ------------------------: |
| `data.train_batch_size`     |                       256 |
| `rollout.n` / `n_agent`   | smoke test 用 2，正式用 5 |
| `max_prompt_length`         |                      4096 |
| `max_response_length`       |                      2048 |
| `max_tool_response_length`  |                      2048 |
| `max_turns`                 |                    2 或 3 |
| `gpu_memory_utilization`    |                 0.35 起步 |
| `ppo_micro_batch_size`      |            从 4 或 8 起步 |
| `log_prob_micro_batch_size` |                   32 起步 |
| `val_before_train`          |                     False |

已有经验应保留：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

每次重新训练前清理 Ray：

```bash
ray stop
```

---

## 14. 验收标准

### 14.1 Smoke test

条件：

- 10 条样本；
- `rollout.n=2`；
- `max_turns=2`；
- 本地检索服务正常；
- 模型至少能成功调用 search；
- console / wandb 能看到 tool metrics。

通过标准：

- 训练不报错；
- 检索服务无崩溃；
- 至少产生有效 tool call；
- 能计算 reward；
- 能输出 trajectory metrics。

### 14.2 小规模训练

条件：

- 500 条 train；
- 100 条 val；
- 50～100 step。

通过标准：

- 无 OOM；
- `trajectory/valid_rate > 70%`；
- `tool/search_success_rate > 95%`；
- reward 曲线不全为 0；
- 能保存 checkpoint。

### 14.3 正式训练

条件：

- NQ + HotpotQA 全量或主要子集；
- Qwen2.5-1.5B SFT checkpoint 起训；
- GRPO `rollout.n=5`；
- 检索服务常驻。

必须记录：

- EM；
- F1，如果需要；
- valid trajectory rate；
- search call rate；
- throughput samples/s；
- GPU 利用率；
- 检索延迟 p95；
- 异常 trajectory 占比。

---

## 15. 关于“得分提高 27%”的证明方式

不要只写“提高 27%”。必须固定对照条件：

1. 相同测试集；
2. 相同 EM 计算脚本；
3. 相同模型规模，或明确说明模型差异；
4. 相同检索语料；
5. 相同检索 topk；
6. 至少报告 3 个 seed 或方差；
7. 做异常 mask ablation。

推荐对照表：

| 实验                        | Tool Agent Loop | Async Rollout | 异常 Mask |       EM |
| --------------------------- | --------------- | ------------- | --------- | -------: |
| 原 Search-R1 复现           | 否              | 否            | 否        | baseline |
| 新 verl tool                | 是              | 否/基础 async | 否        |        x |
| 新 verl tool + async        | 是              | 是            | 否        |        x |
| 新 verl tool + async + mask | 是              | 是            | 是        |        x |

如果最终要写“提高 27%”，建议写成：

```text
在相同 NQ/HotpotQA 测试集、相同 EM 评价协议下，采用新版 verl Tool Agent Loop、异步 rollout 与异常 trajectory loss mask 后，模型 EM 相较旧版 Search-R1 复现基线提升 27%。
```

---

## 16. 可直接交给另一个 AI 的实现指令

请基于最新版 `verl-project/verl`，把 `T-zy/Search-R1` 中旧版 Search-R1 的检索增强推理训练迁移为原生 verl Tool Agent Loop 实现。不要继续修改旧仓库 vendor 的 `verl/`。

保留并增强现有 `E5 + FAISS IVF4096 + FastAPI` 检索服务；新增 `recipe/search_r1_verl/tools/search_tool.py`，基于 `verl.tools.base_tool.BaseTool` 实现 `search` 工具，通过 HTTP 调用本地 `/retrieve`，返回 `ToolResponse(text="<information>...</information>")`，并在 metrics 中记录 tool 成功率、失败类型、延迟、截断等信息。

新增 NQ/HotpotQA 数据转换脚本，输出新版 verl 可用 parquet，开启 `data.return_raw_chat=True`。使用教师模型生成 CoT + Tool Call + Answer 轨迹，在 LLaMA-Factory 对 Qwen2.5-1.5B 做 SFT 冷启动，导出合并权重后作为 GRPO 初始模型。

新增 custom reward，EM 为主奖励，格式约束为辅。新增 trajectory monitor/filter：追踪 `tool_timeout`、`tool_http_error`、`tool_response_truncated`、`sequence_truncated`、`max_tool_calls_exceeded` 等异常；在 GRPO advantage 计算前，对异常 trajectory 做 loss mask，并确保异常 trajectory 不参与同 uid 的 GRPO group baseline 统计。如果某个 uid 下有效 trajectory 少于 2 条，则整组跳过或重采样。

训练先使用普通 `verl.trainer.main_ppo` + `actor_rollout_ref.rollout.mode=async` + `multi_turn.enable=True` + `agent.default_agent_loop=tool_agent` 跑通；稳定后再迁移到 `verl.experimental.fully_async_policy.fully_async_main` 做吞吐优化。

最终输出：

- 训练脚本；
- 配置文件；
- SearchTool 实现；
- 检索服务增强；
- custom reward；
- 异常 trajectory filter；
- NQ/HotpotQA 数据转换脚本；
- SFT 数据构造脚本；
- 评测脚本；
- wandb 指标；
- ablation 结果。

---

## 17. 参考链接

- Search-R1 仓库：[https://github.com/T-zy/Search-R1](https://github.com/T-zy/Search-R1)
- verl multi-turn 文档：[https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
- verl Search Tool 集成文档：[https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
- verl fully async 文档：[https://verl.readthedocs.io/en/latest/advance/fully_async.html](https://verl.readthedocs.io/en/latest/advance/fully_async.html)
