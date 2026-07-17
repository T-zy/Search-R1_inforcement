# Search-R1 SFT 数据处理逻辑审查报告

## 1. 审查结论

本次检查基于仓库：

```text
https://github.com/T-zy/Search-R1_inforcement.git
```

主要对照以下内容：

- `Search-R1_verl_实验文档.md` 的 **3.8 Tool-Agent SFT 冷启动**
- `recipe/search_r1_verl/data/` 下的数据处理代码
- verl 的 Hermes Tool Parser
- LLaMA-Factory 的 ShareGPT 数据解析逻辑
- GRPO 阶段的 Tool Agent Loop 配置

当前整体思路是合理的：

```text
教师模型生成 Search-R1 轨迹
        ↓
旧版 <search> 标签转换为 Hermes <tool_call>
        ↓
过滤答案正确、格式合法的轨迹
        ↓
使用 LLaMA-Factory 进行 SFT
        ↓
将 SFT 模型用于 verl GRPO
```

但是，当前代码仍存在多个会导致：

- 数据被覆盖；
- 数据集为空；
- LLaMA-Factory 跳过样本；
- SFT 格式和 GRPO 格式不一致；
- 长轨迹未被正确过滤；

的问题。

因此，**当前版本不建议直接生成 2 万条正式 SFT 数据**。应先修复下面列出的 P0 问题，再用 50～200 条数据跑通完整流程。

---

# 2. 必须修复的 P0 问题

## 2.1 NQ 和 HotpotQA 的 Parquet 文件会互相覆盖

当前两个转换脚本都使用：

```python
output_path = os.path.join(args.output_dir, f"{args.split}.parquet")
```

因此：

```bash
python convert_nq_to_parquet.py \
    --output_dir /path/to/nq_hotpotqa_data \
    --split train

python convert_hotpotqa_to_parquet.py \
    --output_dir /path/to/nq_hotpotqa_data \
    --split train
```

最终都会写入：

```text
/path/to/nq_hotpotqa_data/train.parquet
```

后执行的 HotpotQA 会覆盖前面的 NQ。

但教师轨迹生成逻辑又假设 `train.parquet` 同时包含：

```python
df[df["data_source"] == "nq"]
df[df["data_source"] == "hotpotqa"]
```

这会导致其中一个数据源不存在，后续执行：

```python
.sample(n=args.nq_samples)
```

时直接报错。

## 推荐修改

分别保存：

```text
nq_train.parquet
hotpotqa_train.parquet
```

例如将 NQ 修改为：

```python
output_path = os.path.join(
    args.output_dir,
    f"nq_{args.split}.parquet"
)
```

将 HotpotQA 修改为：

```python
output_path = os.path.join(
    args.output_dir,
    f"hotpotqa_{args.split}.parquet"
)
```

之后再显式合并：

```python
import pandas as pd

nq_df = pd.read_parquet("nq_train.parquet")
hotpotqa_df = pd.read_parquet("hotpotqa_train.parquet")

train_df = pd.concat(
    [nq_df, hotpotqa_df],
    ignore_index=True
)

train_df.to_parquet(
    "train.parquet",
    index=False
)
```

验证集和测试集也应采用相同方式，避免文件名冲突。

---

## 2.2 文档中的两个核心脚本没有实际提交到仓库

实验文档 3.8 中使用了：

```text
recipe/search_r1_verl/data/generate_teacher_trajectories.py
recipe/search_r1_verl/data/convert_to_hermes_sft.py
```

但当前仓库中实际存在的主要 SFT 数据脚本只有：

```text
recipe/search_r1_verl/data/build_sft_data.py
```

也就是说，文档描述的完整流水线：

```text
Parquet
  ↓
generate_teacher_trajectories.py
  ↓
teacher_trajectories.jsonl
  ↓
convert_to_hermes_sft.py
  ↓
hermes_trajectories.jsonl
  ↓
build_sft_data.py
  ↓
sft_data.jsonl
```

目前并没有在仓库中完整落地。

## 推荐修改

将以下两个脚本正式加入仓库：

```text
recipe/search_r1_verl/data/generate_teacher_trajectories.py
recipe/search_r1_verl/data/convert_to_hermes_sft.py
```

并增加对应测试：

```text
tests/test_generate_teacher_trajectories.py
tests/test_convert_to_hermes_sft.py
tests/test_build_sft_data.py
```

---

## 2.3 教师轨迹脚本读取 question 的逻辑错误

文档中的教师生成代码使用：

```python
question = row.get("question", "") or row.get("prompt", "")
```

但转换后的 Parquet 结构实际为：

```python
{
    "data_source": "nq",
    "prompt": [
        {
            "role": "system",
            "content": "..."
        },
        {
            "role": "user",
            "content": "..."
        }
    ],
    "reward_model": {
        "ground_truth": {
            "target": [...]
        }
    }
}
```

其中没有独立的 `question` 字段。

因此：

```python
row.get("prompt", "")
```

得到的不是字符串，而是消息列表。

之后若使用：

```python
f"Question: {question}"
```

模型实际看到的可能是：

```text
Question: [
    {'role': 'system', 'content': '...'},
    {'role': 'user', 'content': '...'}
]
```

这不是正常的用户问题。

## 推荐修改

新增统一提取函数：

```python
def extract_prompt_messages(row):
    prompt = row["prompt"]

    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()

    system_prompt = ""
    question = ""

    for message in prompt:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system":
            system_prompt = content
        elif role == "user":
            question = content

    if not question:
        raise ValueError(
            "No user question found in prompt."
        )

    return system_prompt, question
```

调用：

```python
system_prompt, question = extract_prompt_messages(row)
```

同时建议保留原始 system prompt，不要再使用另一套 `build_prompt()` 重新构造完全不同的训练指令。

---

## 2.4 Hermes 转换后，`build_sft_data.py` 会把所有数据过滤掉

3.8 的处理流程先把：

```xml
<search>query</search>
```

转换为：

```xml
<tool_call>
{"name": "search", "arguments": {"query": "query"}}
</tool_call>
```

但是当前 `build_sft_data.py` 的工具调用计数仍是：

```python
count += len(
    re.findall(r"<search>", content)
)
```

随后执行：

```python
if num_tool_calls < 1:
    discard_reasons["no_search"] += 1
    continue
```

这意味着：

```text
convert_to_hermes_sft.py
        ↓
所有 <search> 已经被替换
        ↓
build_sft_data.py 只查找 <search>
        ↓
num_tool_calls 永远为 0
        ↓
全部样本被过滤为 no_search
```

这是当前最严重的问题之一。

## 推荐修改

统一基于 Hermes 格式解析工具调用：

```python
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
```

实现严格解析：

```python
def parse_tool_calls(content: str) -> list[dict]:
    calls = []

    for raw_call in TOOL_CALL_PATTERN.findall(content):
        try:
            payload = json.loads(raw_call)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "tool_call is not valid JSON"
            ) from exc

        if payload.get("name") != "search":
            raise ValueError(
                "Tool name must be 'search'."
            )

        arguments = payload.get("arguments")

        if not isinstance(arguments, dict):
            raise ValueError(
                "Tool arguments must be a JSON object."
            )

        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Search query must be a non-empty string."
            )

        topk = arguments.get("topk", 3)

        if not isinstance(topk, int):
            raise ValueError(
                "topk must be an integer."
            )

        if not 1 <= topk <= 5:
            raise ValueError(
                "topk must be between 1 and 5."
            )

        calls.append(payload)

    return calls
```

计数逻辑修改为：

```python
def count_tool_calls(
    trajectory: list[dict]
) -> int:
    count = 0

    for turn in trajectory:
        if turn.get("role") != "assistant":
            continue

        content = turn.get("content", "")
        count += len(parse_tool_calls(content))

    return count
```

---

## 2.5 当前的 `is_tool_call_valid()` 实际上永远返回 True

当前逻辑大致为：

```python
for match in matches:
    try:
        json.loads(match.strip())
    except json.JSONDecodeError:
        pass

return True
```

即使工具调用 JSON 非法，也只是 `pass`，最后仍返回 `True`。

而且该函数只检查：

```xml
<search>...</search>
```

不检查转换后的：

```xml
<tool_call>...</tool_call>
```

所以以下非法内容都会通过过滤：

```xml
<tool_call>{invalid json}</tool_call>
```

```json
{
    "name": "wrong_tool",
    "arguments": {
        "query": "abc"
    }
}
```

```json
{
    "name": "search",
    "arguments": {}
}
```

```json
{
    "name": "search",
    "arguments": "not an object"
}
```

## 推荐修改

删除旧版：

```python
is_tool_call_valid()
```

统一使用：

```python
parse_tool_calls()
```

例如：

```python
def validate_tool_calls(
    trajectory: list[dict]
) -> tuple[bool, str]:
    try:
        total_calls = 0

        for turn in trajectory:
            if turn.get("role") != "assistant":
                continue

            content = turn.get("content", "")
            total_calls += len(
                parse_tool_calls(content)
            )

        if total_calls == 0:
            return False, "no_tool_call"

        return True, ""

    except ValueError as exc:
        return False, str(exc)
```

---

## 2.6 LLaMA-Factory 的角色配置不完整

文档中推荐的 `dataset_info.json` 配置只声明：

```json
"tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
}
```

但轨迹中实际还存在：

```text
system
tool
```

LLaMA-Factory 对 ShareGPT 数据会严格检查角色交替：

```text
user / observation
assistant / function
user / observation
assistant / function
```

你的数据去掉 system 后通常是：

```text
user
assistant
tool
assistant
```

其中：

- `tool` 应映射为 `observation`
- assistant 的工具调用可以保持 assistant 文本，也可以建模为 function_call
- `system` 应显式声明为 system_tag

如果没有配置 `observation_tag` 和 `system_tag`，样本可能被识别为异常数据并跳过。

## 推荐配置

建议最终输出字段使用：

```text
messages
tools
```

对应 `dataset_info.json`：

```json
{
  "search_r1_sft": {
    "file_name": "search_r1_sft.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "tools": "tools"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "observation_tag": "tool",
      "function_tag": "function_call",
      "system_tag": "system"
    }
  }
}
```

最终消息顺序应是：

```text
system
user
assistant
tool
assistant
tool
assistant
```

去除 system 后：

```text
user → assistant → tool → assistant → tool → assistant
```

满足：

```text
user / observation
assistant / function
```

交替要求。

---

# 3. 当前输出缺少工具 Schema

当前 `build_sft_data.py` 最终输出类似：

```python
{
    "question": question,
    "trajectory": trajectory,
    "answer": golden_answers,
    "num_tool_calls": num_tool_calls,
}
```

其中没有 `tools` 字段。

但 GRPO 阶段，verl 的 ToolAgentLoop 会把工具 schema 传入：

```python
tokenizer.apply_chat_template(
    messages,
    tools=tool_schemas,
    ...
)
```

如果 SFT 阶段没有给模型提供同样的工具定义，模型只会看到：

```xml
<tool_call>
{"name": "search", "arguments": {...}}
</tool_call>
```

却不知道：

- `search` 工具是什么；
- `query` 参数是什么；
- `topk` 范围是什么；
- 工具返回什么格式；
- 什么情况下应调用工具。

这会造成 SFT 和 GRPO 阶段的提示分布不一致。

## 推荐工具 Schema

```python
SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia passages relevant "
                "to a factual question. Returns the "
                "top-k passages formatted as "
                "<information>...</information>."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query string."
                        )
                    },
                    "topk": {
                        "type": "integer",
                        "description": (
                            "Number of passages to "
                            "retrieve, from 1 to 5."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

最终输出：

```python
output_record = {
    "messages": trajectory,
    "tools": json.dumps(
        SEARCH_TOOL_SCHEMA,
        ensure_ascii=False
    ),
    "metadata": {
        "question": question,
        "answers": golden_answers,
        "num_tool_calls": num_tool_calls,
        "data_source": example.get(
            "data_source",
            ""
        ),
    },
}
```

建议将 `tools` 保存为 JSON 字符串，以符合 LLaMA-Factory 的工具数据字段使用方式。

---

# 4. `max_length` 参数当前没有实际作用

当前脚本虽然声明：

```python
parser.add_argument(
    "--max_length",
    type=int,
    default=4096
)
```

但后续没有：

- 加载 tokenizer；
- 调用 chat template；
- 统计 token；
- 根据 token 数过滤；
- 对过长数据进行截断或丢弃。

因此：

```bash
--max_length 4096
```

目前只是一个没有实际作用的命令行参数。

搜索结果一次可能返回数千字符，多轮搜索后很容易超过 4096 token。

## 推荐修改

新增参数：

```python
parser.add_argument(
    "--tokenizer_path",
    type=str,
    required=True
)
```

加载 tokenizer：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_path,
    trust_remote_code=True,
)
```

对完整消息和工具 schema 统计真实 token 数：

```python
token_ids = tokenizer.apply_chat_template(
    trajectory,
    tools=SEARCH_TOOL_SCHEMA,
    tokenize=True,
    add_generation_prompt=False,
)

token_length = len(token_ids)

if token_length > args.max_length:
    discard_reasons["too_long"] = (
        discard_reasons.get(
            "too_long",
            0
        ) + 1
    )
    continue
```

不能简单使用：

```python
len(json.dumps(trajectory))
```

或字符数量代替 token 数量。

---

# 5. 教师模型生成方式需要调整

文档中的教师轨迹生成方式是：

```python
inputs = tokenizer(
    current_text,
    return_tensors="pt"
).to(device)
```

这种方式直接编码裸文本，没有使用 Qwen2.5 的 chat template。

Qwen2.5 Instruct 模型通常依赖类似：

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
```

的对话模板。

如果直接编码裸文本，可能导致：

- 教师模型输出质量下降；
- Search-R1 标签格式不稳定；
- system prompt 失效；
- SFT 数据分布与后续推理分布不一致。

## 推荐生成方式

第一轮：

```python
messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": question
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
```

生成 assistant 回复后：

```python
messages.append(
    {
        "role": "assistant",
        "content": response
    }
)
```

如果模型产生：

```xml
<search>...</search>
```

执行检索后增加：

```python
messages.append(
    {
        "role": "tool",
        "content": (
            "<information>"
            + search_results
            + "</information>"
        )
    }
)
```

下一轮继续通过：

```python
tokenizer.apply_chat_template(...)
```

重建输入，不要直接使用：

```python
current_text += response
current_text += info_block
```

维护裸字符串状态。

---

# 6. 过滤逻辑缺少完整的轨迹状态检查

当前过滤主要检查：

- 是否有 search；
- 是否有 answer；
- 最终答案是否 EM 正确；
- assistant turn 是否超限；
- tool call 数是否超限。

但缺少以下结构校验：

```text
assistant 输出 tool_call
        ↓
下一条必须是 tool
        ↓
tool 返回必须成功
        ↓
之后必须有 assistant
        ↓
最终 assistant 必须输出完整 answer 标签
```

## 推荐实现角色顺序校验

```python
def validate_role_sequence(
    messages: list[dict]
) -> tuple[bool, str]:
    valid_roles = {
        "system",
        "user",
        "assistant",
        "tool"
    }

    for message in messages:
        role = message.get("role")

        if role not in valid_roles:
            return False, "unknown_role"

    non_system = [
        message
        for message in messages
        if message.get("role") != "system"
    ]

    if not non_system:
        return False, "empty_messages"

    if non_system[0].get("role") != "user":
        return False, "missing_user"

    for index, message in enumerate(non_system):
        role = message.get("role")
        content = message.get("content", "")

        if role == "tool":
            if index == 0:
                return False, "orphan_tool_response"

            previous = non_system[index - 1]

            if previous.get("role") != "assistant":
                return False, "orphan_tool_response"

            if "<tool_call>" not in previous.get(
                "content",
                ""
            ):
                return False, "tool_without_tool_call"

        if role == "assistant":
            if "<tool_call>" not in content:
                continue

            if index + 1 >= len(non_system):
                return False, "missing_tool_response"

            next_message = non_system[index + 1]

            if next_message.get("role") != "tool":
                return False, "missing_tool_response"

    return True, ""
```

---

## 6.1 过滤失败的搜索结果

应过滤包含以下内容的工具响应：

```text
Search failed:
timed out
HTTP error
Unexpected error
Query is empty
```

示例：

```python
SEARCH_FAILURE_MARKERS = (
    "Search failed:",
    "timed out",
    "HTTP error",
    "Unexpected error",
    "Query is empty",
)
```

```python
def has_failed_tool_response(
    trajectory: list[dict]
) -> bool:
    for turn in trajectory:
        if turn.get("role") != "tool":
            continue

        content = turn.get("content", "")

        if any(
            marker in content
            for marker in SEARCH_FAILURE_MARKERS
        ):
            return True

        if (
            "<information>" not in content
            or "</information>" not in content
        ):
            return True

        inner = re.search(
            r"<information>(.*?)</information>",
            content,
            flags=re.DOTALL
        )

        if not inner or not inner.group(1).strip():
            return True

    return False
```

---

## 6.2 最终答案标签应完整

当前逻辑只检查：

```python
"<answer>" in content
```

这会允许：

```text
<answer>Beijing
```

这样的不完整输出通过。

应严格检查：

```python
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.DOTALL
)
```

```python
def extract_final_answer(
    trajectory: list[dict]
) -> str:
    for turn in reversed(trajectory):
        if turn.get("role") != "assistant":
            continue

        content = turn.get("content", "")
        matches = ANSWER_PATTERN.findall(content)

        if matches:
            return matches[-1].strip()

    return ""
```

---

# 7. SFT 与 GRPO 的最大轮数不一致

文档中的 SFT 数据过滤命令使用：

```bash
--max_turns 4
--max_tool_calls 10
```

但是正式 GRPO 配置中：

```yaml
max_assistant_turns: 3
max_parallel_calls: 1
```

一次只允许一个工具调用时，一个典型双跳轨迹是：

```text
assistant：search 1
tool：information 1
assistant：search 2
tool：information 2
assistant：final answer
```

即：

```text
assistant turns = 3
tool calls = 2
```

如果 SFT 中保留：

```text
4 个 assistant turn
10 个 tool call
```

模型可能学习生成更长的搜索轨迹，但 GRPO 第 3 个 assistant turn 后就被强制终止，导致还没来得及输出最终答案。

## 推荐统一参数

```bash
--max_turns 3
--max_tool_calls 2
```

如果未来修改 GRPO：

```yaml
max_assistant_turns: 4
```

再同步调整 SFT 过滤参数。

SFT 和 GRPO 的以下参数应保持一致：

```text
最大 assistant turn
最大 tool call 数
工具 schema
system prompt
最终 answer 标签格式
tool response 格式
最大上下文长度
```

---

# 8. 推荐的最终 SFT 数据格式

建议每条数据采用：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a retrieval-augmented question answering agent. Use the search tool when external evidence is needed. Output the final answer in <answer>...</answer>."
    },
    {
      "role": "user",
      "content": "Which city is the birthplace of the author of The Old Man and the Sea?"
    },
    {
      "role": "assistant",
      "content": "<think>I need to identify the author first.</think>\n<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"The Old Man and the Sea author\"}}\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "<information>Doc 1(Title: The Old Man and the Sea): The novel was written by Ernest Hemingway.</information>"
    },
    {
      "role": "assistant",
      "content": "<think>The author is Ernest Hemingway. I need his birthplace.</think>\n<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"Ernest Hemingway birthplace city\"}}\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "<information>Doc 1(Title: Ernest Hemingway): Hemingway was born in Oak Park, Illinois.</information>"
    },
    {
      "role": "assistant",
      "content": "<think>The birthplace is Oak Park.</think>\n<answer>Oak Park</answer>"
    }
  ],
  "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"search\",\"description\":\"Search Wikipedia passages relevant to a factual question.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"topk\":{\"type\":\"integer\"}},\"required\":[\"query\"]}}}]",
  "metadata": {
    "data_source": "hotpotqa",
    "golden_answers": [
      "Oak Park"
    ],
    "num_tool_calls": 2,
    "em_correct": true,
    "retrieval_success": true
  }
}
```

其中：

- `messages` 用于 LLaMA-Factory SFT；
- `tools` 用于注入 search 工具定义；
- `metadata` 不参与训练，仅用于审计和统计。

---

# 9. 推荐的正确数据处理顺序

## 阶段一：构造原始数据

```text
1. NQ 转为 nq_train.parquet
2. HotpotQA 转为 hotpotqa_train.parquet
3. 合并为 train.parquet
4. 分别构造 validation/test
```

## 阶段二：生成教师轨迹

```text
5. 从 prompt 中提取 system 和 user
6. 使用 Qwen chat template
7. 教师模型生成 <search>...</search>
8. 调用本地检索服务
9. 将结果写入 <information>...</information>
10. 继续生成直到 <answer>...</answer>
```

## 阶段三：协议转换

```text
11. 将 <search>query</search>
    转为 Hermes <tool_call>
12. 保留 tool response
13. 保留 final answer
```

## 阶段四：严格过滤

```text
14. 检查角色顺序
15. 检查 Hermes JSON
16. 检查工具名为 search
17. 检查 query 非空
18. 检查 topk 范围
19. 检查 tool response 成功
20. 检查 answer 标签完整
21. 检查 EM 正确
22. 检查 assistant turn <= 3
23. 检查 tool calls <= 2
24. 使用 tokenizer 检查 token <= 4096
```

## 阶段五：输出 LLaMA-Factory 数据

```text
25. 输出 messages
26. 输出 tools
27. 输出 metadata
28. 注册 dataset_info.json
```

## 阶段六：小规模验收

```text
29. 先生成 50 条
30. 检查 build_sft_data 的保留率
31. 使用 LLaMA-Factory preview_dataset
32. 人工检查 10～20 条
33. 训练一个几十步的 SFT smoke test
34. 测试模型能否生成合法 tool_call
```

## 阶段七：扩大数据规模

```text
35. 生成约 2,000 条流程验证数据
36. 训练小规模 SFT
37. 验证 Tool Agent Loop
38. 最终扩展到约 20,000 条正式 SFT 数据
```

---

# 10. 建议增加的数据质量统计

处理结束后至少输出以下统计：

```python
stats = {
    "total": 0,
    "kept": 0,
    "no_tool_call": 0,
    "invalid_tool_json": 0,
    "invalid_tool_name": 0,
    "empty_query": 0,
    "failed_retrieval": 0,
    "missing_tool_response": 0,
    "invalid_role_sequence": 0,
    "missing_final_answer": 0,
    "wrong_answer": 0,
    "too_many_turns": 0,
    "too_many_tool_calls": 0,
    "too_long": 0,
}
```

还建议输出：

```text
NQ 原始数量
HotpotQA 原始数量
NQ 保留数量
HotpotQA 保留数量
平均 token 长度
P50 token 长度
P90 token 长度
P95 token 长度
最大 token 长度
平均工具调用次数
1 次搜索占比
2 次搜索占比
答案 EM 通过率
检索失败率
```

防止过滤后数据严重偏向某一个数据集。

---

# 11. SFT 训练配置中的额外问题

文档中训练使用：

```bash
--finetuning_type full
```

但导出阶段又使用：

```bash
--adapter_name_or_path ...
--finetuning_type full
```

两者逻辑不一致。

## 全参数训练

如果使用：

```bash
--finetuning_type full
```

训练输出目录本身就是完整模型，一般不需要进行 LoRA 合并。

## LoRA 训练

如果使用：

```bash
--finetuning_type lora
```

训练后才需要：

```bash
llamafactory-cli export \
    --model_name_or_path BASE_MODEL \
    --adapter_name_or_path LORA_PATH \
    --finetuning_type lora \
    --export_dir MERGED_MODEL
```

因此应在训练前明确选择：

```text
全参数 SFT
或
LoRA SFT
```

不要混用两套导出逻辑。

对于 Qwen2.5-1.5B 和 4×L20，显存足以考虑全参数 SFT；如果希望更快迭代和节省存储，也可以使用 LoRA。

---

# 12. 最终问题优先级

## P0：不修复不能开始正式处理

1. NQ 和 HotpotQA Parquet 文件互相覆盖。
2. `generate_teacher_trajectories.py` 未实际加入仓库。
3. `convert_to_hermes_sft.py` 未实际加入仓库。
4. 教师脚本错误读取 `prompt` 为 question。
5. Hermes 转换后仍统计 `<search>`。
6. `is_tool_call_valid()` 永远返回 True。
7. LLaMA-Factory 缺少 system/tool 角色映射。
8. 最终 SFT 数据缺少 tools schema。
9. `max_length` 没有实际生效。

## P1：建议正式训练前修复

1. 教师生成未使用 Qwen chat template。
2. 未严格检查角色顺序。
3. 未过滤失败检索结果。
4. 未严格检查完整 `<answer>...</answer>`。
5. SFT 最大轮数与 GRPO 不一致。
6. 缺少完整的数据质量统计。
7. 缺少小规模 SFT 数据单元测试。

## P2：后续优化

1. NQ 和 HotpotQA 分层采样。
2. 控制单跳和多跳问题比例。
3. 对搜索 query 去重。
4. 限制重复搜索。
5. 对 teacher trajectory 做 F1 或 alias match，而不只用严格 EM。
6. 增加无需搜索即可直接回答的少量样本。
7. 增加检索结果支持答案但教师答错的诊断统计。

---

# 13. 最终结论

当前的数据处理方案在方向上是正确的：

```text
Search-R1 教师轨迹
→ Hermes 工具调用格式
→ 高质量过滤
→ Tool-Agent SFT
→ verl GRPO
```

但是当前实现还没有真正形成可用的闭环。

在正式开始 SFT 数据生成前，至少应完成以下五项核心修复：

```text
1. 修复 NQ/HotpotQA 文件覆盖
2. 修复 prompt/question 提取
3. 修复 Hermes 工具调用计数和校验
4. 修复 LLaMA-Factory 角色映射
5. 在 SFT 数据中加入 search 工具 schema
```

建议先生成：

```text
50 条
```

进行人工和程序验收，再生成：

```text
2,000 条
```

跑通 SFT smoke test，最后扩展到：

```text
约 20,000 条
```

正式训练数据。
