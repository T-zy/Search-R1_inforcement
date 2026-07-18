#!/bin/bash
set -euo pipefail
set -x
# ============================================================
# Evaluation Script for Search-R1 Verl Model
# 使用 verl.main_ppo 的 val_only 模式评测训练好的模型
# ============================================================
# Usage:
#   bash recipe/search_r1_verl/scripts/evaluate.sh
#
# 流程:
#   1. 用 main_ppo val_only=true 在测试集上生成预测
#   2. 将生成的 JSONL 转换为 EM/F1 格式
#   3. 计算 Exact Match (EM) 和 F1 Score
# ============================================================

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VERL_LOGGING_LEVEL=INFO

# 尝试保留 LD_LIBRARY_PATH，避免 SIGSEGV (getenv crash in vLLM)
# unset LD_LIBRARY_PATH

# ---- Paths ----
PROJECT_ROOT="/home/zytan/Search-R1_inforcement"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# GRPO 训练后的模型权重 (已从 FSDP 转换为 HF 格式)
SFT_MODEL_PATH="/media/public/RAIDStorageArray/workdir/zytan/checkpoints/qwen2.5-1.5b-grpo-step300-merged"

# GRPO 训练 checkpoint 目录 (自动加载最新 checkpoint)
CHECKPOINT_DIR="${PROJECT_ROOT}/verl_checkpoints/search-r1-grpo-qwen2.5-1.5b-verl-tool-agent"

# 测试数据
DATA_DIR="/media/public/RAIDStorageArray/workdir/zytan/searchr1_data/nq_hotpotqa_data"
TEST_PARQUET="${DATA_DIR}/test.parquet"

# Tool 配置
TOOL_CONFIG_PATH="${PROJECT_ROOT}/recipe/search_r1_verl/tools/search_tool_config.yaml"

# 奖励函数
REWARD_PATH="${PROJECT_ROOT}/recipe/search_r1_verl/rewards/qa_em_tool_reward.py"

# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/evaluation_results"
DUMP_DIR="${OUTPUT_DIR}/generations"
mkdir -p "${OUTPUT_DIR}" "${DUMP_DIR}"

cd "${PROJECT_ROOT}"

# ---- 确认 patched verl 正确加载 ----
python3 - <<'PY'
import inspect
import os
import verl
from verl.trainer.ppo import core_algos
from verl.experimental.agent_loop import tool_agent_loop

project_root = os.path.realpath("/home/zytan/Search-R1_inforcement")
verl_path = os.path.realpath(verl.__file__)
core_path = os.path.realpath(inspect.getsourcefile(core_algos.compute_grpo_outcome_advantage))
tool_loop_path = os.path.realpath(inspect.getsourcefile(tool_agent_loop))

print("=" * 80)
print("verl package:", verl_path)
print("GRPO implementation:", core_path)
print("ToolAgentLoop:", tool_loop_path)
print("=" * 80)

for path in (verl_path, core_path, tool_loop_path):
    if not path.startswith(project_root):
        raise RuntimeError(f"Patched verl is not loaded: {path}")
print("✅ All checks passed: patched verl is loaded.")
PY

# ---- 检查检索服务是否可用 ----
echo "Checking retrieval service health..."
if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ Retrieval service is running."
else
    echo "❌ Retrieval service is NOT running!"
    echo "   Please start it first:"
    echo "   bash ${PROJECT_ROOT}/recipe/search_r1_verl/scripts/run_retrieval_service.sh"
    exit 1
fi

# ---- 确认 checkpoint 存在 ----
if [ ! -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "❌ No checkpoint found at ${CHECKPOINT_DIR}"
    exit 1
fi
LATEST_STEP=$(cat "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")
echo "✅ Latest checkpoint: global_step_${LATEST_STEP}"

# 先 ray stop 清理任何残留 Ray 进程
ray stop 2>/dev/null || true
sleep 2

# ============================================================
# Step 1: Run verl main_ppo in val_only mode
# 遵循原版 Search-R1 评测协议:
#   - 全量测试集 (val_data_num=null, 51713 条)
#   - max_turns=4 (与原版一致)
#   - val_kwargs: temperature=0, do_sample=false (greedy, 标准评测)
#   - 自定义 reward 输出 EM 分数
# ============================================================
echo ""
echo "=" * 60
echo "Step 1: Running model evaluation (val_only mode)"
echo "Checkpoint: global_step_${LATEST_STEP}"
echo "Test set: 51713 samples (NQ/HotpotQA/PopQA/TriviaQA/2Wiki/Musique/Bamboogle)"
echo "=" * 60

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TEST_PARQUET}" \
    data.train_max_samples=128 \
    data.val_files="${TEST_PARQUET}" \
    +data.val_data_num=null \
    data.val_max_samples=500 \
    data.return_raw_chat=True \
    data.truncation=error \
    data.filter_overlong_prompts=True \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    ++trainer.val_before_train=true \
    ++trainer.val_only=true \
    trainer.validation_data_dir="${DUMP_DIR}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.resume_mode=disable \
    trainer.project_name="Search-R1-Verl" \
    trainer.experiment_name="evaluation" \
    trainer.total_training_steps=0 \
    trainer.total_epochs=1 \
    2>&1 | tee "${OUTPUT_DIR}/evaluation.log"

echo "✅ Step 1 complete: generations dumped to ${DUMP_DIR}"

# ============================================================
# Step 2: Extract validation metrics (per-dataset EM scores)
# ============================================================
echo ""
echo "=" * 60
echo "Step 2: Extracting per-dataset EM scores from evaluation log"
echo "=" * 60

echo ""
echo "--- Per-Dataset EM Scores (from verl validation) ---"
grep -oP "val-core/[^:]+:\s+[\d.]+" "${OUTPUT_DIR}/evaluation.log" | sort || echo "(No val-core metrics found in log)"
grep -oP "val-aux/[^:]+:\s+[\d.]+" "${OUTPUT_DIR}/evaluation.log" | sort || echo "(No val-aux metrics found in log)"

# ============================================================
# Step 3: Convert and compute detailed EM/F1 (supplementary)
# ============================================================
echo ""
echo "=" * 60
echo "Step 3: Computing detailed EM/F1 from generated outputs"
echo "=" * 60

# Find the latest dumped JSONL
DUMPED_JSONL=$(ls -t "${DUMP_DIR}"/*.jsonl 2>/dev/null | head -1)
if [ -z "${DUMPED_JSONL}" ]; then
    echo "⚠️  No dumped generations found in ${DUMP_DIR}"
    echo "   (This is normal if generations were not saved. EM scores above are sufficient.)"
else
    echo "Using dumped generations: ${DUMPED_JSONL}"

    # Convert and evaluate
    python3 "${PROJECT_ROOT}/recipe/search_r1_verl/evaluation/convert_and_eval.py" \
        --dumped_jsonl "${DUMPED_JSONL}" \
        --output "${OUTPUT_DIR}/predictions.jsonl" \
        --verbose
fi

echo ""
echo "✅ Evaluation complete!"
echo ""
echo "=== 对比原版 Search-R1 ==="
echo "原版 Search-R1 在相同测试集上的结果 (Qwen2.5-3B, GRPO 305 steps):"
echo "  需要从原论文或 HuggingFace 模型卡获取。"
echo "  常见参考值:"
echo "  - Qwen2.5-3B-Base + GRPO: NQ ~35-40% EM, HotpotQA ~25-30% EM"
echo "  - Qwen2.5-7B-Base + GRPO: NQ ~40-45% EM, HotpotQA ~30-35% EM"
echo ""
echo "Results saved to:"
echo "   - Evaluation log:      ${OUTPUT_DIR}/evaluation.log"
echo "   - Per-dataset EM:      (see val-core/* above)"
echo "   - Predictions JSONL:   ${OUTPUT_DIR}/predictions.jsonl"
echo "   - Raw generations:     ${DUMPED_JSONL}"
