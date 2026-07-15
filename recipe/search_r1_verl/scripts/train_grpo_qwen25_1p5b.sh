#!/bin/bash
set -euo pipefail
set -x
# ============================================================
# GRPO Training Script - Qwen2.5-1.5B with verl Tool Agent Loop
# 硬件: 4 × NVIDIA L20 (46GB)
# 基于 patched verl (verl -> verl_src symlink)
# ============================================================
# Usage:
#   1. Start retrieval service (separate terminal):
#      python recipe/search_r1_verl/retrieval_service/server.py \
#          --index_path /path/to/e5_IVF4096_Flat.index \
#          --corpus_path /path/to/wiki-18.jsonl
#
#   2. Run this script:
#      bash recipe/search_r1_verl/scripts/train_grpo_qwen25_1p5b.sh
# ============================================================

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=INFO

# ---- Paths (customize these!) ----
PROJECT_ROOT="/home/zytan/Search-R1_inforcement"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# SFT冷启动后的模型权重路径 (先用Base模型测试，后续替换为SFT权重)
SFT_MODEL_PATH="/media/public/RAIDStorageArray/workdir/zytan/Qwen2.5-1.5B-Instruct"

# 数据路径 (预先用 convert_nq_to_parquet.py 等脚本生成)
DATA_DIR="/media/public/RAIDStorageArray/workdir/zytan/searchr1_data/nq_hotpotqa_data"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
VAL_PARQUET="${DATA_DIR}/test.parquet"

# Tool配置
TOOL_CONFIG_PATH="${PROJECT_ROOT}/recipe/search_r1_verl/tools/search_tool_config.yaml"

# 实验名称
WAND_PROJECT="Search-R1-Verl"
EXPERIMENT_NAME="search-r1-grpo-qwen2.5-1.5b-verl-tool-agent"

# ---- Training Hyperparameters (第一阶段: 小batch验证) ----
TRAIN_BATCH_SIZE=64
ROLLOUT_N=4              # GRPO group size
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=1024
MAX_TOOL_RESPONSE_LENGTH=1024
MAX_TURNS=3
GPU_MEM_UTIL=0.45
PPO_MICRO_BATCH_SIZE=4
LOG_PROB_MICRO_BATCH_SIZE=16
LR=5e-7
KL_COEF=0.001
TOTAL_EPOCHS=15
TOTAL_STEPS=150

# ---- Run Training ----
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/verl_checkpoints"

cd "${PROJECT_ROOT}"

# ---- 强制启动检查：确认 patched verl 被加载 ----
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

signature = inspect.signature(core_algos.compute_grpo_outcome_advantage)
if "valid_sample_mask" not in signature.parameters:
    raise RuntimeError("Loaded GRPO implementation does not support valid_sample_mask.")
print("✅ All checks passed: patched verl is loaded.")
PY

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.return_raw_chat=True \
    data.truncation=error \
    data.filter_overlong_prompts=True \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=64 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.shuffle_train_dataloader=True \
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.08 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=false \
    reward.custom_reward_function.path="${PROJECT_ROOT}/recipe/search_r1_verl/rewards/qa_em_tool_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.default_local_dir="${PROJECT_ROOT}/verl_checkpoints/${EXPERIMENT_NAME}" \
    2>&1 | tee "${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}.log"
