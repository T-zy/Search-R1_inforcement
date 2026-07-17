#!/bin/bash
set -euo pipefail
set -x
# ============================================================
# Smoke Test GRPO Training Script
# 快速验证 tool loop、reward、mask 能否跑通
# 使用: 10条样本, rollout.n=2, max_turns=2
# ============================================================

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 与 vllm 0.19+ CuMemAllocator 不兼容
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VLLM_ATTENTION_BACKEND 让 vllm 自动选择（vllm 0.19+ 默认 FLASH_ATTN/FLASHINFER）
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=INFO

# 清除系统 CUDA 库路径，避免与 torch 自带的 CUDA 12.4 冲突
unset LD_LIBRARY_PATH

PROJECT_ROOT="/home/zytan/Search-R1_inforcement"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SEARCH_R1_DEBUG_PIPELINE=1

SFT_MODEL_PATH="/media/public/RAIDStorageArray/workdir/zytan/checkpoints/qwen2.5-1.5b-searchr1-sft-merged"
TOOL_CONFIG_PATH="${PROJECT_ROOT}/recipe/search_r1_verl/tools/search_tool_config.yaml"

# 使用仅包含10条样本的tiny数据
TRAIN_PARQUET="${PROJECT_ROOT}/data/smoke_test/train.parquet"
VAL_PARQUET="${PROJECT_ROOT}/data/smoke_test/test.parquet"

EXPERIMENT_NAME="search-r1-smoke-test-grpo-qwen2.5-1.5b"

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
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    +data.shuffle_train_dataloader=True \
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=false \
    reward.custom_reward_function.path="${PROJECT_ROOT}/recipe/search_r1_verl/rewards/qa_em_tool_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.project_name="Search-R1-Verl-SmokeTest" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${PROJECT_ROOT}/verl_checkpoints/${EXPERIMENT_NAME}" \
    2>&1 | tee "${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}.log"
