#!/bin/bash
# ============================================================
# Smoke Test GRPO Training Script
# 快速验证 tool loop、reward、mask 能否跑通
# 使用: 10条样本, rollout.n=2, max_turns=2
# ============================================================

set -x

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=INFO

PROJECT_ROOT="/home/zytan/Search-R1_inforcement"
VERL_ROOT="/home/zytan/verl"

SFT_MODEL_PATH="/media/public/RAIDStorageArray/workdir/zytan/Qwen2.5-1.5B-Instruct"
TOOL_CONFIG_PATH="${PROJECT_ROOT}/recipe/search_r1_verl/tools/search_tool_config.yaml"

# 使用仅包含10条样本的tiny数据
TRAIN_PARQUET="${PROJECT_ROOT}/data/smoke_test/train.parquet"
VAL_PARQUET="${PROJECT_ROOT}/data/smoke_test/test.parquet"

EXPERIMENT_NAME="search-r1-smoke-test-grpo-qwen2.5-1.5b"

cd "${VERL_ROOT}" || { echo "Error: cannot cd to ${VERL_ROOT}"; exit 1; }

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.return_raw_chat=True \
    data.truncation=error \
    data.filter_overlong_prompts=True \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.shuffle_train_dataloader=True \
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.08 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_penalty=kl \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.project_name="Search-R1-Verl-SmokeTest" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${PROJECT_ROOT}/verl_checkpoints/${EXPERIMENT_NAME}" \
    2>&1 | tee "${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}.log"
