#!/bin/bash
# ============================================================
# Evaluation Script for Search-R1 Verl Model
# ============================================================
# Usage:
#   bash recipe/search_r1_verl/scripts/evaluate.sh
#
# 评估训练好的模型在 NQ / HotpotQA 测试集上的 EM 和 F1
# ============================================================

set -x

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="/home/zytan/Search-R1_inforcement"
VERL_ROOT="/home/zytan/verl"

# 模型路径 (训练完成后生成的 checkpoint)
MODEL_PATH="${PROJECT_ROOT}/verl_checkpoints/search-r1-grpo-qwen2.5-1.5b-verl-tool-agent/actor"

# 测试数据路径
DATA_DIR="/media/public/RAIDStorageArray/workdir/zytan/searchr1_data/nq_hotpotqa_data"
TEST_PARQUET="${DATA_DIR}/test.parquet"

# 输出路径
OUTPUT_DIR="${PROJECT_ROOT}/evaluation_results"
mkdir -p "${OUTPUT_DIR}"

cd "${VERL_ROOT}" || { echo "Error: cannot cd to ${VERL_ROOT}"; exit 1; }

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_eval \
    data.test_files="${TEST_PARQUET}" \
    data.return_raw_chat=True \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/recipe/search_r1_verl/tools/search_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger=['console'] \
    trainer.project_name="Search-R1-Verl" \
    trainer.experiment_name="evaluation" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/evaluation.log"

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"
