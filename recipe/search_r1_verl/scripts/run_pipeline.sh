#!/bin/bash
# ============================================================
# 完整 Pipeline 执行脚本
# 按顺序执行: 数据准备 → SFT → GRPO训练 → 评估
# ============================================================
# 注意: 检索服务需要在单独的终端中先启动
# ============================================================

set -x

PROJECT_ROOT="/home/zytan/Search-R1_inforcement"
VERL_ROOT="/home/zytan/verl"

# ---- 0. 环境准备 ----
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 1. 数据转换 ----
echo "=== Step 1: Converting NQ data ==="
python "${PROJECT_ROOT}/recipe/search_r1_verl/data/convert_nq_to_parquet.py" \
    --output_dir "/media/public/RAIDStorageArray/workdir/zytan/searchr1_data/nq_hotpotqa_data" \
    --split train \
    --max_samples 10000

echo "=== Step 1b: Converting HotpotQA data ==="
python "${PROJECT_ROOT}/recipe/search_r1_verl/data/convert_hotpotqa_to_parquet.py" \
    --output_dir "/media/public/RAIDStorageArray/workdir/zytan/searchr1_data/nq_hotpotqa_data" \
    --split train \
    --max_samples 10000

# ---- 2. SFT 冷启动 (使用LLaMA-Factory或其他工具) ----
echo "=== Step 2: SFT Cold Start ==="
echo "SFT should be done separately using LLaMA-Factory with teacher trajectories."
echo "See: recipe/search_r1_verl/data/build_sft_data.py for teacher trajectory processing."

# ---- 3. Smoke Test ----
echo "=== Step 3: Smoke Test ==="
echo "Ensure retrieval service is running first (separate terminal):"
echo "  bash ${PROJECT_ROOT}/recipe/search_r1_verl/scripts/run_retrieval_service.sh"
echo ""
echo "Then run smoke test:"
echo "  bash ${PROJECT_ROOT}/recipe/search_r1_verl/scripts/train_grpo_smoke_test.sh"

# ---- 4. Full GRPO Training ----
echo ""
echo "=== Step 4: Full GRPO Training ==="
echo "After smoke test passes, run full training:"
echo "  bash ${PROJECT_ROOT}/recipe/search_r1_verl/scripts/train_grpo_qwen25_1p5b.sh"

# ---- 5. Evaluation ----
echo ""
echo "=== Step 5: Evaluation ==="
echo "After training, evaluate:"
echo "  bash ${PROJECT_ROOT}/recipe/search_r1_verl/scripts/evaluate.sh"

echo ""
echo "=== Pipeline overview complete ==="
