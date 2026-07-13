#!/bin/bash
# ============================================================
# 启动 Enhanced Search-R1 Retrieval Service
# ============================================================
# Usage:
#   bash recipe/search_r1_verl/scripts/run_retrieval_service.sh
#
# 确保先激活 conda 环境:
#   conda activate retriever
# ============================================================

set -x

# ---- 自定义路径 (根据实际环境修改) ----
INDEX_PATH="/media/public/RAIDStorageArray/workdir/zytan/index/wiki-18/e5_IVF4096_Flat.index"
CORPUS_PATH="/media/public/RAIDStorageArray/workdir/zytan/data/retrieval-corpus/wiki-18.jsonl"
RETRIEVER_NAME="e5"
RETRIEVER_MODEL="intfloat/e5-base-v2"
PORT=8000
HOST="0.0.0.0"
TOP_K=3

# ---- 启动服务 ----
python /home/zytan/Search-R1_inforcement/recipe/search_r1_verl/retrieval_service/server.py \
    --index_path "${INDEX_PATH}" \
    --corpus_path "${CORPUS_PATH}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    --topk "${TOP_K}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --use_fp16
