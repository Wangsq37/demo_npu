#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CONFIG_PATH="${PROJECT_ROOT}/configs/multi_tower_din_taobao_local_npu.yaml"
MODEL_DIR="${PROJECT_ROOT}/experiments/multi_tower_din_taobao_local_npu"
EXPORT_DIR="${MODEL_DIR}/export"
PREDICT_DIR="${MODEL_DIR}/predict_result"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

cd "${PROJECT_ROOT}"

python -m easyrec_npu.train_eval \
  --config_path "${CONFIG_PATH}" \
  --model_dir "${MODEL_DIR}"

python -m easyrec_npu.eval \
  --config_path "${MODEL_DIR}/pipeline.yaml" \
  --checkpoint_path "${MODEL_DIR}/best.ckpt"

python -m easyrec_npu.export \
  --config_path "${MODEL_DIR}/pipeline.yaml" \
  --checkpoint_path "${MODEL_DIR}/best.ckpt" \
  --export_dir "${EXPORT_DIR}"

python -m easyrec_npu.predict \
  --config_path "${MODEL_DIR}/pipeline.yaml" \
  --scripted_model_path "${EXPORT_DIR}" \
  --predict_output_path "${PREDICT_DIR}" \
  --reserved_columns user_id,adgroup_id,clk
