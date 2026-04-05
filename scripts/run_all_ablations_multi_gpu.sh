#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run pN + mN ablation suites concurrently on different GPUs.
# Each suite still uses a single GPU per python process; see PARALLEL_EXPORTS in
# scripts/run_{pN,mN}_ablations.sh for within-suite parallel export jobs.
#
# Usage:
#   GPUS="0 1" bash scripts/run_all_ablations_multi_gpu.sh
#   GPUS="2 3" OUTPUT_ROOT_PN=... OUTPUT_ROOT_MN=... bash scripts/run_all_ablations_multi_gpu.sh

GPUS=${GPUS:-""}  # space-separated GPU ids, e.g. "0 1"

if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi --list-gpus | awk -F: '{print $1}' | awk '{print $2}' | head -n 2 | tr '\n' ' ')
    GPUS=${GPUS%% }  # trim trailing space
  fi
fi

read -r -a GPU_LIST <<< "${GPUS:-0}"

PN_GPU=${GPU_LIST[0]:-0}
MN_GPU=${GPU_LIST[1]:-$PN_GPU}

OUTPUT_ROOT_PN=${OUTPUT_ROOT_PN:-"output/pN_ablations_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_ROOT_MN=${OUTPUT_ROOT_MN:-"output/mN_ablations_$(date +%Y%m%d_%H%M%S)"}

echo "[multi] pN on GPU $PN_GPU -> $OUTPUT_ROOT_PN"
echo "[multi] mN on GPU $MN_GPU -> $OUTPUT_ROOT_MN"

(
  export CUDA_VISIBLE_DEVICES="$PN_GPU"
  export OUTPUT_ROOT="$OUTPUT_ROOT_PN"
  bash scripts/run_pN_ablations.sh
) &
PN_PID=$!

(
  export CUDA_VISIBLE_DEVICES="$MN_GPU"
  export OUTPUT_ROOT="$OUTPUT_ROOT_MN"
  bash scripts/run_mN_ablations.sh
) &
MN_PID=$!

wait "$PN_PID" "$MN_PID"
echo "[done] Both suites finished."
