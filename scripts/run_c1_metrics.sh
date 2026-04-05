#!/usr/bin/env bash
set -euo pipefail

# Runs the unified baseline evaluation to produce the requested C1 metrics
# (AUPRC, AUROC, Precision@K / Recall@K, coverage, per-TF Top-k stats + p-values).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-output/evaluate/c1_metrics}"
K_VALUES="${K_VALUES:-100,500,1000,5000,10000}"
PER_TF_TOPK="${PER_TF_TOPK:-200}"
PVAL_TRIALS="${PVAL_TRIALS:-500}"
PVAL_SEED="${PVAL_SEED:-13}"
PVAL_MAX_EDGES="${PVAL_MAX_EDGES:-50000}"

echo "[run_c1_metrics] Writing metrics to ${OUTPUT_ROOT}"
python3 evaluate/unified_baseline_eval.py \
  --output_root "${OUTPUT_ROOT}" \
  --k "${K_VALUES}" \
  --per_tf_topk "${PER_TF_TOPK}" \
  --random_trials "${PVAL_TRIALS}" \
  --random_seed "${PVAL_SEED}" \
  --pvalue_max_edges "${PVAL_MAX_EDGES}"

echo "[run_c1_metrics] Finished. Aggregate tables under ${OUTPUT_ROOT}"
