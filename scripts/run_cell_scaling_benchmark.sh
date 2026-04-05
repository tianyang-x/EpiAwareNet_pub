#!/usr/bin/env bash
# Measure inference/export time as the number of cells grows (fixed G/P/K).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONDA_BIN=${CONDA_BIN:-$(which conda 2>/dev/null || echo conda)}
EPIAWARE_ENV=${EPIAWARE_ENV:-grn_transformer}
DATASETS=${DATASETS:-"mN pN"}

# Default subsample sizes per dataset (override via CELL_COUNTS_<dataset>=...).
CELL_COUNTS_mN=${CELL_COUNTS_mN:-"5000 8000"}
CELL_COUNTS_pN=${CELL_COUNTS_pN:-"5000 10000 12000"}

TF_LIST=${TF_LIST:-"$REPO_ROOT/datasets/plusNshare/regulators.txt"}
POS_LINKS=${POS_LINKS:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_train.txt"}
VAL_LINKS=${VAL_LINKS:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_val.txt"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/output/scaling_benchmark/cell_scaling"}
EXPORT_TOPK=${EXPORT_TOPK:-200}
EXPORT_THRESHOLD=${EXPORT_THRESHOLD:-0.0}
EXPORT_TF_CHUNK=${EXPORT_TF_CHUNK:-16}
EXPORT_TARGET_CHUNK=${EXPORT_TARGET_CHUNK:-2048}
EXPORT_SEED=${EXPORT_SEED:-42}

mkdir -p "$OUTPUT_ROOT"

declare -A RNA_PATHS=(
  [mN]="$REPO_ROOT/datasets/mN_combined/rna.npy"
  [pN]="$REPO_ROOT/datasets/pN_combined/rna.npy"
)
declare -A ATAC_PATHS=(
  [mN]="$REPO_ROOT/datasets/mN_combined_ATAC/atac.npy"
  [pN]="$REPO_ROOT/datasets/pN_combined_ATAC/atac.npy"
)
declare -A GENE_NAMES=(
  [mN]="$REPO_ROOT/datasets/mN_combined/genes.txt"
  [pN]="$REPO_ROOT/datasets/pN_combined/genes.txt"
)
declare -A PEAK_NAMES=(
  [mN]="$REPO_ROOT/datasets/mN_combined_ATAC/peaks.txt"
  [pN]="$REPO_ROOT/datasets/pN_combined_ATAC/peaks.txt"
)
declare -A CAND_JSON=(
  [mN]="$REPO_ROOT/datasets/mN_combined/candidate_peaks.json"
  [pN]="$REPO_ROOT/datasets/pN_combined/candidate_peaks.json"
)
declare -A BACKBONE=(
  [mN]="${BACKBONE_mN:-$REPO_ROOT/output/mN_pipeline/shared/stage1_multiomic/epiaware_backbone.pth}"
  [pN]="${BACKBONE_pN:-$REPO_ROOT/output/pN_pipeline/shared/stage1_multiomic/epiaware_backbone.pth}"
)
declare -A HEAD=(
  [mN]="${HEAD_mN:-$REPO_ROOT/output/mN_pipeline/shared/stage2_baseline_pu/grn_head.pth}"
  [pN]="${HEAD_pN:-$REPO_ROOT/output/pN_pipeline/shared/stage2_baseline_pu/grn_head.pth}"
)

_cell_counts_for() {
  local ds="$1"
  local var="CELL_COUNTS_${ds}"
  local value="${!var:-}"
  if [[ -z "$value" ]]; then
    echo "[warn] No CELL_COUNTS specified for $ds; skipping." >&2
    return 1
  fi
  echo "$value"
}

_need_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] Missing file: $path" >&2
    exit 2
  fi
}

for ds in $DATASETS; do
  if [[ -z "${RNA_PATHS[$ds]:-}" ]]; then
    echo "[warn] Unknown dataset '$ds'; skipping."
    continue
  fi
  for path in \
    "${RNA_PATHS[$ds]}" "${ATAC_PATHS[$ds]}" "${GENE_NAMES[$ds]}" "${PEAK_NAMES[$ds]}" \
    "${CAND_JSON[$ds]}" "${BACKBONE[$ds]}" "${HEAD[$ds]}" "$TF_LIST" "$POS_LINKS" "$VAL_LINKS"; do
    _need_file "$path"
  done

  counts="$(_cell_counts_for "$ds")" || continue
  for N in $counts; do
    out_dir="$OUTPUT_ROOT/${ds}_N${N}"
    log="$out_dir/run.log"
    mkdir -p "$out_dir"
    echo "[run] dataset=$ds cells=$N -> $out_dir"
    /usr/bin/time -f "elapsed=%e\nmax_rss_kb=%M" \
      "$CONDA_BIN" run -n "$EPIAWARE_ENV" python training/finetune_epiaware_grn.py \
        --rna_path "${RNA_PATHS[$ds]}" \
        --atac_path "${ATAC_PATHS[$ds]}" \
        --candidate_file "${CAND_JSON[$ds]}" \
        --gene_names "${GENE_NAMES[$ds]}" \
        --peak_names "${PEAK_NAMES[$ds]}" \
        --tf_list "$TF_LIST" \
        --positive_links "$POS_LINKS" \
        --val_positive_links "$VAL_LINKS" \
        --backbone_checkpoint "${BACKBONE[$ds]}" \
        --head_checkpoint "${HEAD[$ds]}" \
        --output_dir "$out_dir" \
        --epochs 0 \
        --eval_only \
        --reuse_knn_indices \
        --export_grn_csv "$out_dir/grn.csv" \
        --export_probability_threshold "$EXPORT_THRESHOLD" \
        --export_max_cells "$N" \
        --export_cell_sampling random \
        --export_cell_seed "$EXPORT_SEED" \
        --export_topk_per_tf "$EXPORT_TOPK" \
        --export_tf_chunk_size "$EXPORT_TF_CHUNK" \
        --export_target_chunk_size "$EXPORT_TARGET_CHUNK" \
        --cell_batch_size 32 \
        --pos_batch_size 256 \
        --unl_batch_size 1024 \
        --skip_validation \
        >>"$log" 2>&1
  done
done

echo "[done] Cell-scaling measurements stored in $OUTPUT_ROOT"
