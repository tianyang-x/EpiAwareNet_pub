#!/usr/bin/env bash
# Re-run the baseline metrics script over every ablation GRN directory.
#
# Usage:
#   bash scripts/run_ablation_metrics.sh
# Environment overrides:
#   ABLATION_ROOTS   - whitespace-separated list of ablation directories to evaluate.
#                      Defaults to all output/*_ablations_* folders.
#   TARGET_DATASETS  - subset of datasets to keep (space/comma separated; default: all).
#   DEST_ROOT        - optional global output root; results will go under
#                      DEST_ROOT/<dataset>/<ablation_basename>. Default: write
#                      next to each ablation folder (<ablation>/eval_metrics).
#   EVAL_K           - comma-separated K values (default: 100,500,1000).
#   INCLUDE_MISSING  - set to 1 to account for missing positives.
#   MISSING_SCORE    - override missing-positive score when INCLUDE_MISSING=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Dataset -> gold links (validation splits).
declare -A GOLD_MAP=(
  [mN]="$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_val.txt"
  [pN]="$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_val.txt"
  [pbmc]="$REPO_ROOT/datasets/dorothea_high_confidence_links_val.csv"
  [mouse_brain]="$REPO_ROOT/datasets/mouse_brain_combined/positive_links_trrust_mouse_val_1266.tsv"
)

# Collect ablation directories.
read -r -a ROOTS <<<"${ABLATION_ROOTS:-}"
if (( ${#ROOTS[@]} == 0 )); then
  mapfile -t ROOTS < <(find "$REPO_ROOT/output" -maxdepth 1 -mindepth 1 -type d -name '*_ablations_*' | sort)
fi

if (( ${#ROOTS[@]} == 0 )); then
  echo "[run_ablation_metrics] No ablation directories found. Set ABLATION_ROOTS to override." >&2
  exit 1
fi

TARGET_SET=()
if [[ -n "${TARGET_DATASETS:-}" ]]; then
  IFS=', ' read -r -a TARGET_SET <<<"$TARGET_DATASETS"
fi

_dataset_enabled() {
  local ds="$1"
  if (( ${#TARGET_SET[@]} == 0 )); then
    return 0
  fi
  local item
  for item in "${TARGET_SET[@]}"; do
    if [[ "$item" == "$ds" ]]; then
      return 0
    fi
  done
  return 1
}

K_VALUES=${EVAL_K:-"100,500,1000"}
INCLUDE_MISSING=${INCLUDE_MISSING:-0}
MISSING_SCORE=${MISSING_SCORE:-""}

echo "[run_ablation_metrics] Evaluating ${#ROOTS[@]} ablation directories using unified_baseline_eval full-grid"
for root in "${ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "[run_ablation_metrics][skip] Not a directory: $root"
    continue
  fi
  base="$(basename "$root")"
  dataset="${base%%_ablations_*}"
  if [[ -z "$dataset" || "$dataset" == "$base" ]]; then
    echo "[run_ablation_metrics][skip] Cannot infer dataset from $base"
    continue
  fi
  if ! _dataset_enabled "$dataset"; then
    echo "[run_ablation_metrics][skip] Dataset filter excludes $dataset"
    continue
  fi
  gold="${GOLD_MAP[$dataset]:-}"
  if [[ -z "$gold" ]]; then
    echo "[run_ablation_metrics][skip] Unknown dataset for $base"
    continue
  fi
  if [[ ! -f "$gold" ]]; then
    echo "[run_ablation_metrics][skip] Gold file missing for $dataset: $gold"
    continue
  fi

  if [[ -n "${DEST_ROOT:-}" ]]; then
    eval_dir="${DEST_ROOT%/}/$dataset/$base"
  else
    eval_dir="$root/eval_metrics"
  fi
  mkdir -p "$eval_dir"

  echo "[run_ablation_metrics][$dataset] $base -> $eval_dir"
  env \
    OUTPUT_ROOT="$root" \
    EVAL_GOLD="$gold" \
    EVAL_OUTPUT_DIR="$eval_dir" \
    EVAL_K="$K_VALUES" \
    EVAL_INCLUDE_MISSING_POSITIVES="$INCLUDE_MISSING" \
    EVAL_MISSING_POSITIVE_SCORE="$MISSING_SCORE" \
    bash "$REPO_ROOT/run/eval_ablation_output.sh"
done

if [[ -n "${DEST_ROOT:-}" ]]; then
  echo "[run_ablation_metrics] Aggregating unified metrics under $DEST_ROOT"
  python3 "$REPO_ROOT/scripts/aggregate_unified_ablation_metrics.py" --root "$DEST_ROOT"
else
  echo "[run_ablation_metrics] Skipping aggregation (DEST_ROOT not set)."
fi

echo "[run_ablation_metrics] Done."
