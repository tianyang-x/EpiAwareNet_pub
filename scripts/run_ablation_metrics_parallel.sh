#!/usr/bin/env bash
# Parallel version of scripts/run_ablation_metrics.sh
#
# Runs unified_baseline_eval.py over every ablation OUTPUT_ROOT in parallel and
# (optionally) aggregates results under DEST_ROOT.
#
# Usage examples:
#   DEST_ROOT=output/evaluate/ablations_fullgrid JOBS=8 bash scripts/run_ablation_metrics_parallel.sh
#   TARGET_DATASETS="mN,pN" DEST_ROOT=output/evaluate/ablations_fullgrid JOBS=4 bash scripts/run_ablation_metrics_parallel.sh
#
# Environment overrides (same meaning as scripts/run_ablation_metrics.sh):
#   ABLATION_ROOTS, TARGET_DATASETS, DEST_ROOT, EVAL_K, INCLUDE_MISSING, MISSING_SCORE
#   JOBS  - number of parallel workers (default: nproc or 4)
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

# Dataset -> gene list / TF list (used to avoid relying on OUTPUT_ROOT basename inference).
declare -A GENES_MAP=(
  [mN]="$REPO_ROOT/datasets/mN_combined/genes.txt"
  [pN]="$REPO_ROOT/datasets/pN_combined/genes.txt"
  [pbmc]="$REPO_ROOT/datasets/pbmc_preprocessed/genes.txt"
  [mouse_brain]="$REPO_ROOT/datasets/mouse_brain_combined/genes.txt"
)

declare -A TF_MAP=(
  [mN]="$REPO_ROOT/datasets/plusNshare/regulators.txt"
  [pN]="$REPO_ROOT/datasets/plusNshare/regulators.txt"
  [pbmc]="$REPO_ROOT/datasets/pbmc_preprocessed/tf_list.txt"
  [mouse_brain]="$REPO_ROOT/datasets/mouse_brain_combined/tfs_trrust_mouse.txt"
)

# Collect ablation directories.
read -r -a ROOTS <<<"${ABLATION_ROOTS:-}"
if (( ${#ROOTS[@]} == 0 )); then
  mapfile -t ROOTS < <(find "$REPO_ROOT/output" -maxdepth 1 -mindepth 1 -type d -name '*_ablations_*' | sort)
fi

if (( ${#ROOTS[@]} == 0 )); then
  echo "[run_ablation_metrics_parallel] No ablation directories found. Set ABLATION_ROOTS to override." >&2
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
JOBS=${JOBS:-""}
if [[ -z "$JOBS" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS=4
  fi
fi

# Build a task list (tab-separated):
# dataset<TAB>task_root<TAB>eval_dir<TAB>gold<TAB>genes<TAB>tfs
#
# We create one task per grn.csv directory (ablation subdir) to maximize parallelism.
TASKS=()
for root in "${ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    continue
  fi
  base="$(basename "$root")"
  dataset="${base%%_ablations_*}"
  if [[ -z "$dataset" || "$dataset" == "$base" ]]; then
    continue
  fi
  if ! _dataset_enabled "$dataset"; then
    continue
  fi
  gold="${GOLD_MAP[$dataset]:-}"
  if [[ -z "$gold" || ! -f "$gold" ]]; then
    continue
  fi

  genes="${GENES_MAP[$dataset]:-}"
  tfs="${TF_MAP[$dataset]:-}"
  if [[ -z "$genes" || -z "$tfs" || ! -f "$genes" || ! -f "$tfs" ]]; then
    continue
  fi

  # Collect every grn.csv under this ablation run root.
  while IFS= read -r grn_csv; do
    [[ -z "$grn_csv" ]] && continue
    task_root="$(dirname "$grn_csv")"
    rel_dir="${task_root#"$root"/}"
    tag="${rel_dir//\//__}"
    if [[ -z "$tag" || "$tag" == "$task_root" ]]; then
      tag="root"
    fi

    if [[ -n "${DEST_ROOT:-}" ]]; then
      eval_dir="${DEST_ROOT%/}/$dataset/$base/$tag"
    else
      eval_dir="$task_root/eval_metrics"
    fi

    TASKS+=("${dataset}"$'\t'"${task_root}"$'\t'"${eval_dir}"$'\t'"${gold}"$'\t'"${genes}"$'\t'"${tfs}")
  done < <(find "$root" -type f -name 'grn.csv' 2>/dev/null | sort)
done

if (( ${#TASKS[@]} == 0 )); then
  echo "[run_ablation_metrics_parallel] No tasks after filtering (TARGET_DATASETS/paths)." >&2
  exit 2
fi

echo "[run_ablation_metrics_parallel] tasks=${#TASKS[@]} workers=$JOBS dest_root=${DEST_ROOT:-"(per-run eval_metrics)"}"

_run_one() {
  local line="$1"
  local dataset task_root eval_dir gold genes tfs
  IFS=$'\t' read -r dataset task_root eval_dir gold genes tfs <<<"$line"

  if [[ -z "${dataset}" || -z "${task_root}" || -z "${eval_dir}" || -z "${gold}" || -z "${genes}" || -z "${tfs}" ]]; then
    echo "[eval][fail ][$(date +%H:%M:%S)] Bad task line (expected 6 tab-separated fields): $line" >&2
    return 2
  fi

  mkdir -p "$eval_dir"

  local base
  base="$(basename "$task_root")"
  echo "[eval][start][$(date +%H:%M:%S)][${dataset}] ${base} -> ${eval_dir}" >&2

  set +e
  env \
    OUTPUT_ROOT="$task_root" \
    EVAL_GOLD="$gold" \
    EVAL_OUTPUT_DIR="$eval_dir" \
    EVAL_GENES="$genes" \
    EVAL_TF_LIST="$tfs" \
    EVAL_K="$K_VALUES" \
    EVAL_INCLUDE_MISSING_POSITIVES="$INCLUDE_MISSING" \
    EVAL_MISSING_POSITIVE_SCORE="$MISSING_SCORE" \
    bash "$REPO_ROOT/run/eval_ablation_output.sh"
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[eval][done ][$(date +%H:%M:%S)][${dataset}] ${base}" >&2
  else
    echo "[eval][fail ][$(date +%H:%M:%S)][${dataset}] ${base} (exit=$rc)" >&2
  fi
  return $rc
}

# Export things needed by subshells.
export -f _run_one _dataset_enabled
export REPO_ROOT K_VALUES INCLUDE_MISSING MISSING_SCORE

if command -v parallel >/dev/null 2>&1; then
  # --bar shows a global progress bar (preferred). --line-buffer keeps each job's logs readable.
  printf '%s\n' "${TASKS[@]}" | parallel -j "$JOBS" --halt soon,fail=1 --bar --line-buffer _run_one {}
else
  # Fallback: xargs-based parallelism.
  # Note: uses bash to call the function in a subshell.
  printf '%s\n' "${TASKS[@]}" | xargs -P "$JOBS" -I{} bash -lc '_run_one "$1"' _ {}
fi

if [[ -n "${DEST_ROOT:-}" ]]; then
  echo "[run_ablation_metrics_parallel] Aggregating unified metrics under $DEST_ROOT" >&2
  python3 "$REPO_ROOT/scripts/aggregate_unified_ablation_metrics.py" --root "$DEST_ROOT"
fi

echo "[run_ablation_metrics_parallel] Done." >&2
