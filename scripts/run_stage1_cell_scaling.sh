#!/usr/bin/env bash
# Run Stage 1 backbone training on subsampled cell counts to measure scalability.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CONDA_BIN=${CONDA_BIN:-$(which conda 2>/dev/null || echo conda)}
EPIAWARE_ENV=${EPIAWARE_ENV:-grn_transformer}
DATASETS=${DATASETS:-"mN pN"}
GPU_LIST=${GPU_LIST:-"0"}

# Cell counts per dataset (override via env).
CELL_COUNTS_mN=${CELL_COUNTS_mN:-"5000 8000"}
CELL_COUNTS_pN=${CELL_COUNTS_pN:-"5000 10000 20000 40000"}
SUBSET_SEED=${SUBSET_SEED:-123}

SUBSET_ROOT=${SUBSET_ROOT:-"$REPO_ROOT/output/scaling_benchmark/cell_subsets"}
STAGE1_OUT_ROOT=${STAGE1_OUT_ROOT:-"$REPO_ROOT/output/scaling_benchmark/stage1"}

STAGE1_EPOCHS=${STAGE1_EPOCHS:-5}
STAGE1_BATCH=${STAGE1_BATCH:-8}
STAGE1_LR=${STAGE1_LR:-1e-4}
STAGE1_WEIGHT_DECAY=${STAGE1_WEIGHT_DECAY:-0.02}
VAL_FRACTION=${VAL_FRACTION:-0.1}
NUM_WORKERS=${NUM_WORKERS:-4}
EXTRA_ARGS=${EXTRA_ARGS:-""}

mkdir -p "$SUBSET_ROOT" "$STAGE1_OUT_ROOT"

read -r -a GPU_ARRAY <<<"$GPU_LIST"
GPU_COUNT=${#GPU_ARRAY[@]}
if (( GPU_COUNT == 0 )); then
  echo "[error] GPU_LIST is empty; set GPU_LIST=\"0\" (or similar)." >&2
  exit 2
fi

declare -A RNA_PATHS=(
  [mN]="$REPO_ROOT/datasets/mN_combined/rna.npy"
  [pN]="$REPO_ROOT/datasets/pN_combined/rna.npy"
)
declare -A ATAC_PATHS=(
  [mN]="$REPO_ROOT/datasets/mN_combined_ATAC/atac.npy"
  [pN]="$REPO_ROOT/datasets/pN_combined_ATAC/atac.npy"
)
declare -A CELLS_TXT=(
  [mN]="$REPO_ROOT/datasets/mN_combined/cells.txt"
  [pN]="$REPO_ROOT/datasets/pN_combined/cells.txt"
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

_cell_counts_for() {
  local ds="$1"
  local var="CELL_COUNTS_${ds}"
  local value="${!var:-}"
  if [[ -z "$value" ]]; then
    echo "[warn] No CELL_COUNTS defined for $ds; skipping." >&2
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

_prepare_subset() {
  local ds="$1"
  local count="$2"
  local subset_dir="$3"
  local rna_src="${RNA_PATHS[$ds]}"
  local atac_src="${ATAC_PATHS[$ds]}"
  local cells_src="${CELLS_TXT[$ds]}"
  local rna_out="$subset_dir/rna.npy"
  local atac_out="$subset_dir/atac.npy"
  local cells_out="$subset_dir/cells.txt"
  local idx_out="$subset_dir/indices.npy"

   # skip impossible requests
  local total_cells
  total_cells=$(/usr/bin/python3 - <<'PY' "$rna_src"
import sys, numpy as np
arr = np.load(sys.argv[1], mmap_mode="r")
print(arr.shape[0])
PY
)
  if (( count > total_cells )); then
    echo "[skip] $ds: requested $count cells but dataset has only $total_cells; skipping this N" >&2
    return 1
  fi

  if [[ -f "$rna_out" && -f "$atac_out" ]]; then
    return
  fi
  echo "[subset] Sampling $count cells for $ds -> $subset_dir"
  PYTHONPATH="$REPO_ROOT" /usr/bin/python3 - "$rna_src" "$atac_src" "$rna_out" "$atac_out" "$cells_src" "$cells_out" "$idx_out" "$count" "$((SUBSET_SEED + count))" <<'PY'
import sys
import numpy as np

rna_in, atac_in, rna_out, atac_out, cells_in, cells_out, indices_out, count, seed = sys.argv[1:]
count = int(count)
seed = int(seed)

rng = np.random.default_rng(seed)
rna = np.load(rna_in, mmap_mode="r")
atac = np.load(atac_in, mmap_mode="r")
total = rna.shape[0]
if total < count:
    raise SystemExit(f"Requested {count} cells but dataset only has {total}.")
idx = np.sort(rng.choice(total, size=count, replace=False))
np.save(indices_out, idx.astype(np.int64), allow_pickle=False)
np.save(rna_out, np.asarray(rna[idx], dtype=rna.dtype))
np.save(atac_out, np.asarray(atac[idx], dtype=atac.dtype))
with open(cells_in, "r", encoding="utf-8") as fin, open(cells_out, "w", encoding="utf-8") as fout:
    cells = [line.rstrip("\n") for line in fin]
    for i in idx:
        fout.write(cells[int(i)] + "\n")
PY
}

job_idx=0
for ds in $DATASETS; do
  if [[ -z "${RNA_PATHS[$ds]:-}" ]]; then
    echo "[warn] Unknown dataset '$ds'; skipping."
    continue
  fi
  for path in "${RNA_PATHS[$ds]}" "${ATAC_PATHS[$ds]}" "${CELLS_TXT[$ds]}" \
    "${GENE_NAMES[$ds]}" "${PEAK_NAMES[$ds]}" "${CAND_JSON[$ds]}"; do
    _need_file "$path"
  done
  counts="$(_cell_counts_for "$ds")" || continue
  for N in $counts; do
    subset_dir="$SUBSET_ROOT/${ds}_N${N}"
    mkdir -p "$subset_dir"
    _prepare_subset "$ds" "$N" "$subset_dir"

    out_dir="$STAGE1_OUT_ROOT/${ds}_N${N}"
    mkdir -p "$out_dir"
    log="$out_dir/stage1.log"
    gpu="${GPU_ARRAY[$((job_idx % GPU_COUNT))]}"
    echo "[stage1] dataset=$ds cells=$N gpu=$gpu -> $out_dir" | tee "$log"
    CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/time -f "stage1_elapsed=%e\nstage1_max_rss_kb=%M" \
      "$CONDA_BIN" run -n "$EPIAWARE_ENV" python training/train_epiaware_backbone.py \
        --rna_path "$subset_dir/rna.npy" \
        --atac_path "$subset_dir/atac.npy" \
        --candidate_file "${CAND_JSON[$ds]}" \
        --gene_names "${GENE_NAMES[$ds]}" \
        --peak_names "${PEAK_NAMES[$ds]}" \
        --output_dir "$out_dir" \
        --epochs "$STAGE1_EPOCHS" \
        --batch_size "$STAGE1_BATCH" \
        --learning_rate "$STAGE1_LR" \
        --weight_decay "$STAGE1_WEIGHT_DECAY" \
        --val_fraction "$VAL_FRACTION" \
        --num_workers "$NUM_WORKERS" \
        --wandb_mode disabled \
        $EXTRA_ARGS \
        >>"$log" 2>&1
    job_idx=$((job_idx + 1))
  done
done

echo "[done] Stage1 scalability runs stored in $STAGE1_OUT_ROOT"
