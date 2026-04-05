#!/usr/bin/env bash
# Run Gene/Peak/Candidate (K) scaling sweeps using benchmark_scaling.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

OUT_ROOT=${OUT_ROOT:-"$REPO_ROOT/output/scaling_benchmark"}
mkdir -p "$OUT_ROOT"

# Common hyperparameters (roughly match pN backbone settings).
BATCH=${BATCH:-1}
EMBED_DIM=${EMBED_DIM:-256}
HEADS=${HEADS:-8}
DEPTH=${DEPTH:-6}
KNN=${KNN:-16}
CROSS_CHUNK=${CROSS_CHUNK:-1024}
SELF_CHUNK=${SELF_CHUNK:-1024}
CAND_MODE=${CAND_MODE:-random}
SEED=${SEED:-13}

# Training measurement knobs.
TRAIN_WARMUP=${TRAIN_WARMUP:-2}
TRAIN_STEPS=${TRAIN_STEPS:-6}
TRAIN_LR=${TRAIN_LR:-1e-4}

# AMP + RNA-only flags.
AMP=${AMP:-1}
AMP_FLAG=()
if [[ "$AMP" == "0" ]]; then
  AMP_FLAG=(--no-amp)
fi
INCLUDE_GENE_ONLY=${INCLUDE_GENE_ONLY:-0}
GENEONLY_FLAG=()
if [[ "$INCLUDE_GENE_ONLY" == "0" ]]; then
  GENEONLY_FLAG=(--no-geneonly)
fi

# Gene scaling settings (vary G, hold P + K fixed).
GENE_SWEEP=${GENE_SWEEP:-"2000 5000 10000 20000"}
FIXED_P_FOR_G=${FIXED_P_FOR_G:-200000}
K_FIXED=${K_FIXED:-200}
MAX_CAND_G=${MAX_CAND_G:-200}

# Peak scaling settings (vary P, hold G + K fixed).
PEAK_SWEEP=${PEAK_SWEEP:-"50000 100000 200000 400000"}
FIXED_G_FOR_P=${FIXED_G_FOR_P:-20000}
MAX_CAND_P=${MAX_CAND_P:-200}

# Candidate (top-K) scaling settings (vary top-k routing, hold G + P fixed).
K_SWEEP=${K_SWEEP:-"50 100 200 500 1000"}
FIXED_G_FOR_K=${FIXED_G_FOR_K:-20000}
FIXED_P_FOR_K=${FIXED_P_FOR_K:-200000}
MAX_CAND_K=${MAX_CAND_K:-1000}

COMMON_FLAGS=(
  --batch "$BATCH"
  --embed_dim "$EMBED_DIM"
  --num_heads "$HEADS"
  --depth "$DEPTH"
  --k_neighbors "$KNN"
  --cross_chunk_size "$CROSS_CHUNK"
  --self_chunk_size "$SELF_CHUNK"
  --candidate_mode "$CAND_MODE"
  --seed "$SEED"
  --warmup 3
  --steps 10
  --train_warmup "$TRAIN_WARMUP"
  --train_steps "$TRAIN_STEPS"
  --train_lr "$TRAIN_LR"
  "${AMP_FLAG[@]}"
  "${GENEONLY_FLAG[@]}"
)

run_section() {
  local label="$1"
  shift
  local out_csv="$OUT_ROOT/${label}.csv"
  echo "[sweep][$label] -> $out_csv"
  python3 scripts/benchmark_scaling.py \
    "$@" \
    --out_csv "$out_csv"
}

run_section gene_scaling \
  --tag gene_scaling \
  --G "$GENE_SWEEP" \
  --P "$FIXED_P_FOR_G" \
  --max_candidates "$MAX_CAND_G" \
  --topk_peaks "$K_FIXED" \
  "${COMMON_FLAGS[@]}"

run_section peak_scaling \
  --tag peak_scaling \
  --G "$FIXED_G_FOR_P" \
  --P "$PEAK_SWEEP" \
  --max_candidates "$MAX_CAND_P" \
  --topk_peaks "$K_FIXED" \
  "${COMMON_FLAGS[@]}"

run_section candidate_scaling \
  --tag candidate_scaling \
  --G "$FIXED_G_FOR_K" \
  --P "$FIXED_P_FOR_K" \
  --max_candidates "$MAX_CAND_K" \
  --topk_peaks "$K_SWEEP" \
  "${COMMON_FLAGS[@]}"

echo "[done] Gene/Peak/K scaling CSVs stored in $OUT_ROOT"
