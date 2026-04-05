#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

# Complexity & scalability benchmark.
# Sweeps:
#   G: 2k -> 10k
#   P: 50k -> 300k
# Records runtime and peak GPU memory.
#
# Defaults are chosen to be safe-ish on a single GPU; override as needed.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_scaling_benchmark.sh
#   CUDA_VISIBLE_DEVICES=0 MAX_CAND=200 STEPS=5 WARMUP=2 bash scripts/run_scaling_benchmark.sh

OUT_CSV=${OUT_CSV:-"output/scaling_benchmark/scaling_$(date +%Y%m%d_%H%M%S).csv"}

G_LIST=${G_LIST:-"2000 5000 10000"}
P_LIST=${P_LIST:-"50000 100000 300000"}

# Candidate set size per gene (controls cross-attn cost)
MAX_CAND=${MAX_CAND:-50}

# Stage1 routing K values: include 0 for no routing (attend all candidates)
TOPK_LIST=${TOPK_LIST:-"8 0"}

BATCH=${BATCH:-1}
EMBED_DIM=${EMBED_DIM:-256}
HEADS=${HEADS:-8}
DEPTH=${DEPTH:-6}
KNN=${KNN:-16}

CROSS_CHUNK=${CROSS_CHUNK:-1024}
SELF_CHUNK=${SELF_CHUNK:-1024}

WARMUP=${WARMUP:-3}
STEPS=${STEPS:-10}

CAND_MODE=${CAND_MODE:-random}  # random|fixed
SEED=${SEED:-13}

AMP=${AMP:-1}
AMP_FLAG=()
if [[ "$AMP" == "0" ]]; then
  AMP_FLAG=(--no-amp)
fi

mkdir -p "$(dirname "$OUT_CSV")"

echo "[bench] Writing -> $OUT_CSV"

echo "[bench] G: $G_LIST"
echo "[bench] P: $P_LIST"
echo "[bench] max_candidates=$MAX_CAND topk=$TOPK_LIST batch=$BATCH"

time python3 scripts/benchmark_scaling.py \
  --G "$G_LIST" \
  --P "$P_LIST" \
  --batch "$BATCH" \
  --embed_dim "$EMBED_DIM" \
  --num_heads "$HEADS" \
  --depth "$DEPTH" \
  --k_neighbors "$KNN" \
  --max_candidates "$MAX_CAND" \
  --topk_peaks "$TOPK_LIST" \
  --candidate_mode "$CAND_MODE" \
  --seed "$SEED" \
  --cross_chunk_size "$CROSS_CHUNK" \
  --self_chunk_size "$SELF_CHUNK" \
  --warmup "$WARMUP" \
  --steps "$STEPS" \
  --out_csv "$OUT_CSV" \
  "${AMP_FLAG[@]}"

echo "[done] Benchmark finished: $OUT_CSV"