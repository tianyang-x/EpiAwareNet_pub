#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RNA_DIR=${RNA_DIR:-datasets/mN_combined}
ATAC_DIR=${ATAC_DIR:-datasets/mN_combined_ATAC}
CANDIDATE_JSON=${CANDIDATE_JSON:-$RNA_DIR/candidate_peaks.json}
TF_LIST=${TF_LIST:-datasets/plusNshare/regulators.txt}
POS_LINKS=${POS_LINKS:-datasets/TF_Target_Validation_inSingleCell_train.txt}
VAL_POS_LINKS=${VAL_POS_LINKS:-datasets/TF_Target_Validation_inSingleCell_val.txt}

OUTPUT_ROOT=${OUTPUT_ROOT:-output/mN_ablations_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTPUT_ROOT"

# Shared checkpoints
SHARED_DIR="$OUTPUT_ROOT/shared"
STAGE1_MULTIOMIC_OUT="$SHARED_DIR/stage1_multiomic"
STAGE1_RNA_ONLY_OUT="$SHARED_DIR/stage1_rna_only"
BASELINE_STAGE2_OUT="$SHARED_DIR/stage2_baseline_pu"
BASELINE_HEAD_CKPT=${BASELINE_HEAD_CKPT:-$BASELINE_STAGE2_OUT/grn_head.pth}

# Random candidate set for weak-prior ablation
RAND_SEED=${RAND_SEED:-13}
MAX_PEAKS=${MAX_PEAKS:-50}
RAND_CANDIDATE_JSON="$SHARED_DIR/candidates/random_k${MAX_PEAKS}_seed${RAND_SEED}.json"

# Training hyperparams (override via env)
STAGE1_EPOCHS=${STAGE1_EPOCHS:-2}
STAGE1_RNA_ONLY_EPOCHS=${STAGE1_RNA_ONLY_EPOCHS:-2}
BASELINE_HEAD_EPOCHS=${BASELINE_HEAD_EPOCHS:-1}
BCE_HEAD_EPOCHS=${BCE_HEAD_EPOCHS:-1}
RNA_ONLY_HEAD_EPOCHS=${RNA_ONLY_HEAD_EPOCHS:-1}
TOPK_STAGE1_LIST=${TOPK_STAGE1_LIST:-"0 2 4 8 16"}
TOPK_STAGE1_HEAD_EPOCHS=${TOPK_STAGE1_HEAD_EPOCHS:-1}

# Optional: candidate-swap stress test (inference-time only; not a fair Stage1 ablation)
ENABLE_CANDIDATE_SWAP_STRESS_TEST=${ENABLE_CANDIDATE_SWAP_STRESS_TEST:-0}

# Fair-comparison Stage2 hyperparameters (override via env)
CELL_BATCH_SIZE=${CELL_BATCH_SIZE:-32}
POS_BATCH_SIZE=${POS_BATCH_SIZE:-256}
UNL_BATCH_SIZE=${UNL_BATCH_SIZE:-1024}
VAL_PROB_THRESHOLD=${VAL_PROB_THRESHOLD:-0.5}
VAL_FRACTION=${VAL_FRACTION:-0.15}
POSITIVE_PRIOR=${POSITIVE_PRIOR:-0.05}
NEGATIVE_RISK_WEIGHT=${NEGATIVE_RISK_WEIGHT:-1.0}
NUM_WORKERS=${NUM_WORKERS:-4}

# Optional: run custom_evaluate.py over every exported grn.csv (can be slow for very large GRNs)
RUN_CUSTOM_EVAL=${RUN_CUSTOM_EVAL:-1}
FORCE_CUSTOM_EVAL=${FORCE_CUSTOM_EVAL:-0}
CUSTOM_EVAL_GOLD=${CUSTOM_EVAL_GOLD:-datasets/TF_Target_Validation_inSingleCell.txt}
CUSTOM_EVAL_GENES=${CUSTOM_EVAL_GENES:-$RNA_DIR/genes.txt}
CUSTOM_EVAL_PRECISION_TARGETS=${CUSTOM_EVAL_PRECISION_TARGETS:-"0.20 0.25 0.30 0.35"}

maybe_run_custom_eval() {
  local grn_csv="$1"
  if [[ "$RUN_CUSTOM_EVAL" == "0" ]]; then
    return 0
  fi
  if [[ ! -f "$grn_csv" ]]; then
    echo "[eval][warn] Missing GRN CSV: $grn_csv"
    return 0
  fi
  local outdir
  outdir="$(dirname "$grn_csv")/custom_eval"
  if [[ "$FORCE_CUSTOM_EVAL" == "0" && -f "$outdir/metrics.json" ]]; then
    echo "[eval] Skip (exists): $outdir/metrics.json"
    return 0
  fi
  echo "[eval] $grn_csv -> $outdir"
  python3 custom_evaluate.py \
    --predictions "$grn_csv" \
    --gold "$CUSTOM_EVAL_GOLD" \
    --genes "$CUSTOM_EVAL_GENES" \
    --output_dir "$outdir" \
    --precision_targets $CUSTOM_EVAL_PRECISION_TARGETS \
    ${CUSTOM_EVAL_ARGS:-}
}

FAIR_STAGE2_ARGS=(
  --cell_batch_size "$CELL_BATCH_SIZE"
  --pos_batch_size "$POS_BATCH_SIZE"
  --unl_batch_size "$UNL_BATCH_SIZE"
  --val_prob_threshold "$VAL_PROB_THRESHOLD"
  --val_fraction "$VAL_FRACTION"
  --positive_prior "$POSITIVE_PRIOR"
  --negative_risk_weight "$NEGATIVE_RISK_WEIGHT"
  --num_workers "$NUM_WORKERS"
)

# ------------------------------
# Optional multi-GPU export mode
# ------------------------------
# Note: each python process uses a single GPU. This mode parallelizes export-only
# runs across multiple GPUs (e.g., ablation2/ablation4), while keeping Stage1/Stage2
# training sequential for simplicity.
PARALLEL_EXPORTS=${PARALLEL_EXPORTS:-0}
GPUS=${GPUS:-""}                # e.g. "0 1 2 3" (space-separated)
MAX_JOBS=${MAX_JOBS:-0}          # 0 => one job per GPU

declare -a GPU_LIST=()
declare -a JOB_PIDS=()
NEXT_GPU_IDX=0

init_gpus() {
  if [[ -n "$GPUS" ]]; then
    read -r -a GPU_LIST <<< "$GPUS"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    # Extract numeric GPU ids from `nvidia-smi --list-gpus`
    mapfile -t GPU_LIST < <(nvidia-smi --list-gpus | awk -F: '{print $1}' | awk '{print $2}')
  else
    GPU_LIST=(0)
  fi

  if (( ${#GPU_LIST[@]} == 0 )); then
    GPU_LIST=(0)
  fi
  if (( MAX_JOBS <= 0 )); then
    MAX_JOBS=${#GPU_LIST[@]}
  fi
}

prune_finished_jobs() {
  local -a keep=()
  local pid
  for pid in "${JOB_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      keep+=("$pid")
    fi
  done
  JOB_PIDS=("${keep[@]}")
}

wait_for_job_slot() {
  while true; do
    prune_finished_jobs
    if (( ${#JOB_PIDS[@]} < MAX_JOBS )); then
      return 0
    fi
    sleep 2
  done
}

run_job_on_next_gpu() {
  local label="$1"; shift
  local -a cmd=("$@")

  wait_for_job_slot
  local gpu="${GPU_LIST[$NEXT_GPU_IDX]}"
  NEXT_GPU_IDX=$(( (NEXT_GPU_IDX + 1) % ${#GPU_LIST[@]} ))

  echo "[parallel][gpu=$gpu] $label"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "${cmd[@]}"
  ) &
  JOB_PIDS+=("$!")
}

finalize_parallel_jobs() {
  if (( ${#JOB_PIDS[@]} > 0 )); then
    echo "[parallel] Waiting for ${#JOB_PIDS[@]} export jobs to finish..."
    wait "${JOB_PIDS[@]}"
  fi
}

if [[ "$PARALLEL_EXPORTS" != "0" ]]; then
  init_gpus
  echo "[parallel] Enabled export parallelism on GPUs: ${GPU_LIST[*]} (MAX_JOBS=$MAX_JOBS)"
fi

VAL_POS_FLAG=()
if [[ -n "${VAL_POS_LINKS:-}" ]]; then
  VAL_POS_FLAG=(--val_positive_links "$VAL_POS_LINKS")
fi

mkdir -p "$SHARED_DIR/candidates"

if [[ ! -f "$RAND_CANDIDATE_JSON" ]]; then
  echo "[prep] Generating random candidate set -> $RAND_CANDIDATE_JSON"
  python3 scripts/generate_random_candidate_set.py \
    --genes_txt "$RNA_DIR/genes.txt" \
    --peaks_txt "$ATAC_DIR/peaks.txt" \
    --output "$RAND_CANDIDATE_JSON" \
    --max_peaks_per_gene "$MAX_PEAKS" \
    --seed "$RAND_SEED"
fi

# ------------------------------
# Baseline Stage1 (multi-omic)
# ------------------------------
if [[ ! -f "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth" ]]; then
  echo "[baseline][stage1] Training multi-omic backbone -> $STAGE1_MULTIOMIC_OUT"
  python3 training/train_epiaware_backbone.py \
    --rna_path "$RNA_DIR/rna.npy" \
    --atac_path "$ATAC_DIR/atac.npy" \
    --candidate_file "$CANDIDATE_JSON" \
    --gene_names "$RNA_DIR/genes.txt" \
    --peak_names "$ATAC_DIR/peaks.txt" \
    --output_dir "$STAGE1_MULTIOMIC_OUT" \
    --epochs "$STAGE1_EPOCHS" \
    ${STAGE1_ARGS:-}
else
  echo "[baseline][stage1] Reusing $STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth"
fi

# ------------------------------
# Baseline Stage2 head (nnPU)
# ------------------------------
if [[ ! -f "$BASELINE_HEAD_CKPT" ]]; then
  echo "[baseline][stage2] Training baseline nnPU head -> $BASELINE_STAGE2_OUT"
  mkdir -p "$BASELINE_STAGE2_OUT"
  python3 training/finetune_epiaware_grn.py \
    --rna_path "$RNA_DIR/rna.npy" \
    --atac_path "$ATAC_DIR/atac.npy" \
    --candidate_file "$CANDIDATE_JSON" \
    --gene_names "$RNA_DIR/genes.txt" \
    --peak_names "$ATAC_DIR/peaks.txt" \
    --tf_list "$TF_LIST" \
    --positive_links "$POS_LINKS" \
    "${VAL_POS_FLAG[@]}" \
    --backbone_checkpoint "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth" \
    --output_dir "$BASELINE_STAGE2_OUT" \
    --epochs "$BASELINE_HEAD_EPOCHS" \
    "${FAIR_STAGE2_ARGS[@]}" \
    ${BASELINE_STAGE2_ARGS:-}
else
  echo "[baseline][stage2] Reusing $BASELINE_HEAD_CKPT"
fi

# Always ensure baseline Stage2 exports a GRN CSV for downstream evaluation.
if [[ ! -f "$BASELINE_STAGE2_OUT/grn.csv" ]]; then
  echo "[baseline][stage2] Exporting baseline GRN -> $BASELINE_STAGE2_OUT/grn.csv"
  python3 training/finetune_epiaware_grn.py \
    --rna_path "$RNA_DIR/rna.npy" \
    --atac_path "$ATAC_DIR/atac.npy" \
    --candidate_file "$CANDIDATE_JSON" \
    --gene_names "$RNA_DIR/genes.txt" \
    --peak_names "$ATAC_DIR/peaks.txt" \
    --tf_list "$TF_LIST" \
    --positive_links "$POS_LINKS" \
    "${VAL_POS_FLAG[@]}" \
    --backbone_checkpoint "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth" \
    --output_dir "$BASELINE_STAGE2_OUT" \
    --eval_only \
    --head_checkpoint "$BASELINE_HEAD_CKPT" \
    --epochs 0 \
    "${FAIR_STAGE2_ARGS[@]}" \
    --export_logit_path "$BASELINE_STAGE2_OUT/grn_logits.pt" \
    --export_grn_csv "$BASELINE_STAGE2_OUT/grn.csv" \
    ${BASELINE_EXPORT_ARGS:-}
fi
  maybe_run_custom_eval "$BASELINE_STAGE2_OUT/grn.csv"

# ============================================================
# Ablation 1: Multi-omics removal (RNA-only)
# ============================================================
A1_OUT="$OUTPUT_ROOT/ablation_multiomics_rna_only"
A1_STAGE2_OUT="$A1_OUT/stage2"

if [[ ! -f "$STAGE1_RNA_ONLY_OUT/epiaware_backbone_rna_only.pth" ]]; then
  echo "[ablation1][stage1] Training RNA-only backbone -> $STAGE1_RNA_ONLY_OUT"
  python3 training/train_epiaware_backbone_rna_only.py \
    --rna_path "$RNA_DIR/rna.npy" \
    --gene_names "$RNA_DIR/genes.txt" \
    --output_dir "$STAGE1_RNA_ONLY_OUT" \
    --epochs "$STAGE1_RNA_ONLY_EPOCHS" \
    ${STAGE1_RNA_ONLY_ARGS:-}
else
  echo "[ablation1][stage1] Reusing $STAGE1_RNA_ONLY_OUT/epiaware_backbone_rna_only.pth"
fi

if [[ ! -f "$A1_STAGE2_OUT/grn_head.pth" ]]; then
  echo "[ablation1][stage2] Training RNA-only head -> $A1_STAGE2_OUT"
  mkdir -p "$A1_STAGE2_OUT"
  python3 training/finetune_epiaware_grn_rna_only.py \
    --rna_path "$RNA_DIR/rna.npy" \
    --gene_names "$RNA_DIR/genes.txt" \
    --tf_list "$TF_LIST" \
    --positive_links "$POS_LINKS" \
    "${VAL_POS_FLAG[@]}" \
    --backbone_checkpoint "$STAGE1_RNA_ONLY_OUT/epiaware_backbone_rna_only.pth" \
    --output_dir "$A1_STAGE2_OUT" \
    --epochs "$RNA_ONLY_HEAD_EPOCHS" \
    "${FAIR_STAGE2_ARGS[@]}" \
    --export_logit_path "$A1_STAGE2_OUT/grn_logits.pt" \
    --export_grn_csv "$A1_STAGE2_OUT/grn.csv" \
    ${RNA_ONLY_STAGE2_ARGS:-}
else
  echo "[ablation1][stage2] Reusing $A1_STAGE2_OUT/grn_head.pth"
fi
maybe_run_custom_eval "$A1_STAGE2_OUT/grn.csv"

# ============================================================
# Ablation 2: Weak priors (prior vs full vs random) (FAIR)
#   - Trains a separate Stage1 backbone per candidate mode (prior/full/random)
#   - Trains a Stage2 head on top of that backbone
#   - Exports grn.csv/grn_logits.pt for each mode
# ============================================================
A2_OUT="$OUTPUT_ROOT/ablation_weak_priors"
mkdir -p "$A2_OUT"

for MODE in prior random; do
  OUT_DIR="$A2_OUT/$MODE"
  mkdir -p "$OUT_DIR"

  A2_STAGE1_OUT="$SHARED_DIR/stage1_candidates_$MODE"

  CAND_FLAG=()
  if [[ "$MODE" == "prior" ]]; then
    CAND_FLAG=(--candidate_file "$CANDIDATE_JSON")
  elif [[ "$MODE" == "random" ]]; then
    CAND_FLAG=(--candidate_file "$RAND_CANDIDATE_JSON")
  else
    CAND_FLAG=()
  fi

  if [[ "$MODE" == "prior" && -f "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth" ]]; then
    echo "[ablation2][stage1][$MODE] Reusing baseline backbone"
    A2_BACKBONE_CKPT="$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth"
  else
    if [[ ! -f "$A2_STAGE1_OUT/epiaware_backbone.pth" ]]; then
      echo "[ablation2][stage1][$MODE] Training backbone -> $A2_STAGE1_OUT"
      python3 training/train_epiaware_backbone.py \
        --rna_path "$RNA_DIR/rna.npy" \
        --atac_path "$ATAC_DIR/atac.npy" \
        "${CAND_FLAG[@]}" \
        --gene_names "$RNA_DIR/genes.txt" \
        --peak_names "$ATAC_DIR/peaks.txt" \
        --output_dir "$A2_STAGE1_OUT" \
        --epochs "$STAGE1_EPOCHS" \
        ${STAGE1_ARGS:-}
    else
      echo "[ablation2][stage1][$MODE] Reusing $A2_STAGE1_OUT/epiaware_backbone.pth"
    fi
    A2_BACKBONE_CKPT="$A2_STAGE1_OUT/epiaware_backbone.pth"
  fi

  if [[ ! -f "$OUT_DIR/grn_head.pth" ]]; then
    echo "[ablation2][stage2][$MODE] Training head -> $OUT_DIR"
    python3 training/finetune_epiaware_grn.py \
      --rna_path "$RNA_DIR/rna.npy" \
      --atac_path "$ATAC_DIR/atac.npy" \
      "${CAND_FLAG[@]}" \
      --gene_names "$RNA_DIR/genes.txt" \
      --peak_names "$ATAC_DIR/peaks.txt" \
      --tf_list "$TF_LIST" \
      --positive_links "$POS_LINKS" \
      "${VAL_POS_FLAG[@]}" \
      --backbone_checkpoint "$A2_BACKBONE_CKPT" \
      --output_dir "$OUT_DIR" \
      --epochs "$BASELINE_HEAD_EPOCHS" \
      "${FAIR_STAGE2_ARGS[@]}" \
      --export_logit_path "$OUT_DIR/grn_logits.pt" \
      --export_grn_csv "$OUT_DIR/grn.csv" \
      ${WEAK_PRIOR_STAGE2_ARGS:-}
  else
    echo "[ablation2][stage2][$MODE] Reusing $OUT_DIR/grn_head.pth"
  fi
  maybe_run_custom_eval "$OUT_DIR/grn.csv"
done

# Optional: inference-time candidate swap stress test (keeps baseline Stage1+Stage2 fixed)
if [[ "$ENABLE_CANDIDATE_SWAP_STRESS_TEST" != "0" ]]; then
  A2_STRESS_OUT="$OUTPUT_ROOT/stress_candidate_swap"
  mkdir -p "$A2_STRESS_OUT"
  for MODE in prior full random; do
    OUT_DIR="$A2_STRESS_OUT/$MODE"
    mkdir -p "$OUT_DIR"

    CAND_FLAG=()
    if [[ "$MODE" == "prior" ]]; then
      CAND_FLAG=(--candidate_file "$CANDIDATE_JSON")
    elif [[ "$MODE" == "random" ]]; then
      CAND_FLAG=(--candidate_file "$RAND_CANDIDATE_JSON")
    else
      CAND_FLAG=()
    fi

    echo "[stress][candidate_swap][$MODE] Exporting with baseline head -> $OUT_DIR"
    cmd=(
      python3 training/finetune_epiaware_grn.py
      --rna_path "$RNA_DIR/rna.npy"
      --atac_path "$ATAC_DIR/atac.npy"
      "${CAND_FLAG[@]}"
      --gene_names "$RNA_DIR/genes.txt"
      --peak_names "$ATAC_DIR/peaks.txt"
      --tf_list "$TF_LIST"
      --positive_links "$POS_LINKS"
      "${VAL_POS_FLAG[@]}"
      --backbone_checkpoint "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth"
      --output_dir "$OUT_DIR"
      --eval_only
      --head_checkpoint "$BASELINE_HEAD_CKPT"
      --epochs 0
      "${FAIR_STAGE2_ARGS[@]}"
      --export_logit_path "$OUT_DIR/grn_logits.pt"
      --export_grn_csv "$OUT_DIR/grn.csv"
      ${WEAK_PRIOR_EXPORT_ARGS:-}
    )
    if [[ "$PARALLEL_EXPORTS" != "0" ]]; then
      run_job_on_next_gpu "stress_candidate_swap/$MODE" "${cmd[@]}"
    else
      "${cmd[@]}"
    fi
    maybe_run_custom_eval "$OUT_DIR/grn.csv"
  done
fi

# ============================================================
# Ablation 3: PU learning (nnPU baseline vs BCE head)
# ============================================================
A3_OUT="$OUTPUT_ROOT/ablation_pu_learning"
A3_BCE_OUT="$A3_OUT/bce"
mkdir -p "$A3_BCE_OUT"

if [[ ! -f "$A3_BCE_OUT/grn_head.pth" ]]; then
  echo "[ablation3][bce] Training BCE head -> $A3_BCE_OUT"
  python3 training/finetune_epiaware_grn.py \
    --rna_path "$RNA_DIR/rna.npy" \
    --atac_path "$ATAC_DIR/atac.npy" \
    --candidate_file "$CANDIDATE_JSON" \
    --gene_names "$RNA_DIR/genes.txt" \
    --peak_names "$ATAC_DIR/peaks.txt" \
    --tf_list "$TF_LIST" \
    --positive_links "$POS_LINKS" \
    "${VAL_POS_FLAG[@]}" \
    --backbone_checkpoint "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth" \
    --output_dir "$A3_BCE_OUT" \
    --use_bce_loss \
    --epochs "$BCE_HEAD_EPOCHS" \
    "${FAIR_STAGE2_ARGS[@]}" \
    --export_logit_path "$A3_BCE_OUT/grn_logits.pt" \
    --export_grn_csv "$A3_BCE_OUT/grn.csv" \
    ${BCE_STAGE2_ARGS:-}
else
  echo "[ablation3][bce] Reusing $A3_BCE_OUT/grn_head.pth"
fi
maybe_run_custom_eval "$A3_BCE_OUT/grn.csv"

A3_PU_EXPORT_OUT="$A3_OUT/nnpu_export"
mkdir -p "$A3_PU_EXPORT_OUT"
echo "[ablation3][nnpu] Exporting baseline nnPU head -> $A3_PU_EXPORT_OUT"
python3 training/finetune_epiaware_grn.py \
  --rna_path "$RNA_DIR/rna.npy" \
  --atac_path "$ATAC_DIR/atac.npy" \
  --candidate_file "$CANDIDATE_JSON" \
  --gene_names "$RNA_DIR/genes.txt" \
  --peak_names "$ATAC_DIR/peaks.txt" \
  --tf_list "$TF_LIST" \
  --positive_links "$POS_LINKS" \
  "${VAL_POS_FLAG[@]}" \
  --backbone_checkpoint "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth" \
  --output_dir "$A3_PU_EXPORT_OUT" \
  --eval_only \
  --head_checkpoint "$BASELINE_HEAD_CKPT" \
  --epochs 0 \
  "${FAIR_STAGE2_ARGS[@]}" \
  --export_logit_path "$A3_PU_EXPORT_OUT/grn_logits.pt" \
  --export_grn_csv "$A3_PU_EXPORT_OUT/grn.csv" \
  ${PU_EXPORT_ARGS:-}

maybe_run_custom_eval "$A3_PU_EXPORT_OUT/grn.csv"

# ============================================================
# Ablation 4: Stage1 Top-K routing (cross-attention) (vary K)
#   - K<=0 means no routing: attend all candidate peaks.
#   - Trains a separate backbone (and head) per K for fair comparison.
# ============================================================
A4_OUT="$OUTPUT_ROOT/ablation_stage1_topk_routing"
mkdir -p "$A4_OUT"

for K in $TOPK_STAGE1_LIST; do
  A4_K_OUT="$A4_OUT/topk_stage1_${K}"
  A4_K_STAGE1_OUT="$SHARED_DIR/stage1_topk_stage1_${K}"
  A4_K_STAGE2_OUT="$A4_K_OUT/stage2"
  mkdir -p "$A4_K_OUT"

  if [[ "$K" == "8" && -f "$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth" ]]; then
    echo "[ablation4][stage1][topk=$K] Reusing baseline backbone (topk=8)"
    A4_BACKBONE_CKPT="$STAGE1_MULTIOMIC_OUT/epiaware_backbone.pth"
  else
    if [[ ! -f "$A4_K_STAGE1_OUT/epiaware_backbone.pth" ]]; then
      echo "[ablation4][stage1][topk=$K] Training backbone -> $A4_K_STAGE1_OUT"
      python3 training/train_epiaware_backbone.py \
        --rna_path "$RNA_DIR/rna.npy" \
        --atac_path "$ATAC_DIR/atac.npy" \
        --candidate_file "$CANDIDATE_JSON" \
        --gene_names "$RNA_DIR/genes.txt" \
        --peak_names "$ATAC_DIR/peaks.txt" \
        --output_dir "$A4_K_STAGE1_OUT" \
        --epochs "$STAGE1_EPOCHS" \
        --topk_peaks "$K" \
        ${STAGE1_ARGS:-}
    else
      echo "[ablation4][stage1][topk=$K] Reusing $A4_K_STAGE1_OUT/epiaware_backbone.pth"
    fi
    A4_BACKBONE_CKPT="$A4_K_STAGE1_OUT/epiaware_backbone.pth"
  fi

  if [[ ! -f "$A4_K_STAGE2_OUT/grn_head.pth" ]]; then
    echo "[ablation4][stage2][topk=$K] Training head -> $A4_K_STAGE2_OUT"
    mkdir -p "$A4_K_STAGE2_OUT"
    python3 training/finetune_epiaware_grn.py \
      --rna_path "$RNA_DIR/rna.npy" \
      --atac_path "$ATAC_DIR/atac.npy" \
      --candidate_file "$CANDIDATE_JSON" \
      --gene_names "$RNA_DIR/genes.txt" \
      --peak_names "$ATAC_DIR/peaks.txt" \
      --tf_list "$TF_LIST" \
      --positive_links "$POS_LINKS" \
      "${VAL_POS_FLAG[@]}" \
      --backbone_checkpoint "$A4_BACKBONE_CKPT" \
      --output_dir "$A4_K_STAGE2_OUT" \
      --epochs "$TOPK_STAGE1_HEAD_EPOCHS" \
      "${FAIR_STAGE2_ARGS[@]}" \
      --export_logit_path "$A4_K_STAGE2_OUT/grn_logits.pt" \
      --export_grn_csv "$A4_K_STAGE2_OUT/grn.csv" \
      ${TOPK_STAGE1_STAGE2_ARGS:-}
  else
    echo "[ablation4][stage2][topk=$K] Reusing $A4_K_STAGE2_OUT/grn_head.pth"
  fi
  maybe_run_custom_eval "$A4_K_STAGE2_OUT/grn.csv"
done

# Safety net: if enabled, sweep any remaining grn.csv that were exported but not evaluated.
if [[ "$RUN_CUSTOM_EVAL" != "0" ]]; then
  mapfile -t GRN_FILES < <(find "$OUTPUT_ROOT" -type f -name 'grn.csv' | sort)
  for grn in "${GRN_FILES[@]}"; do
    maybe_run_custom_eval "$grn"
  done
fi

echo "[done] Outputs stored under $OUTPUT_ROOT"
