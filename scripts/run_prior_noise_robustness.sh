#!/usr/bin/env bash
set -euo pipefail

# Launches Stage 2 experiments that corrupt bulk priors (drop/flip) and measures
# robustness for both the full EpiAwareNet (multi-omic) and RNA-only variants.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASETS=${DATASETS:-"pN"}
NOISE_TYPES=${NOISE_TYPES:-"drop"}
NOISE_LEVELS=${NOISE_LEVELS:-"0.2 0.4 0.6 0.8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/output/prior_noise_robustness_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTPUT_ROOT"

# Toggle RNA-only branch (disable for faster/simpler runs)
RUN_RNA_ONLY=${RUN_RNA_ONLY:-0}

# Stage 1/2 hyperparameters (override via env as needed)
STAGE1_EPOCHS=${STAGE1_EPOCHS:-2}
STAGE1_RNA_ONLY_EPOCHS=${STAGE1_RNA_ONLY_EPOCHS:-2}
EPI_HEAD_EPOCHS=${EPI_HEAD_EPOCHS:-1}
RNA_ONLY_HEAD_EPOCHS=${RNA_ONLY_HEAD_EPOCHS:-1}

CELL_BATCH_SIZE=${CELL_BATCH_SIZE:-32}
POS_BATCH_SIZE=${POS_BATCH_SIZE:-256}
UNL_BATCH_SIZE=${UNL_BATCH_SIZE:-1024}
VAL_PROB_THRESHOLD=${VAL_PROB_THRESHOLD:-0.5}
VAL_FRACTION=${VAL_FRACTION:-0.15}
POSITIVE_PRIOR=${POSITIVE_PRIOR:-0.05}
NEGATIVE_RISK_WEIGHT=${NEGATIVE_RISK_WEIGHT:-1.0}
NUM_WORKERS=${NUM_WORKERS:-4}

FORCE_RERUN=${FORCE_RERUN:-0}
PRIOR_NOISE_SEED=${PRIOR_NOISE_SEED:-123}

RUN_CUSTOM_EVAL=${RUN_CUSTOM_EVAL:-1}
FORCE_CUSTOM_EVAL=${FORCE_CUSTOM_EVAL:-0}
CUSTOM_EVAL_PRECISION_TARGETS=${CUSTOM_EVAL_PRECISION_TARGETS:-"0.20 0.25 0.30 0.35"}
CUSTOM_EVAL_GOLD_mN=${CUSTOM_EVAL_GOLD_mN:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell.txt"}
CUSTOM_EVAL_GOLD_pN=${CUSTOM_EVAL_GOLD_pN:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell.txt"}

PRETRAINED_STAGE1_MULTIOMIC_mN=${PRETRAINED_STAGE1_MULTIOMIC_mN:-"$REPO_ROOT/output/mN_pipeline/shared/stage1_multiomic"}
PRETRAINED_STAGE1_RNA_ONLY_mN=${PRETRAINED_STAGE1_RNA_ONLY_mN:-"$REPO_ROOT/output/mN_pipeline/shared/stage1_rna_only"}
PRETRAINED_STAGE1_MULTIOMIC_pN=${PRETRAINED_STAGE1_MULTIOMIC_pN:-"$REPO_ROOT/output/pN_pipeline/shared/stage1_multiomic"}
PRETRAINED_STAGE1_RNA_ONLY_pN=${PRETRAINED_STAGE1_RNA_ONLY_pN:-"$REPO_ROOT/output/pN_pipeline/shared/stage1_rna_only"}

# GPU parallelism knobs
GPUS=${GPUS:-""}      # e.g. "0 1 2 3"
MAX_JOBS=${MAX_JOBS:-0}

GPU_LIST=()
JOB_PIDS=()
NEXT_GPU_IDX=0
declare -a EVAL_TARGETS=()

init_gpus() {
  if [[ -n "$GPUS" ]]; then
    read -r -a GPU_LIST <<< "$GPUS"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --list-gpus | awk -F: '{print $1}' | awk '{print $2}')
  else
    GPU_LIST=(0)
  fi
  if ((${#GPU_LIST[@]} == 0)); then
    GPU_LIST=(0)
  fi
  if ((MAX_JOBS <= 0)); then
    MAX_JOBS=${#GPU_LIST[@]}
  fi
  echo "[gpu] Using devices: ${GPU_LIST[*]} (MAX_JOBS=$MAX_JOBS)"
}

prune_jobs() {
  local -a keep=()
  local pid
  for pid in "${JOB_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      keep+=("$pid")
    fi
  done
  JOB_PIDS=("${keep[@]}")
}

wait_for_slot() {
  while true; do
    prune_jobs
    if ((${#JOB_PIDS[@]} < MAX_JOBS)); then
      return 0
    fi
    sleep 2
  done
}

launch_async() {
  local label="$1"
  shift
  wait_for_slot
  local gpu="${GPU_LIST[$NEXT_GPU_IDX]}"
  NEXT_GPU_IDX=$(( (NEXT_GPU_IDX + 1) % ${#GPU_LIST[@]} ))
  echo "[launch][gpu=$gpu] $label"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@"
  ) &
  JOB_PIDS+=("$!")
}

finish_jobs() {
  if ((${#JOB_PIDS[@]} == 0)); then
    return 0
  fi
  echo "[gpu] Waiting for ${#JOB_PIDS[@]} job(s)..."
  wait "${JOB_PIDS[@]}"
}

fmt_pct() {
  local value="$1"
  python3 - "$value" <<'PY'
import sys
val = float(sys.argv[1])
print(int(round(val * 100)))
PY
}

copy_pretrained_stage1() {
  local src_dir="$1" dest_dir="$2" marker="$3" label="$4"
  if [[ -z "$src_dir" || ! -d "$src_dir" ]]; then
    return 1
  fi
  if [[ ! -f "$src_dir/$marker" ]]; then
    return 1
  fi
  echo "[stage1][reuse] Copying $label from $src_dir -> $dest_dir"
  mkdir -p "$dest_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src_dir"/ "$dest_dir"/
  else
    cp -a "$src_dir"/. "$dest_dir"/
  fi
  return 0
}

ensure_stage1_multiomic() {
  local dataset_slug="$1" rna_dir="$2" atac_dir="$3" candidate_json="$4" out_dir="$5" pretrained_src="$6"
  if [[ -f "$out_dir/epiaware_backbone.pth" ]]; then
    echo "[stage1][${dataset_slug}] Reusing $out_dir/epiaware_backbone.pth"
    return 0
  fi
  if copy_pretrained_stage1 "$pretrained_src" "$out_dir" "epiaware_backbone.pth" "${dataset_slug} multi-omic backbone"; then
    return 0
  fi
  echo "[stage1][${dataset_slug}] Training multi-omic backbone -> $out_dir"
  mkdir -p "$out_dir"
  python3 training/train_epiaware_backbone.py \
    --rna_path "$rna_dir/rna.npy" \
    --atac_path "$atac_dir/atac.npy" \
    --candidate_file "$candidate_json" \
    --gene_names "$rna_dir/genes.txt" \
    --peak_names "$atac_dir/peaks.txt" \
    --output_dir "$out_dir" \
    --epochs "$STAGE1_EPOCHS"
}

ensure_stage1_rna_only() {
  local dataset_slug="$1" rna_dir="$2" out_dir="$3" pretrained_src="$4"
  if [[ -f "$out_dir/epiaware_backbone_rna_only.pth" ]]; then
    echo "[stage1][${dataset_slug}][rna_only] Reusing $out_dir/epiaware_backbone_rna_only.pth"
    return 0
  fi
  if copy_pretrained_stage1 "$pretrained_src" "$out_dir" "epiaware_backbone_rna_only.pth" "${dataset_slug} RNA-only backbone"; then
    return 0
  fi
  echo "[stage1][${dataset_slug}][rna_only] Training RNA-only backbone -> $out_dir"
  mkdir -p "$out_dir"
  python3 training/train_epiaware_backbone_rna_only.py \
    --rna_path "$rna_dir/rna.npy" \
    --gene_names "$rna_dir/genes.txt" \
    --output_dir "$out_dir" \
    --epochs "$STAGE1_RNA_ONLY_EPOCHS"
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

make_prior_variant() {
  local noise_type="$1" level="$2" base_links="$3" genes_txt="$4" tf_list="$5" out_file="$6"
  local metadata="${out_file%.tsv}.json"
  if [[ -f "$out_file" && "$FORCE_RERUN" == "0" ]]; then
    return 0
  fi
  local drop_frac="0.0"
  local flip_frac="0.0"
  if [[ "$noise_type" == "drop" ]]; then
    drop_frac="$level"
  elif [[ "$noise_type" == "flip" ]]; then
    flip_frac="$level"
  else
    echo "[prior] Unknown noise type: $noise_type" >&2
    exit 1
  fi
  echo "[prior] Building $noise_type@$level -> $out_file"
  python3 scripts/perturb_positive_links.py \
    --positive_links "$base_links" \
    --genes_txt "$genes_txt" \
    --tf_list "$tf_list" \
    --output "$out_file" \
    --metadata_out "$metadata" \
    --drop_frac "$drop_frac" \
    --flip_frac "$flip_frac" \
    --seed "$PRIOR_NOISE_SEED"
}

init_gpus

for DATASET in $DATASETS; do
  PRETRAINED_MULTI_SRC=""
  PRETRAINED_RNA_SRC=""
  case "$DATASET" in
    mN)
      RNA_DIR=${RNA_DIR_mN:-"$REPO_ROOT/datasets/mN_combined"}
      ATAC_DIR=${ATAC_DIR_mN:-"$REPO_ROOT/datasets/mN_combined_ATAC"}
      CANDIDATE_JSON=${CANDIDATE_JSON_mN:-"$RNA_DIR/candidate_peaks.json"}
      TF_LIST=${TF_LIST_mN:-"$REPO_ROOT/datasets/plusNshare/regulators.txt"}
      POS_LINKS=${POS_LINKS_mN:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_train.txt"}
      VAL_POS_LINKS=${VAL_POS_LINKS_mN:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_val.txt"}
      CUSTOM_EVAL_GOLD_CUR="$CUSTOM_EVAL_GOLD_mN"
      PRETRAINED_MULTI_SRC="$PRETRAINED_STAGE1_MULTIOMIC_mN"
      PRETRAINED_RNA_SRC="$PRETRAINED_STAGE1_RNA_ONLY_mN"
      ;;
    pN)
      RNA_DIR=${RNA_DIR_pN:-"$REPO_ROOT/datasets/pN_combined"}
      ATAC_DIR=${ATAC_DIR_pN:-"$REPO_ROOT/datasets/pN_combined_ATAC"}
      CANDIDATE_JSON=${CANDIDATE_JSON_pN:-"$RNA_DIR/candidate_peaks.json"}
      TF_LIST=${TF_LIST_pN:-"$REPO_ROOT/datasets/plusNshare/regulators.txt"}
      POS_LINKS=${POS_LINKS_pN:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_train.txt"}
      VAL_POS_LINKS=${VAL_POS_LINKS_pN:-"$REPO_ROOT/datasets/TF_Target_Validation_inSingleCell_val.txt"}
      CUSTOM_EVAL_GOLD_CUR="$CUSTOM_EVAL_GOLD_pN"
      PRETRAINED_MULTI_SRC="$PRETRAINED_STAGE1_MULTIOMIC_pN"
      PRETRAINED_RNA_SRC="$PRETRAINED_STAGE1_RNA_ONLY_pN"
      ;;
    *)
      echo "[error] Unsupported dataset: $DATASET" >&2
      exit 1
      ;;
  esac

  DATASET_OUT="$OUTPUT_ROOT/$DATASET"
  SHARED_DIR="$DATASET_OUT/shared"
  PRIORS_DIR="$DATASET_OUT/priors"
  mkdir -p "$SHARED_DIR" "$PRIORS_DIR"

  STAGE1_MULTI="$SHARED_DIR/stage1_multiomic"
  STAGE1_RNA_ONLY="$SHARED_DIR/stage1_rna_only"
  CUSTOM_EVAL_GENES_CUR="$RNA_DIR/genes.txt"

  ensure_stage1_multiomic "$DATASET" "$RNA_DIR" "$ATAC_DIR" "$CANDIDATE_JSON" "$STAGE1_MULTI" "$PRETRAINED_MULTI_SRC"
  if [[ "$RUN_RNA_ONLY" == "1" ]]; then
    ensure_stage1_rna_only "$DATASET" "$RNA_DIR" "$STAGE1_RNA_ONLY" "$PRETRAINED_RNA_SRC"
  fi

  VAL_POS_FLAG=()
  if [[ -n "$VAL_POS_LINKS" && -f "$VAL_POS_LINKS" ]]; then
    VAL_POS_FLAG=(--val_positive_links "$VAL_POS_LINKS")
  fi

  for NOISE in $NOISE_TYPES; do
    for LEVEL in $NOISE_LEVELS; do
      level_pct=$(fmt_pct "$LEVEL")
      level_slug=$(printf "%02d" "$level_pct")
      label="${NOISE}_${level_slug}"
      PRIOR_FILE="$PRIORS_DIR/${label}.tsv"
      make_prior_variant "$NOISE" "$LEVEL" "$POS_LINKS" "$RNA_DIR/genes.txt" "$TF_LIST" "$PRIOR_FILE"

      # Stage2 output dirs
      EPI_OUT="$DATASET_OUT/epi/${label}"
      RNA_OUT="$DATASET_OUT/rna_only/${label}"
      mkdir -p "$EPI_OUT"
      if [[ "$RUN_RNA_ONLY" == "1" ]]; then
        mkdir -p "$RNA_OUT"
      fi

      maybe_launch_epi=1
      if [[ -f "$EPI_OUT/grn_head.pth" && "$FORCE_RERUN" == "0" ]]; then
        echo "[skip][${DATASET}][${label}] EpiAwareNet head exists"
        maybe_launch_epi=0
      fi
      if (( maybe_launch_epi )); then
        launch_async \
          "dataset=$DATASET model=epi label=$label" \
          python3 training/finetune_epiaware_grn.py \
            --rna_path "$RNA_DIR/rna.npy" \
            --atac_path "$ATAC_DIR/atac.npy" \
            --candidate_file "$CANDIDATE_JSON" \
            --gene_names "$RNA_DIR/genes.txt" \
            --peak_names "$ATAC_DIR/peaks.txt" \
            --tf_list "$TF_LIST" \
            --positive_links "$PRIOR_FILE" \
            "${VAL_POS_FLAG[@]}" \
            --backbone_checkpoint "$STAGE1_MULTI/epiaware_backbone.pth" \
            --output_dir "$EPI_OUT" \
            --epochs "$EPI_HEAD_EPOCHS" \
            --use_bce_loss \
            "${FAIR_STAGE2_ARGS[@]}" \
            --export_logit_path "$EPI_OUT/grn_logits.pt" \
            --export_grn_csv "$EPI_OUT/grn.csv"
      fi

      maybe_launch_rna=0
      if [[ "$RUN_RNA_ONLY" == "1" ]]; then
        maybe_launch_rna=1
        if [[ -f "$RNA_OUT/grn_head.pth" && "$FORCE_RERUN" == "0" ]]; then
          echo "[skip][${DATASET}][${label}] RNA-only head exists"
          maybe_launch_rna=0
        fi
        if (( maybe_launch_rna )); then
          launch_async \
            "dataset=$DATASET model=rna_only label=$label" \
            python3 training/finetune_epiaware_grn_rna_only.py \
              --rna_path "$RNA_DIR/rna.npy" \
              --gene_names "$RNA_DIR/genes.txt" \
              --tf_list "$TF_LIST" \
              --positive_links "$PRIOR_FILE" \
              "${VAL_POS_FLAG[@]}" \
              --backbone_checkpoint "$STAGE1_RNA_ONLY/epiaware_backbone_rna_only.pth" \
              --output_dir "$RNA_OUT" \
              --epochs "$RNA_ONLY_HEAD_EPOCHS" \
              --use_bce_loss \
              "${FAIR_STAGE2_ARGS[@]}" \
              --export_logit_path "$RNA_OUT/grn_logits.pt" \
              --export_grn_csv "$RNA_OUT/grn.csv"
        fi
      fi
      EVAL_TARGETS+=("$EPI_OUT/grn.csv|$CUSTOM_EVAL_GOLD_CUR|$CUSTOM_EVAL_GENES_CUR")
      if [[ "$RUN_RNA_ONLY" == "1" ]]; then
        EVAL_TARGETS+=("$RNA_OUT/grn.csv|$CUSTOM_EVAL_GOLD_CUR|$CUSTOM_EVAL_GENES_CUR")
      fi
    done
  done
done

finish_jobs

maybe_run_eval() {
  local predictions="$1"
  local gold="$2"
  local genes="$3"
  if [[ "$RUN_CUSTOM_EVAL" == "0" ]]; then
    return 0
  fi
  if [[ ! -f "$predictions" ]]; then
    echo "[eval][warn] Missing predictions: $predictions"
    return 0
  fi
  local outdir
  outdir="$(dirname "$predictions")/custom_eval"
  if [[ "$FORCE_CUSTOM_EVAL" == "0" && -f "$outdir/metrics.json" ]]; then
    echo "[eval] Skip (exists): $outdir/metrics.json"
    return 0
  fi
  mkdir -p "$outdir"
  echo "[eval] $predictions -> $outdir"
  python3 custom_evaluate.py \
    --predictions "$predictions" \
    --gold "$gold" \
    --genes "$genes" \
    --output_dir "$outdir" \
    --precision_targets $CUSTOM_EVAL_PRECISION_TARGETS
}

for entry in "${EVAL_TARGETS[@]}"; do
  IFS='|' read -r pred gold genes <<< "$entry"
  maybe_run_eval "$pred" "$gold" "$genes"
done

echo "[done] Prior-noise robustness jobs queued under $OUTPUT_ROOT"
