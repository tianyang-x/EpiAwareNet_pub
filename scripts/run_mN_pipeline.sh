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
OUTPUT_ROOT=${OUTPUT_ROOT:-output/mN_pipeline_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUTPUT_ROOT"

STAGE1_OUT="$OUTPUT_ROOT/backbone"
STAGE2_OUT="$OUTPUT_ROOT/grn_head"

echo "[stage1] Training backbone -> $STAGE1_OUT"
python3 training/train_epiaware_backbone.py \
  --rna_path "$RNA_DIR/rna.npy" \
  --atac_path "$ATAC_DIR/atac.npy" \
  --candidate_file "$CANDIDATE_JSON" \
  --gene_names "$RNA_DIR/genes.txt" \
  --peak_names "$ATAC_DIR/peaks.txt" \
  --output_dir "$STAGE1_OUT" \
  ${STAGE1_ARGS:-}

echo "[stage2] Fine-tuning GRN head -> $STAGE2_OUT"
VAL_POS_FLAG=()
if [[ -n "${VAL_POS_LINKS:-}" ]]; then
  VAL_POS_FLAG=(--val_positive_links "$VAL_POS_LINKS")
fi
python3 training/finetune_epiaware_grn.py \
  --rna_path "$RNA_DIR/rna.npy" \
  --atac_path "$ATAC_DIR/atac.npy" \
  --candidate_file "$CANDIDATE_JSON" \
  --gene_names "$RNA_DIR/genes.txt" \
  --peak_names "$ATAC_DIR/peaks.txt" \
  --tf_list "$TF_LIST" \
  --positive_links "$POS_LINKS" \
  "${VAL_POS_FLAG[@]}" \
  --backbone_checkpoint "$STAGE1_OUT/epiaware_backbone.pth" \
  --output_dir "$STAGE2_OUT" \
  ${STAGE2_ARGS:-}

echo "[done] Outputs stored under $OUTPUT_ROOT"
