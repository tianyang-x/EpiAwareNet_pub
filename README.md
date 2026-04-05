# EpiAwareNet

Epigenome-aware Transformer for gene regulatory network (GRN) inference from single-cell multi-omics data.

## Overview

EpiAwareNet is a two-stage deep learning framework that integrates single-cell RNA-seq and ATAC-seq to infer context-specific gene regulatory networks:

1. **Stage 1 -- Representation Learning.** A dual-path Transformer backbone fuses RNA-seq and ATAC-seq features with gene-gene sparse self-attention and candidate-aware gene-peak cross-attention. Training uses a masked negative binomial reconstruction objective.
2. **Stage 2 -- GRN Fine-Tuning.** A lightweight prediction head performs positive-unlabeled (nnPU) learning on top of the frozen backbone to infer TF-to-gene regulatory edges supervised by bulk positive priors.
3. **Context-Specific GRNs.** After fine-tuning, the backbone+head stack scores TF-gene pairs per cell and aggregates probabilities across any cell subset to obtain context-specific adjacency matrices.

## Installation

```bash
git clone https://github.com/tianyang-x/EpiAwareNet_pub.git
cd EpiAwareNet_pub
pip install -r requirements.txt
```

**Optional:** Install [faiss-gpu](https://github.com/facebookresearch/faiss) for faster kNN computation in gene self-attention:
```bash
conda install -c conda-forge faiss-gpu
```

## Repository Structure

```
epiaware/                  # Core library
  data/                    # Dataset loaders, candidate sets, PU sampler
  models/                  # Transformer backbone, attention, GRN head
training/                  # Stage 1 & Stage 2 training scripts
evaluate/                  # AUPRC/AUROC metrics, p-value testing
scripts/                   # Entry-point shell scripts & utilities
custom_evaluate.py         # Main evaluation orchestrator
```

## Data Requirements

| Input | Format |
| --- | --- |
| `rna_path`, `atac_path` | `.npy`, `.npz`, `.pt`, or `.h5ad` matrices (N cells x features) |
| `gene_names`, `peak_names` | Optional TXT/JSON lists if not embedded in matrices |
| `candidate_file` | JSON/TSV mapping each gene to candidate peak indices (or names) |
| `tf_list` | TXT/JSON list of known TF symbols contained in `gene_names` |
| `positive_links` | CSV/TSV/JSON with `tf` and `target` columns for bulk positive pairs |

### Preparing 10x Datasets

Convert 10x-format matrices to dense `.npy` files:

```bash
python scripts/convert_10x_to_epiaware.py \
  --rna_dir datasets/pN_combined \
  --atac_dir datasets/pN_combined_ATAC
```

Build the candidate peak JSON:

```bash
python scripts/prepare_candidate_set.py \
  --input datasets/plusNshare/gene_peak_10k.txt \
  --output datasets/pN_combined/candidate_peaks.json \
  --peaks_txt datasets/pN_combined_ATAC/peaks.txt \
  --max_peaks_per_gene 50
```

For 10x Multiome data (single MTX with both Gene Expression and Peaks):

```bash
python scripts/preprocess_10x_multiome_dir.py \
  --multiome_dir datasets/mouse_brain/filtered_feature_bc_matrix \
  --rna_dir datasets/mouse_brain_combined \
  --atac_dir datasets/mouse_brain_combined_ATAC \
  --gene_peaks datasets/mouse_brain/gene_peaks_5kb.txt \
  --max_peaks_per_gene 50
```

## Training

### Stage 1: Backbone Pretraining

```bash
python training/train_epiaware_backbone.py \
  --rna_path datasets/pN_combined/rna.npy \
  --atac_path datasets/pN_combined/atac.npy \
  --candidate_file datasets/pN_combined/candidate_peaks.json \
  --gene_names datasets/pN_combined/genes.txt \
  --peak_names datasets/pN_combined/peaks.txt \
  --output_dir output/epiaware_backbone \
  --epochs 50 --batch_size 16 --mask_ratio 0.15
```

Key flags: `--use_faiss_knn` (FAISS-backed kNN), `--embed_dim`, `--num_heads`, `--topk_peaks`, `--depth`.

### Stage 2: nnPU Fine-Tuning

```bash
python training/finetune_epiaware_grn.py \
  --rna_path datasets/pN_combined/rna.npy \
  --atac_path datasets/pN_combined/atac.npy \
  --candidate_file datasets/pN_combined/candidate_peaks.json \
  --gene_names datasets/pN_combined/genes.txt \
  --peak_names datasets/pN_combined/peaks.txt \
  --tf_list datasets/regulators.txt \
  --positive_links datasets/TF_Target_train.txt \
  --backbone_checkpoint output/epiaware_backbone/epiaware_backbone.pth \
  --output_dir output/epiaware_grn \
  --positive_prior 0.08 --pos_batch_size 512 --unl_batch_size 4096
```

### Preconfigured Pipelines

Run both stages end-to-end for specific datasets:

```bash
bash scripts/run_pN_pipeline.sh    # pN condition
bash scripts/run_mN_pipeline.sh    # mN condition
bash scripts/run_pbmc_pipeline.sh  # PBMC
```

Override dataset paths via environment variables (`RNA_DIR`, `ATAC_DIR`, `TF_LIST`, `POS_LINKS`, `OUTPUT_ROOT`). Customize hyperparameters with `STAGE1_ARGS` and `STAGE2_ARGS`.

### Ablation Studies

```bash
bash scripts/run_pN_ablations.sh   # pN ablations
bash scripts/run_mN_ablations.sh   # mN ablations
```

These scripts run the full ablation suite: multi-omic vs. RNA-only backbone, nnPU vs. BCE loss, cross-attention top-k variants, and prior noise robustness.

## Evaluation

Compute AUPRC/AUROC and plot PR/ROC curves:

```bash
python evaluate/epiaware_metrics.py \
  --predictions output/epiaware_grn/grn.csv \
  --gold datasets/TF_Target_Validation.txt \
  --output_dir output/metrics
```

For comprehensive evaluation with threshold analysis and p-values:

```bash
python custom_evaluate.py \
  --predictions output/epiaware_grn/grn.csv \
  --gold datasets/TF_Target_Validation.txt \
  --genes datasets/pN_combined/genes.txt \
  --output_dir output/custom_eval
```

## Scaling Benchmarks

```bash
bash scripts/run_scaling_benchmark.sh       # Gene/peak scaling
bash scripts/run_cell_scaling_benchmark.sh  # Cell count scaling
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use EpiAwareNet in your research, please cite:

```bibtex
@software{xu2025epiawarenet,
  title={EpiAwareNet: Epigenome-aware Transformer for Gene Regulatory Network Inference from Single-Cell Multi-Omics Data},
  author={Xu, Tianyang},
  year={2025},
  url={https://github.com/tianyang-x/EpiAwareNet_pub}
}
```
