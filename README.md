# EpiAwareNet

1. **Stage 1 – Representation Learning.** A dual-path Transformer backbone fuses RNA-seq and ATAC-seq features with gene-gene sparse attention and candidate-aware gene-peak cross-attention. Training uses a masked negative binomial reconstruction objective so the model must leverage chromatin context to impute masked gene expressions.
2. **Stage 2 – GRN Fine-Tuning.** A lightweight prediction head performs positive-unlabeled (nnPU) learning on top of the frozen backbone to infer TF→gene edges supervised by bulk positive priors.
3. **Context-Specific GRNs.** After fine-tuning, the backbone+head stack can score TF-gene pairs per cell and aggregate probabilities across any cell subset to obtain context-specific adjacency matrices.

## Repository Additions

| Component | Description |
| --- | --- |
| `epiaware/data/multiomic_dataset.py` | Memory-mapped dataset loader for paired RNA/ATAC matrices plus candidate set utilities. |
| `epiaware/data/pu_links.py` | Positive/unlabeled sampler with robust TF/link parsers. |
| `epiaware/models/backbone.py` | Implements embeddings, Transformer blocks, masked NB head, and training helpers. |
| `epiaware/models/attention.py` | Custom sparse self-attention and masked cross-attention modules. |
| `epiaware/models/grn_head.py` | nnPU-ready GRN prediction head with hinge loss. |
| `training/train_epiaware_backbone.py` | Stage 1 training entrypoint. |
| `training/finetune_epiaware_grn.py` | Stage 2 nnPU fine-tuning script. |
| `training/run_epiaware_pipeline.sh` | Convenience script that chains both stages. |

## Data Requirements

| Input | Format |
| --- | --- |
| `rna_path`, `atac_path` | `.npy`, `.npz`, `.pt`, or `.h5ad` matrices with shape `N cells × features`. |
| `gene_names`, `peak_names` | Optional TXT/JSON lists if not embedded in matrices. |
| `candidate_file` | JSON/TSV mapping each gene to a list of candidate peak indices (or names). |
| `tf_list` | TXT/JSON list of known TF symbols contained in `gene_names`. |
| `positive_links` | CSV/TSV/JSON with `tf` and `target` columns recording bulk positive pairs. |

## Stage 1: Pretraining

### Preparing 10x Datasets

For directories such as `datasets/pN_combined` (RNA) and `datasets/pN_combined_ATAC` (ATAC), convert the 10x-format matrices into the dense `.npy` + metadata files used by the training scripts:

```bash
python scripts/convert_10x_to_epiaware.py \
  --rna_dir datasets/pN_combined \
  --atac_dir datasets/pN_combined_ATAC \
  --chunk_size 512
```

This creates `rna.npy`, `atac.npy`, `genes.txt`, `peaks.txt`, and `cells.txt` inside the respective directories, and rewrites `features.tsv.gz` into the canonical three-column layout when necessary (e.g., for ATAC peaks).

Next, convert the `gene_peak_10k.txt` table into the JSON candidate format:

```bash
python scripts/prepare_candidate_set.py \
  --input datasets/plusNshare/gene_peak_10k.txt \
  --output datasets/pN_combined/candidate_peaks.json \
  --peaks_txt datasets/pN_combined_ATAC/peaks.txt \
  --max_peaks_per_gene 50
```

This filters the peak names against the ATAC vocabulary and keeps the top-N peaks per gene (default 50).

```bash
python training/train_epiaware_backbone.py \
  --rna_path datasets/pN_combined/rna.npy \
  --atac_path datasets/pN_combined/atac.npy \
  --candidate_file datasets/pN_combined/candidate_peaks.json \
  --gene_names datasets/pN_combined/genes.txt \
  --peak_names datasets/pN_combined/peaks.txt \
  --output_dir output/epiaware_backbone \
  --epochs 50 \
  --batch_size 16 \
  --mask_ratio 0.15

```

### Preparing 10x *Multiome* (Gene Expression + Peaks in one MTX)

Some 10x releases provide a single `filtered_feature_bc_matrix/` directory that mixes
`Gene Expression` and `Peaks` features in the same `matrix.mtx.gz`. For these inputs,
use the multiome preprocessor to split the modalities and (optionally) build the
candidate set.

Example for this repo's mouse brain dataset:

```bash
python scripts/preprocess_10x_multiome_dir.py \
  --multiome_dir datasets/mouse_brain/filtered_feature_bc_matrix \
  --rna_dir datasets/mouse_brain_combined \
  --atac_dir datasets/mouse_brain_combined_ATAC \
  --gene_peaks datasets/mouse_brain/gene_peaks_5kb.txt \
  --max_peaks_per_gene 50
```

Then run the ablation suite by overriding the dataset paths:

```bash
RNA_DIR=datasets/mouse_brain_combined \
ATAC_DIR=datasets/mouse_brain_combined_ATAC \
CANDIDATE_JSON=datasets/mouse_brain_combined/candidate_peaks.json \
TF_LIST=datasets/mouse_brain_combined/tfs_trrust_mouse.txt \
POS_LINKS=datasets/mouse_brain_combined/positive_links_trrust_mouse_train_5064.tsv \
VAL_POS_LINKS=datasets/mouse_brain_combined/positive_links_trrust_mouse_val_1266.tsv \
bash scripts/run_mN_ablations.sh
```

Key flags:

- `--use_faiss_knn` enables FAISS-backed kNN selection for gene self-attention when the dependency is installed.
- `--mask_ratio` controls the 15% gene masking rate described in the paper.
- `--embed_dim`, `--num_heads`, `--topk_peaks`, etc. mirror the architecture parameters in the method section.

Outputs include per-epoch checkpoints, a rolling `training_metrics.csv` (epoch vs. reconstruction loss), and `epiaware_backbone.pth` containing the full pretraining model state.

## Stage 2: nnPU Fine-Tuning

```bash
python training/finetune_epiaware_grn.py \
  --rna_path datasets/pN_combined/rna.npy \
  --atac_path datasets/pN_combined/atac.npy \
  --candidate_file datasets/pN_combined/candidate_peaks.json \
  --gene_names datasets/pN_combined/genes.txt \
  --peak_names datasets/pN_combined/peaks.txt \
  --tf_list datasets/tf_train.txt \
  --positive_links datasets/P_bulk.csv \
  --backbone_checkpoint output/epiaware_backbone/epiaware_backbone.pth \
  --output_dir output/epiaware_grn \
  --positive_prior 0.08 \
  --pos_batch_size 512 \
  --unl_batch_size 4096
```

The script freezes the pretrained backbone, streams cell batches, samples TF-target pairs via `PositiveUnlabeledSampler`, and optimizes the nnPU objective from Kiryo et al. (2017). Saved checkpoints plus a `training_metrics.csv` (nnPU loss per epoch) reside in `output_dir`, and the final GRN head weights are stored as `grn_head.pth`.

## End-to-End Automation

To run both stages sequentially with shared inputs:

```bash
./training/run_epiaware_pipeline.sh \
  datasets/pN_combined/rna.npy \
  datasets/pN_combined/atac.npy \
  datasets/pN_combined/candidate_peaks.json \
  datasets/tf_train.txt \
  datasets/P_bulk.csv \
  output/epiaware_pipeline
```

Optional environment variables `GENE_NAMES`, `PEAK_NAMES`, `STAGE1_ARGS`, and `STAGE2_ARGS` let you customize feature lists and hyperparameters without editing the script.

### Preconfigured pN / mN Pipelines

- **pN condition**: `./scripts/run_pN_pipeline.sh`
- **mN condition**: `./scripts/run_mN_pipeline.sh`

## Baseline GRN Models

We bundle reproducible baselines alongside EpiAwareNet so you can compare against
classical gene regulatory network inference tools on identical inputs.

### WGCNA

1. **Dependencies:** R (≥4.1), `WGCNA`, `dynamicTreeCut`.
2. **Run:**

   ```bash
   PYTHONPATH=. python -m baselines.wgcna.run \
     --rna_path datasets/pN_combined/rna.npy \
     --gene_names datasets/pN_combined/genes.txt \
     --output_dir output/baselines/wgcna
   ```

   The helper script exports the expression matrix to CSV and calls
   [`baselines/wgcna/pipeline.R`](baselines/wgcna/pipeline.R) via `Rscript`. Results
   (module assignments, TOM, soft-threshold scan) are saved under the chosen output directory.

   要同时跑 `pN` / `mN`，可以使用批处理封装：

   ```bash
   PYTHONPATH=. python -m baselines.wgcna.run_datasets \
     --output_root output/baselines/wgcna \
     --datasets pN mN
   ```

   如需添加其他数据集，可通过 `--add_dataset name:/path/to/rna.npy:/path/to/genes.txt` 注册。

### GRNBoost2

1. **Dependencies:** `arboreto`, `numpy`, `pandas`, LightGBM backend.
2. **Run:**

   ```bash
   PYTHONPATH=. python -m baselines.grnboost2.run \
     --rna_path datasets/pN_combined/rna.npy \
     --gene_names datasets/pN_combined/genes.txt \
     --tf_list datasets/pN_combined/tfs.txt \
     --output_dir output/baselines/grnboost2
   ```

   Outputs a ranked edge list (`grnboost2_edges.csv`). Adjust `--chunk_size` to trade off
   runtime vs. peak memory.

   To run both pN 与 mN 一起，可用批处理脚本：

   ```bash
   PYTHONPATH=. python -m baselines.grnboost2.run_datasets \
     --output_root output/baselines/grnboost2 \
     --datasets pN mN
   ```

### RNA-only scGPT

- **Run:** `bash scripts/run_scgpt_baseline.sh mN` (datasets: `mN|pN|pbmc|mouse_brain`). Overrides: `SCGPT_N_HVG`, `SCGPT_EPOCHS`, `SCGPT_TOP_K`, `SCGPT_DEVICE`, `SCGPT_NO_HEAD`, `SCGPT_EXTRA_ARGS`.
- **Outputs:** `<out>/grn.csv` (supervised link head when `positive_links` available) + `grn_cosine.csv` (embedding cosine baseline) + `metrics_summary.tsv` 和 `custom_eval/metrics.json`。
### scGLUE (RNA+ATAC)

- **Environment:** Use the CPU env spec in `envs/scglue_cpu.yml` to avoid NCCL/CuPy linkage issues (`conda env create -f envs/scglue_cpu.yml`; `SCGLUE_PYTHON=$(conda run -n scglue-cpu which python)`).
- **Run:** `bash scripts/run_scglue_baseline.sh mN` (datasets: `mN|pN|pbmc|mouse_brain`). Overrides: `SCGLUE_N_HVG`, `SCGLUE_N_HVP`, `SCGLUE_EPOCHS`, `SCGLUE_TOP_K`, `SCGLUE_EXTRA_ARGS`, `SCGLUE_ATAC_PROB` (Bernoulli/NB/ZINB), `SCGLUE_BINARIZE_ATAC`.

### SCENIC+

We provide a thin wrapper around the official SCENIC+ Snakemake workflow. You still
need the SCENIC+ environment (see the
[tutorial](https://scenicplus.readthedocs.io/en/latest/tutorials.html) for installation
and required motif databases), but the script automates folder setup, config editing,
Snakemake execution, and TF→gene export.

1. Install SCENIC+ from GitHub inside a dedicated conda env:

   ```bash
   mamba create -n scenicplus python=3.10 -y
   mamba activate scenicplus
   pip install git+https://github.com/aertslab/SCENICplus.git
   mamba install snakemake -y
   ```

2. Ensure you have the prerequisite files expected by SCENIC+:
   - A pycisTopic object (`cisTopic_obj.pkl`).
   - AnnData files with gene expression / chromatin accessibility counts. You can
     convert the matrices shipped with this repo via:

     ```bash
     PYTHONPATH=. python -m baselines.scenicplus.prepare_inputs \
       --rna_path datasets/pN_combined/rna.npy \
       --atac_path datasets/pN_combined_ATAC/atac.npy \
       --candidate_file datasets/pN_combined/candidate_peaks.json \
       --gene_names datasets/pN_combined/genes.txt \
       --peak_names datasets/pN_combined_ATAC/peaks.txt \
       --tf_list datasets/pN_combined/tfs.txt \
     --output_dir output/baselines/scenicplus/inputs
     ```

     This generates `rna_counts.h5ad`, `atac_counts.h5ad`, candidate tensors, and (optionally) a TF list that the pipeline can reuse.
   - Motif enrichment inputs (BED + motif results). 如果还没有运行 pycistarget，可使用：

     ```bash
     PYTHONPATH=. python -m baselines.scenicplus.run_pycistarget \
       --summary output/baselines/scenicplus/pN_inputs/summary.json \
       --peaks_txt datasets/pN_combined_ATAC/peaks.txt \
       --ctx_db /path/to/cistarget_db.feather \
       --motif_annotations /path/to/motif_annotations.tbl \
       --species arabidopsis_thaliana \
       --output_dir output/baselines/scenicplus/pycistarget
     ```

     脚本会把 `peaks.txt` 转成 BED，并调用 pycisTarget 的 Python API 生成富集结果（TSV/HDF5/HTML）。需要提前准备好适配的 ranking database 和 motif 注释。
   - Region sets folder, cisTarget/DEM databases, and motif annotation table as in the
     SCENIC+ tutorial.

3. Launch the helper script (inside the SCENIC+ environment):

   ```bash
   PYTHONPATH=. python -m baselines.scenicplus.run \
     --cisTopic_obj path/to/cistopic.pkl \
     --rna_anndata output/baselines/scenicplus/inputs/rna_counts.h5ad \
     --region_sets path/to/region_sets \
     --ctx_db path/to/ctx_db.feather \
     --dem_db path/to/dem_db.h5 \
     --motif_annotations path/to/motif_annotations.tbl \
     --output_dir output/baselines/scenicplus \
     --cores 8
   ```

   The script calls ``scenicplus init_snakemake`` under the hood, patches the config,
   runs Snakemake, and writes `predictions_scplus.csv` containing TF→target scores.
   Add `--skip_snakemake` if you only want the configured folder for manual runs, or
   `--snakemake "snakemake --use-conda"` to delegate environment resolution to Snakemake.

   To process multiple datasets (e.g., both `pN_inputs` and `mN_inputs`) in sequence,
   use the batch wrapper:

   ```bash
   PYTHONPATH=. python -m baselines.scenicplus.run_datasets \
     --inputs output/baselines/scenicplus/pN_inputs/summary.json \
              output/baselines/scenicplus/mN_inputs/summary.json \
     --cisTopic_obj path/to/cistopic.pkl \
     --region_sets path/to/region_sets \
     --ctx_db path/to/ctx_db.feather \
     --dem_db path/to/dem_db.h5 \
     --motif_annotations path/to/motif_annotations.tbl \
     --output_root output/baselines/scenicplus/runs \
     --cores 8
   ```

4. Feed the resulting predictions into the evaluation script just like any other
   baseline.

Environment overrides (shared by both scripts):

- `RUN_CONVERSION=0` skips re-running the 10x→npy conversion if you already have `rna.npy`/`atac.npy` in the respective directories.
- `STAGE1_ARGS="--epochs 80 --batch_size 32"` or `STAGE2_ARGS="--epochs 15 --pos_batch_size 1024"` tweak hyperparameters without editing the scripts.
- `OUTPUT_ROOT=/path/to/output_dir` sets the destination directory (defaults to `output/<pN|mN>_pipeline_<timestamp>`).

Each helper script assembles the candidate JSON, runs Stage 1 and Stage 2, and leaves checkpoints + metrics in `OUTPUT_ROOT/backbone` and `OUTPUT_ROOT/grn_head` respectively.

## Evaluation & Visualization

After generating TF-target probabilities (e.g., from Stage 2 inference), compute AUPRC/AUROC and plot PR/ROC curves via:

```bash
python evaluate/epiaware_metrics.py \
  --predictions output/epiaware_pipeline/grn_predictions.json \
  --gold datasets/TF_Target_Validation_inSingleCell.txt \
  --output_dir output/metrics/arabidopsis
```

or simply:

```bash
./evaluate/run_epiaware_metrics.sh \
  output/epiaware_pipeline/grn_predictions.json \
  datasets/TF_Target_Validation_inSingleCell.txt \
  output/metrics/arabidopsis
```

Artifacts dropped into `output_dir`:

- `metrics.json` with coverage stats, AUPRC, and AUROC.
- `pr_curve.png` / `roc_curve.png` for quick visual inspection.

## Next Steps

Once the head is trained, you can construct context-specific GRNs by:

1. Loading the backbone/head weights.
2. Running the frozen backbone on cell subsets (e.g., a Scanpy AnnData mask).
3. Scoring TF-gene pairs per cell with `GRNPredictorHead`.
4. Averaging sigmoid outputs within each context to populate \( \mathcal{A}^C \).

This workflow keeps the original repository functionality intact while extending it with the requested two-stage EpiAwareNet pipeline.
