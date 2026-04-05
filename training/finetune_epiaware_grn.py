import argparse
import csv
import json
import math
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np

try:  # Optional dependency for experiment tracking
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from epiaware import (
    CandidateSet,
    MultiOmicDataset,
    PositiveUnlabeledSampler,
    load_positive_links,
    load_tf_list,
)
from epiaware.models import (
    EpiAwarePretrainingModel,
    EpiAwareTransformer,
    GRNPredictorHead,
    MaskedNegativeBinomialHead,
    nn_pu_loss,
)


VAL_METRIC_HEADERS = [
    "val_pu_loss",
    "val_mse",
    "val_auroc",
    "val_auprc",
    "val_topk_hit_rate",
    "val_degree_mean",
    "val_degree_median",
    "val_degree_p95",
    "val_degree_max",
    "val_pvalue_min",
    "val_pvalue_median",
]


def _resolve_export_path(user_path: str, output_dir: Path) -> Path:
    path = Path(user_path)
    if path.is_absolute():
        return path

    # If output_dir is absolute but the user provided a repo-root-relative path
    # (e.g., "output/<run>/.../grn.csv"), naive prefixing would create
    # "<output_dir>/output/<run>/.../grn.csv". Detect this by resolving relative
    # paths against the repo root and checking whether they already fall under
    # output_dir.
    output_dir_resolved: Optional[Path]
    try:
        output_dir_resolved = output_dir.resolve()
    except Exception:  # pragma: no cover
        output_dir_resolved = None

    if output_dir_resolved is not None and output_dir_resolved.is_absolute() and ROOT.is_absolute():
        abs_candidate = (ROOT / path).resolve()
        try:
            if abs_candidate.is_relative_to(output_dir_resolved):
                return abs_candidate
        except AttributeError:  # pragma: no cover - py<3.9
            if abs_candidate.parts[: len(output_dir_resolved.parts)] == output_dir_resolved.parts:
                return abs_candidate

    # Fallback: if both are relative/compatible, avoid double-prefixing.
    try:
        if path.is_relative_to(output_dir):
            return path
    except AttributeError:  # pragma: no cover - py<3.9
        if path.parts[: len(output_dir.parts)] == output_dir.parts:
            return path

    return output_dir / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 nnPU fine-tuning for GRNs")
    parser.add_argument("--rna_path", type=str, required=True)
    parser.add_argument("--atac_path", type=str, required=True)
    parser.add_argument(
        "--candidate_file",
        type=str,
        default=None,
        help="Candidate peak set JSON/TSV. If omitted, use a full candidate set over all peaks.",
    )
    parser.add_argument("--positive_links", type=str, required=True, help="Bulk positive TF-target pairs")
    parser.add_argument(
        "--val_positive_links",
        type=str,
        default=None,
        help="Optional validation TF-target pairs (falls back to --positive_links when omitted).",
    )
    parser.add_argument(
        "--use_bce_loss",
        action="store_true",
        help="Use binary cross-entropy loss instead of nnPU loss for training.",
    )
    parser.add_argument("--tf_list", type=str, required=True, help="List of TF names (txt/json)")
    parser.add_argument("--gene_names", type=str, default=None)
    parser.add_argument("--peak_names", type=str, default=None)
    parser.add_argument("--backbone_checkpoint", type=str, required=True, help="Stage 1 checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="output/epiaware_grn")
    parser.add_argument("--cell_batch_size", type=int, default=32)
    parser.add_argument("--pos_batch_size", type=int, default=256)
    parser.add_argument("--unl_batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--positive_prior", type=float, default=0.2)
    parser.add_argument(
        "--negative_risk_weight",
        type=float,
        default=1.0,
        help="Weight applied to the non-negative negative-risk term in nnPU loss (values >1 penalise false positives).",
    )
    parser.add_argument("--tf_prefix", type=str, default="gene:", help="Prefix to prepend to TF ids if missing")
    parser.add_argument(
        "--positive_tf_prefix",
        type=str,
        default="gene:",
        help="Prefix enforced on TF column in positive links",
    )
    parser.add_argument(
        "--positive_target_prefix",
        type=str,
        default="gene:",
        help="Prefix enforced on target column in positive links",
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--k_neighbors", type=int, default=12)
    parser.add_argument(
        "--reuse_knn_indices",
        action="store_true",
        default=False,
        help=(
            "Reuse gene-gene kNN indices across all transformer blocks during eval/export. "
            "This can significantly speed up validation/export for large gene counts at the cost of using a fixed "
            "neighbor graph computed from the input embeddings (approximation)."
        ),
    )
    parser.add_argument("--topk_peaks", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--recon_hidden", type=int, default=512)
    parser.add_argument("--recon_depth", type=int, default=12)
    parser.add_argument("--head_hidden", type=int, default=512)
    parser.add_argument("--head_depth", type=int, default=2)
    parser.add_argument(
        "--head_checkpoint",
        type=str,
        default=None,
        help="Optional path to a pre-trained GRN head checkpoint (.pth) to resume or export without retraining.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0,
        help="Fraction of cells reserved for validation (set to 0 to disable).",
    )
    parser.add_argument(
        "--val_cell_batch_size",
        type=int,
        default=None,
        help="Validation cell batch size (defaults to cell_batch_size).",
    )
    parser.add_argument(
        "--val_pos_batch_size",
        type=int,
        default=None,
        help="Number of positive TF-target pairs sampled per validation step (defaults to pos_batch_size).",
    )
    parser.add_argument(
        "--val_unl_batch_size",
        type=int,
        default=None,
        help="Number of unlabeled TF-target pairs sampled per validation step (defaults to unl_batch_size).",
    )
    parser.add_argument(
        "--val_topk",
        type=int,
        default=50,
        help="K value used to compute Top-K hit rate on validation predictions.",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip validation passes entirely (useful for eval-only export).",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases. Requires the wandb package.",
        default=False,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name (required when --enable_wandb is set).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity (team/user) name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional override for the W&B run name.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of tags to attach to the W&B run.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B mode to use when tracking (online/offline/disabled).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=200,
        help="Number of training steps between intermediate checkpoint saves (<=0 disables).",
    )
    parser.add_argument(
        "--export_grn_csv",
        type=str,
        default=None,
        help="Optional filename to export aggregated TF-target probabilities as CSV (relative to output_dir when not absolute).",
    )
    parser.add_argument(
        "--export_grn_each_epoch",
        action="store_true",
        help="If set, export a GRN CSV after every training epoch (filenames get an _epoch{N} suffix).",
    )
    parser.add_argument(
        "--export_topk_per_tf",
        type=int,
        default=None,
        help="If set, limit exported rows to the top-K targets per TF (<=0 exports all targets).",
    )
    parser.add_argument(
        "--export_probability_threshold",
        type=float,
        default=None,
        help="Optional probability floor; rows below this threshold are dropped from the exported CSV.",
    )
    parser.add_argument(
        "--export_tf_chunk_size",
        type=int,
        default=16,
        help="Number of TF indices processed simultaneously when aggregating export scores.",
    )
    parser.add_argument(
        "--export_target_chunk_size",
        type=int,
        default=512,
        help="Number of target indices processed simultaneously when aggregating export scores.",
    )
    parser.add_argument(
        "--export_logit_path",
        type=str,
        default=None,
        help="Optional path to store averaged TF-target logits; when present and file exists, reuse instead of recomputing.",
    )
    parser.add_argument(
        "--export_max_cells",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of cells used when exporting the GRN. "
            "When set, logits are averaged over only a subset of cells to reduce runtime. "
            "(Default: use all cells.)"
        ),
    )
    parser.add_argument(
        "--export_cell_sampling",
        type=str,
        choices=("first", "random"),
        default="first",
        help="How to pick cells when --export_max_cells is set: take the first N, or a random subset.",
    )
    parser.add_argument(
        "--export_cell_seed",
        type=int,
        default=42,
        help="Random seed used when --export_cell_sampling=random.",
    )
    parser.add_argument(
        "--val_prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used when summarizing TF out-degree distribution on validation batches.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip nnPU training and only run validation metrics using a provided GRN head checkpoint.",
    )
    return parser.parse_args()


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0:
        return float("nan")
    labels = labels.astype(np.float64)
    scores = scores.astype(np.float64)
    pos_mask = labels == 1.0
    neg_mask = labels == 0.0
    n_pos = int(np.count_nonzero(pos_mask))
    n_neg = int(np.count_nonzero(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(scores.size)
    rank_sum = ranks[pos_mask].sum() + n_pos  # convert to 1-based ranks
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _compute_topk_hit_rate(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k is None or k <= 0 or labels.size == 0:
        return float("nan")
    k_eff = min(k, labels.size)
    order = np.argsort(scores)[::-1][:k_eff]
    topk_labels = labels[order]
    hits = int(np.count_nonzero(topk_labels == 1.0))
    return float(hits / k_eff)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    total_pos = int(np.count_nonzero(labels > 0.5))
    if total_pos == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    precision_sum = 0.0
    cum_pos = 0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] > 0.5:
            cum_pos += 1
            precision_sum += cum_pos / rank
    if cum_pos == 0:
        return float("nan")
    return float(precision_sum / cum_pos)


def init_wandb_run(args: argparse.Namespace) -> Optional["wandb.sdk.wandb_run.Run"]:
    if not args.enable_wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError(
            "Weights & Biases is not installed. Install it with `pip install wandb` "
            "or run without --enable_wandb."
        )
    if not args.wandb_project:
        raise ValueError("Specify --wandb_project when using --enable_wandb.")
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=config,
    )
    run.define_metric("train/global_step")
    run.define_metric("train/step_loss", step_metric="train/global_step")
    run.define_metric("train/pos_logit_mean", step_metric="train/global_step")
    run.define_metric("train/unl_logit_mean", step_metric="train/global_step")
    run.define_metric("train/pos_prob_mean", step_metric="train/global_step")
    run.define_metric("train/unl_prob_mean", step_metric="train/global_step")
    run.define_metric("epoch")
    run.define_metric("train/epoch_loss", step_metric="epoch")
    run.define_metric("val/pu_loss", step_metric="epoch")
    run.define_metric("val/pos_logit_mean", step_metric="epoch")
    run.define_metric("val/unl_logit_mean", step_metric="epoch")
    run.define_metric("val/pos_prob_mean", step_metric="epoch")
    run.define_metric("val/unl_prob_mean", step_metric="epoch")
    return run


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_backbone(args, dataset, candidate_set, device):
    backbone = EpiAwareTransformer(
        num_genes=dataset.num_genes,
        num_peaks=dataset.num_peaks,
        candidate_index=candidate_set.index,
        candidate_mask=candidate_set.mask,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        k_neighbors=args.k_neighbors,
        topk_peaks=args.topk_peaks,
        ffn_hidden=args.ffn_dim,
        dropout=args.dropout,
        reuse_knn_indices=args.reuse_knn_indices,
    )
    recon_head = MaskedNegativeBinomialHead(
        embed_dim=args.embed_dim,
        hidden_dim=args.recon_hidden,
        depth=args.recon_depth,
        dropout=args.dropout,
    )
    model = EpiAwarePretrainingModel(backbone, recon_head)
    state = torch.load(args.backbone_checkpoint, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print(f"[warn] Missing keys when loading checkpoint: {missing.missing_keys}")
    model = model.to(device)
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()
    return model.backbone


def gather_logits(head, gene_states, pair_idx):
    """Compute mean logits over cells for each TF-target pair.

    gene_states: (cells, genes, dim)
    pair_idx: (pairs, 2) on device

    Note: For large numbers of pairs (e.g. GRN export), advanced indexing can
    create very large intermediate tensors. We therefore chunk the pair list to
    keep peak GPU memory bounded.
    """

    cells = int(gene_states.size(0))
    pair_count = int(pair_idx.size(0))
    if pair_count == 0:
        return torch.empty((0,), device=gene_states.device, dtype=gene_states.dtype)

    # Conservative default to avoid CUDA OOM during export.
    max_pairs_per_chunk = 1024
    if pair_count <= max_pairs_per_chunk:
        tf_idx = pair_idx[:, 0]
        tgt_idx = pair_idx[:, 1]
        tf_states = gene_states[:, tf_idx, :]
        tgt_states = gene_states[:, tgt_idx, :]
        tf_flat = tf_states.reshape(cells * pair_count, -1)
        tgt_flat = tgt_states.reshape(cells * pair_count, -1)
        logits = head(tf_flat, tgt_flat).view(cells, pair_count)
        return logits.mean(dim=0)

    chunks = []
    for start in range(0, pair_count, max_pairs_per_chunk):
        end = min(start + max_pairs_per_chunk, pair_count)
        pair_chunk = pair_idx[start:end]
        chunk_pairs = int(pair_chunk.size(0))
        if chunk_pairs == 0:
            continue

        tf_idx = pair_chunk[:, 0]
        tgt_idx = pair_chunk[:, 1]
        tf_states = gene_states[:, tf_idx, :]
        tgt_states = gene_states[:, tgt_idx, :]
        tf_flat = tf_states.reshape(cells * chunk_pairs, -1)
        tgt_flat = tgt_states.reshape(cells * chunk_pairs, -1)
        logits = head(tf_flat, tgt_flat).view(cells, chunk_pairs)
        chunks.append(logits.mean(dim=0))

    if not chunks:
        return torch.empty((0,), device=gene_states.device, dtype=gene_states.dtype)
    return torch.cat(chunks, dim=0)


def run_validation_pass(
    args: argparse.Namespace,
    backbone: EpiAwareTransformer,
    head: GRNPredictorHead,
    val_loader: Optional[DataLoader],
    val_sampler: Optional[PositiveUnlabeledSampler],
    tf_names: Sequence[str],
    tf_gene_lookup: np.ndarray,
    device: torch.device,
):
    metrics = {
        "val_pu_loss": None,
        "val_pos_logit_mean": None,
        "val_unl_logit_mean": None,
        "val_pos_prob_mean": None,
        "val_unl_prob_mean": None,
        "val_mse": None,
        "val_auroc": None,
        "val_auprc": None,
        "val_topk_hit_rate": None,
        "val_degree_mean": None,
        "val_degree_median": None,
        "val_degree_p95": None,
        "val_degree_max": None,
        "val_pvalue_min": None,
        "val_pvalue_median": None,
    }
    if val_loader is None or val_sampler is None:
        return metrics
    was_training = head.training
    head.eval()
    val_pos_batch_size = args.val_pos_batch_size or args.pos_batch_size
    val_unl_batch_size = args.val_unl_batch_size or args.unl_batch_size
    val_loss_sum = 0.0
    val_steps = 0
    val_pos_logit_sum = 0.0
    val_unl_logit_sum = 0.0
    val_pos_prob_sum = 0.0
    val_unl_prob_sum = 0.0
    val_scores = []
    val_labels = []
    tf_sample_counts = np.zeros(len(tf_names), dtype=np.int64)
    tf_positive_counts = np.zeros(len(tf_names), dtype=np.int64)
    prob_threshold = max(0.0, float(args.val_prob_threshold))
    with torch.no_grad():
        for batch in val_loader:
            rna = batch["rna"].to(device)
            atac = batch["atac"].to(device)
            gene_states, _ = backbone(rna, atac)
            pos_pairs = val_sampler.sample_positive(val_pos_batch_size).to(device)
            unl_pairs = val_sampler.sample_unlabeled(val_unl_batch_size).to(device)
            pos_logits = gather_logits(head, gene_states, pos_pairs)
            unl_logits = gather_logits(head, gene_states, unl_pairs)
            if args.use_bce_loss:
                pos_labels = torch.ones_like(pos_logits)
                unl_labels = torch.zeros_like(unl_logits)
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = (
                    loss_fn(pos_logits, pos_labels)
                    + loss_fn(unl_logits, unl_labels)
                ) / 2.0
            else:
                loss = nn_pu_loss(
                    pos_logits,
                    unl_logits,
                    args.positive_prior,
                    args.negative_risk_weight,
            )
            val_loss_sum += loss.item()
            val_steps += 1
            pos_mean = pos_logits.mean().item()
            unl_mean = unl_logits.mean().item()
            val_pos_logit_sum += pos_mean
            val_unl_logit_sum += unl_mean
            pos_probs = torch.sigmoid(pos_logits)
            unl_probs = torch.sigmoid(unl_logits)
            val_pos_prob_sum += pos_probs.mean().item()
            val_unl_prob_sum += unl_probs.mean().item()
            val_scores.append(
                torch.cat([pos_probs, unl_probs]).detach().cpu().numpy()
            )
            val_labels.append(
                np.concatenate(
                    [
                        np.ones(pos_probs.size(0), dtype=np.float32),
                        np.zeros(unl_probs.size(0), dtype=np.float32),
                    ]
                )
            )

            combined_pairs = torch.cat([pos_pairs, unl_pairs], dim=0)
            combined_probs = torch.cat([pos_probs, unl_probs], dim=0)
            combined_pairs_np = combined_pairs.detach().cpu().numpy()
            combined_probs_np = combined_probs.detach().cpu().numpy()
            tf_gene_indices = combined_pairs_np[:, 0]
            tf_positions = tf_gene_lookup[tf_gene_indices]
            valid_mask = tf_positions >= 0
            if np.any(valid_mask):
                valid_tf_positions = tf_positions[valid_mask]
                np.add.at(tf_sample_counts, valid_tf_positions, 1)
                if prob_threshold > 0.0:
                    predicted_mask = combined_probs_np >= prob_threshold
                else:
                    predicted_mask = np.ones_like(combined_probs_np, dtype=bool)
                positive_mask = valid_mask & predicted_mask
                if np.any(positive_mask):
                    pos_tf_positions = tf_positions[positive_mask]
                    np.add.at(tf_positive_counts, pos_tf_positions, 1)
    if val_steps > 0:
        metrics["val_pu_loss"] = val_loss_sum / val_steps
        metrics["val_pos_logit_mean"] = val_pos_logit_sum / val_steps
        metrics["val_unl_logit_mean"] = val_unl_logit_sum / val_steps
        metrics["val_pos_prob_mean"] = val_pos_prob_sum / val_steps
        metrics["val_unl_prob_mean"] = val_unl_prob_sum / val_steps
        if val_scores:
            combined_scores = np.concatenate(val_scores)
            combined_labels = np.concatenate(val_labels)
            metrics["val_mse"] = float(
                np.mean((combined_scores - combined_labels) ** 2)
            )
            metrics["val_auroc"] = _compute_auc(combined_labels, combined_scores)
            metrics["val_auprc"] = _average_precision(
                combined_labels, combined_scores
            )
            metrics["val_topk_hit_rate"] = _compute_topk_hit_rate(
                combined_labels, combined_scores, args.val_topk
            )
            total_samples = int(tf_sample_counts.sum())
            if total_samples > 0:
                sampled_mask = tf_sample_counts > 0
                positive_counts = tf_positive_counts.astype(np.float64)
                if sampled_mask.any():
                    metrics["val_degree_mean"] = float(
                        positive_counts[sampled_mask].mean()
                    )
                    metrics["val_degree_median"] = float(
                        np.median(positive_counts[sampled_mask])
                    )
                    metrics["val_degree_p95"] = float(
                        np.percentile(positive_counts[sampled_mask], 95)
                    )
                    metrics["val_degree_max"] = float(
                        positive_counts[sampled_mask].max()
                    )
                else:
                    metrics["val_degree_mean"] = metrics["val_degree_median"] = (
                        metrics["val_degree_p95"]
                    ) = metrics["val_degree_max"] = 0.0
                total_predicted = float(positive_counts.sum())
                base_prob = total_predicted / total_samples if total_samples > 0 else 0.0
                if sampled_mask.any() and 0.0 < base_prob < 1.0:
                    counts_tensor = torch.from_numpy(
                        tf_sample_counts[sampled_mask].astype(np.float64)
                    ).to(torch.float64)
                    success_tensor = torch.from_numpy(
                        positive_counts[sampled_mask]
                    ).to(torch.float64)
                    probs_tensor = torch.full_like(
                        counts_tensor, base_prob, dtype=torch.float64
                    )
                    pvals = torch.ones_like(success_tensor, dtype=torch.float64)
                    positive_mask = success_tensor > 0
                    if positive_mask.any():
                        k_tensor = (success_tensor[positive_mask] - 1.0).clamp(min=0.0)

                        def _binomial_survival(n_vals, p_vals, k_vals):
                            result = torch.zeros_like(k_vals, dtype=torch.float64)
                            k_int = k_vals.to(torch.int64)
                            for idx in range(k_int.numel()):
                                n_i = int(n_vals[idx].item())
                                p_i = float(p_vals[idx].item())
                                k_i = int(k_int[idx].item())
                                if n_i <= 0 or k_i >= n_i:
                                    result[idx] = 0.0
                                    continue
                                j_start = k_i + 1
                                if j_start > n_i:
                                    result[idx] = 0.0
                                    continue
                                log_coeff = math.lgamma(n_i + 1)
                                log_p = math.log(p_i)
                                log_q = math.log1p(-p_i)
                                log_terms = []
                                for j in range(j_start, n_i + 1):
                                    log_comb = log_coeff - math.lgamma(j + 1) - math.lgamma(n_i - j + 1)
                                    log_prob = log_comb + j * log_p + (n_i - j) * log_q
                                    log_terms.append(log_prob)
                                if not log_terms:
                                    result[idx] = 0.0
                                    continue
                                max_log = max(log_terms)
                                if math.isinf(max_log):
                                    result[idx] = 0.0
                                    continue
                                sum_exp = sum(math.exp(term - max_log) for term in log_terms)
                                result[idx] = math.exp(max_log) * sum_exp
                            return torch.clamp(result, min=0.0, max=1.0)

                        tail_probs = _binomial_survival(
                            counts_tensor[positive_mask],
                            probs_tensor[positive_mask],
                            k_tensor,
                        )
                        pvals[positive_mask] = tail_probs
                    metrics["val_pvalue_min"] = float(pvals.min().item())
                    metrics["val_pvalue_median"] = float(pvals.median().item())
                else:
                    metrics["val_pvalue_min"] = 1.0
                    metrics["val_pvalue_median"] = 1.0
    head.train(mode=was_training)
    return metrics


def append_metric(
    path: Path,
    epoch: int,
    train_loss: Optional[float],
    val_metrics: Dict[str, Optional[float]],
) -> None:
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as handle:
        metric_headers = VAL_METRIC_HEADERS
        if write_header:
            columns = ["epoch", "train_loss"] + metric_headers
            handle.write(",".join(columns) + "\n")

        def fmt(value: Optional[float]) -> str:
            if value is None:
                return "NA"
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                return str(value)
            if math.isnan(value_f) or math.isinf(value_f):
                return "NA"
            return f"{value_f:.6f}"

        row = [str(epoch), fmt(train_loss)]
        for key in metric_headers:
            row.append(fmt(val_metrics.get(key)))
        handle.write(",".join(row) + "\n")


def export_grn_to_csv(
    args: argparse.Namespace,
    output_dir: Path,
    backbone: EpiAwareTransformer,
    head: GRNPredictorHead,
    dataset: MultiOmicDataset,
    tf_names: Sequence[str],
    positive_links: Sequence[Tuple[str, str]],
    device: torch.device,
    epoch: Optional[int] = None,
) -> None:
    if not args.export_grn_csv:
        return

    export_path = _resolve_export_path(args.export_grn_csv, output_dir)
    if epoch is not None:
        suffix = export_path.suffix or ".csv"
        export_path = export_path.parent / f"{export_path.stem}_epoch{epoch}{suffix}"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    gene_to_idx = {name: idx for idx, name in enumerate(dataset.gene_names)}
    valid_tf_indices: List[int] = []
    valid_tf_names: List[str] = []
    skipped_tfs: List[str] = []
    for name in tf_names:
        idx = gene_to_idx.get(name)
        if idx is None:
            skipped_tfs.append(name)
            continue
        valid_tf_indices.append(idx)
        valid_tf_names.append(name)
    if not valid_tf_indices:
        print("[export] No TFs overlap with dataset gene list; skipping CSV export.")
        return

    batch_size = args.val_cell_batch_size or args.cell_batch_size
    tf_chunk = max(1, int(args.export_tf_chunk_size))
    tgt_chunk = max(1, int(args.export_target_chunk_size))
    num_tfs = len(valid_tf_indices)
    num_targets = len(dataset.gene_names)

    cache_path: Optional[Path] = None
    avg_logits: Optional[torch.Tensor] = None
    if args.export_logit_path:
        cache_path = _resolve_export_path(args.export_logit_path, output_dir)
        if epoch is not None:
            cache_suffix = cache_path.suffix or ".pt"
            cache_path = cache_path.parent / f"{cache_path.stem}_epoch{epoch}{cache_suffix}"
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                if isinstance(payload, dict) and "avg_logits" in payload:
                    avg_logits = torch.as_tensor(payload["avg_logits"], dtype=torch.float32)
                else:
                    avg_logits = torch.as_tensor(payload, dtype=torch.float32)
                if avg_logits.dim() != 2 or avg_logits.size(0) != num_tfs or avg_logits.size(1) != num_targets:
                    print(
                        f"[warn] Cached logits at {cache_path} have shape {tuple(avg_logits.shape)}, "
                        f"expected ({num_tfs}, {num_targets}); recomputing."
                    )
                    avg_logits = None
                else:
                    print(f"[export] Loaded cached logits from {cache_path}")
            except Exception as err:
                print(f"[warn] Failed to load cached logits from {cache_path}: {err}. Recomputing.")
                avg_logits = None
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

    if avg_logits is None:
        export_dataset = dataset
        if args.export_max_cells is not None:
            max_cells = int(args.export_max_cells)
            if max_cells > 0 and hasattr(dataset, "__len__"):
                total = len(dataset)
                if total > max_cells:
                    if args.export_cell_sampling == "random":
                        rng = np.random.default_rng(int(args.export_cell_seed))
                        indices = rng.choice(total, size=max_cells, replace=False)
                        indices = np.sort(indices).tolist()
                    else:
                        indices = list(range(max_cells))
                    export_dataset = Subset(dataset, indices)
                    print(f"[export] Using {len(indices)}/{total} cells for GRN export (--export_max_cells).")

        export_loader = DataLoader(
            export_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        logit_sums = torch.zeros((num_tfs, num_targets), dtype=torch.float64)
        total_cells = 0

        backbone.eval()
        head.eval()
        tf_indices_tensor = torch.tensor(valid_tf_indices, dtype=torch.long, device=device)

        with torch.no_grad():
            for batch in tqdm(export_loader, desc="Exporting GRN", leave=False):
                rna = batch["rna"].to(device)
                atac = batch["atac"].to(device)
                gene_states, _ = backbone(rna, atac)
                cells_in_batch = gene_states.size(0)
                total_cells += cells_in_batch

                for tf_start in range(0, num_tfs, tf_chunk):
                    tf_end = min(tf_start + tf_chunk, num_tfs)
                    tf_idx_chunk = tf_indices_tensor[tf_start:tf_end]
                    if tf_idx_chunk.numel() == 0:
                        continue
                    for tgt_start in range(0, num_targets, tgt_chunk):
                        tgt_end = min(tgt_start + tgt_chunk, num_targets)
                        tgt_idx_chunk = torch.arange(
                            tgt_start, tgt_end, dtype=torch.long, device=device
                        )
                        if tgt_idx_chunk.numel() == 0:
                            continue
                        try:
                            tf_grid, tgt_grid = torch.meshgrid(
                                tf_idx_chunk, tgt_idx_chunk, indexing="ij"
                            )
                        except TypeError:
                            tf_grid, tgt_grid = torch.meshgrid(tf_idx_chunk, tgt_idx_chunk)
                        pair_idx = torch.stack(
                            (tf_grid.reshape(-1), tgt_grid.reshape(-1)), dim=1
                        )
                        logits = gather_logits(head, gene_states, pair_idx)
                        chunk_logits = logits.reshape(tf_end - tf_start, tgt_end - tgt_start)
                        logit_sums[tf_start:tf_end, tgt_start:tgt_end] += (
                            chunk_logits.double().cpu() * cells_in_batch
                        )

        if total_cells == 0:
            print("[export] Dataset contained zero cells; skipping CSV export.")
            return

        avg_logits = (logit_sums / float(total_cells)).float()
        if cache_path is not None:
            torch.save(
                {
                    "avg_logits": avg_logits.cpu(),
                    "tf_names": valid_tf_names,
                    "target_names": dataset.gene_names,
                    "total_cells": total_cells,
                },
                cache_path,
            )
            print(f"[export] Cached averaged logits to {cache_path}")
    else:
        avg_logits = avg_logits.float()

    prob_matrix = torch.sigmoid(avg_logits)

    gene_name_set = set(dataset.gene_names)
    training_targets = {
        target for _, target in positive_links if target in gene_name_set
    }

    threshold = args.export_probability_threshold
    topk = args.export_topk_per_tf
    if topk is not None and topk <= 0:
        topk = None

    exported_seen_targets: Set[str] = set()
    exported_unseen_targets: Set[str] = set()
    seen_target_count = 0
    unseen_target_count = 0
    edge_count = 0
    with open(export_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tf", "target", "score"])
        for tf_idx, tf_name in enumerate(valid_tf_names):
            scores = prob_matrix[tf_idx]
            if scores.numel() == 0:
                continue
            if topk is not None:
                k = min(int(topk), scores.numel())
                values, indices = torch.topk(scores, k)
            else:
                values, indices = scores.sort(descending=True)
            for score, target_index in zip(values.tolist(), indices.tolist()):
                if threshold is not None and score < threshold:
                    break
                target_name = dataset.gene_names[target_index]
                writer.writerow([tf_name, target_name, f"{score:.6f}"])
                edge_count += 1
                if target_name in training_targets:
                    seen_target_count += 1
                    exported_seen_targets.add(target_name)
                else:
                    unseen_target_count += 1
                    exported_unseen_targets.add(target_name)

    print(f"[export] Wrote {edge_count} edges to {export_path}")
    if skipped_tfs:
        print(
            f"[export] Skipped {len(skipped_tfs)} TFs absent from gene list: "
            f"{', '.join(skipped_tfs[:5])}"
            + (" ..." if len(skipped_tfs) > 5 else "")
        )
    print(
        "[export] Targets in training links: {seen_total} edges ({seen_unique} unique); "
        "targets not in training links: {unseen_total} edges ({unseen_unique} unique)".format(
            seen_total=seen_target_count,
            seen_unique=len(exported_seen_targets),
            unseen_total=unseen_target_count,
            unseen_unique=len(exported_unseen_targets),
        )
    )

def train() -> None:
    args = parse_args()
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
    output_dir: Optional[Path] = None
    start_time: Optional[float] = None
    device: Optional[torch.device] = None
    try:
        wandb_run = init_wandb_run(args)
        set_seed(args.seed)
        device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dataset = MultiOmicDataset(
            rna_path=args.rna_path,
            atac_path=args.atac_path,
            gene_names_path=args.gene_names,
            peak_names_path=args.peak_names,
        )
        if args.candidate_file:
            candidate_set = CandidateSet(
                candidate_file=args.candidate_file,
                gene_names=dataset.gene_names,
                peak_names=dataset.peak_names,
            )
        else:
            candidate_set = CandidateSet.generate_full_candidate_set(
                gene_names=dataset.gene_names,
                peak_names=dataset.peak_names,
            )
        total_cells = len(dataset)
        val_fraction = max(0.0, min(1.0, args.val_fraction))
        if args.eval_only:
            train_dataset = dataset
            val_dataset = dataset
        else:
            val_size = int(total_cells * val_fraction)
            if val_fraction > 0 and total_cells > 1 and val_size == 0:
                val_size = 1
            if val_size >= total_cells:
                val_size = max(total_cells - 1, 0)
            train_size = total_cells - val_size
            if train_size <= 0:
                train_size = total_cells
                val_size = 0
            if val_size > 0:
                generator = torch.Generator().manual_seed(args.seed)
                train_dataset, val_dataset = random_split(
                    dataset, [train_size, val_size], generator=generator
                )
            else:
                train_dataset = dataset
                val_dataset = None
        train_loader = None
        if not args.eval_only:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.cell_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=False,
            )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_cell_batch = args.val_cell_batch_size or args.cell_batch_size
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_cell_batch,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
        if args.skip_validation:
            val_loader = None
            val_sampler = None
        if wandb_run is not None:
            wandb_run.config.update(
                {
                    "train_cell_count": len(train_dataset),
                    "val_cell_count": len(val_dataset) if val_dataset is not None else 0,
                },
                allow_val_change=True,
            )
        backbone = build_backbone(args, dataset, candidate_set, device)
        tf_names = load_tf_list(args.tf_list, dataset.gene_names, prefix=args.tf_prefix)
        positive_links = load_positive_links(
            args.positive_links,
            tf_prefix=args.positive_tf_prefix,
            target_prefix=args.positive_target_prefix,
        )
        if not positive_links:
            raise ValueError(f"No positive links loaded from {args.positive_links}")
        
        val_positive_links = None
        if args.val_positive_links:
            val_positive_links = load_positive_links(
                args.val_positive_links,
                tf_prefix=args.positive_tf_prefix,
                target_prefix=args.positive_target_prefix,
            )
            if not val_positive_links:
                raise ValueError(
                    f"No validation positive links loaded from {args.val_positive_links}"
                )
        elif not args.eval_only and val_fraction > 0:
            import random
            rng = random.Random(args.seed)
            links_copy = list(positive_links)
            rng.shuffle(links_copy)
            n_total = len(links_copy)
            n_val = int(n_total * val_fraction)
            if n_val < 1 and n_total > 1:
                n_val = 1
            val_positive_links = links_copy[:n_val]
            positive_links = links_copy[n_val:]
            print(
                f"[info] Auto-splitting positive links: {len(positive_links)} training, "
                f"{len(val_positive_links)} validation (fraction={val_fraction:.2f})"
            )
        else:
            val_positive_links = positive_links

        sampler = PositiveUnlabeledSampler(tf_names, dataset.gene_names, positive_links)
        val_sampler = None
        if val_loader is not None:
            val_sampler = PositiveUnlabeledSampler(tf_names, dataset.gene_names, val_positive_links)
            val_sampler.rng = np.random.default_rng(args.seed + 1)
        val_pos_batch_size = args.val_pos_batch_size or args.pos_batch_size
        val_unl_batch_size = args.val_unl_batch_size or args.unl_batch_size
        tf_gene_lookup = np.full(dataset.num_genes, -1, dtype=np.int64)
        for tf_pos, gene_idx in enumerate(sampler.tf_indices):
            tf_gene_lookup[int(gene_idx)] = tf_pos
        head = GRNPredictorHead(
            embed_dim=args.embed_dim,
            hidden_dim=args.head_hidden,
            depth=args.head_depth,
            dropout=args.dropout,
        ).to(device)
        if args.eval_only and not args.head_checkpoint:
            raise ValueError("--eval_only requires --head_checkpoint to load weights.")
        if args.head_checkpoint:
            head_ckpt_path = Path(args.head_checkpoint)
            if not head_ckpt_path.exists():
                raise FileNotFoundError(f"Head checkpoint not found: {head_ckpt_path}")
            head_state = torch.load(head_ckpt_path, map_location=device)
            if isinstance(head_state, dict) and "head_state" in head_state:
                head_state = head_state["head_state"]
            missing_head = head.load_state_dict(head_state, strict=False)
            if missing_head.missing_keys:
                print(f"[warn] Missing keys when loading head checkpoint: {missing_head.missing_keys}")
            if missing_head.unexpected_keys:
                print(f"[warn] Unexpected keys when loading head checkpoint: {missing_head.unexpected_keys}")

        optimizer = None
        if not args.eval_only:
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_payload = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

        start_time = time.time()
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                torch.cuda.reset_peak_memory_stats()
        metrics_path = output_dir / "training_metrics.csv"
        if metrics_path.exists() and not args.eval_only:
            metrics_path.unlink()

        step_ckpt_dir: Optional[Path] = None
        if not args.eval_only and args.save_every_steps and args.save_every_steps > 0:
            step_ckpt_dir = output_dir / "step_checkpoints"
            step_ckpt_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0
        if not args.eval_only:
            for epoch in range(1, args.epochs + 1):
                head.train()
                epoch_total_loss = 0.0
                steps = 0
                with tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}") as progress:
                    for batch in progress:
                        rna = batch["rna"].to(device)
                        atac = batch["atac"].to(device)
                        with torch.no_grad():
                            gene_states, _ = backbone(rna, atac)
                        pos_pairs = sampler.sample_positive(args.pos_batch_size).to(device)
                        unl_pairs = sampler.sample_unlabeled(args.unl_batch_size).to(device)
                        pos_logits = gather_logits(head, gene_states, pos_pairs)
                        unl_logits = gather_logits(head, gene_states, unl_pairs)
                        pos_logit_mean = pos_logits.mean().item()
                        unl_logit_mean = unl_logits.mean().item()
                        pos_prob_mean = torch.sigmoid(pos_logits).mean().item()
                        unl_prob_mean = torch.sigmoid(unl_logits).mean().item()
                        if args.use_bce_loss:
                            pos_labels = torch.ones_like(pos_logits)
                            unl_labels = torch.zeros_like(unl_logits)
                            loss_fn = torch.nn.BCEWithLogitsLoss()
                            loss = (
                                loss_fn(pos_logits, pos_labels)
                                + loss_fn(unl_logits, unl_labels)
                            ) / 2.0
                        else:
                            loss = nn_pu_loss(
                                pos_logits,
                                unl_logits,
                                args.positive_prior,
                                args.negative_risk_weight,
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_value = loss.item()
                        epoch_total_loss += loss_value
                        steps += 1
                        progress.set_postfix(loss=epoch_total_loss / steps)
                        global_step += 1

                        if (
                            step_ckpt_dir is not None
                            and args.save_every_steps
                            and global_step % args.save_every_steps == 0
                        ):
                            step_path = step_ckpt_dir / f"head_step_{global_step}.pth"
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "head_state": head.state_dict(),
                                    "optimizer_state": optimizer.state_dict(),
                                },
                                step_path,
                            )

                        if wandb_run is not None:
                            metrics = {
                                "train/global_step": global_step,
                                "train/step_loss": loss_value,
                                "train/pos_batch_size": pos_pairs.size(0),
                                "train/unl_batch_size": unl_pairs.size(0),
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/pos_logit_mean": pos_logit_mean,
                                "train/unl_logit_mean": unl_logit_mean,
                                "train/pos_prob_mean": pos_prob_mean,
                                "train/unl_prob_mean": unl_prob_mean,
                            }
                            wandb_run.log(metrics, step=global_step)

                train_epoch_loss = epoch_total_loss / steps if steps > 0 else None

                val_stats = run_validation_pass(
                    args=args,
                    backbone=backbone,
                    head=head,
                    val_loader=val_loader,
                    val_sampler=val_sampler,
                    tf_names=tf_names,
                    tf_gene_lookup=tf_gene_lookup,
                    device=device,
                )
                val_metrics = {key: val_stats.get(key) for key in VAL_METRIC_HEADERS}
                append_metric(metrics_path, epoch, train_epoch_loss, val_metrics)

                if wandb_run is not None:
                    payload = {"epoch": epoch}
                    if train_epoch_loss is not None:
                        payload["train/epoch_loss"] = train_epoch_loss
                    if val_stats.get("val_pos_logit_mean") is not None:
                        payload["val/pos_logit_mean"] = val_stats["val_pos_logit_mean"]
                    if val_stats.get("val_unl_logit_mean") is not None:
                        payload["val/unl_logit_mean"] = val_stats["val_unl_logit_mean"]
                    if val_stats.get("val_pos_prob_mean") is not None:
                        payload["val/pos_prob_mean"] = val_stats["val_pos_prob_mean"]
                    if val_stats.get("val_unl_prob_mean") is not None:
                        payload["val/unl_prob_mean"] = val_stats["val_unl_prob_mean"]
                    wandb_metric_map = {
                        "val_pu_loss": "val/pu_loss",
                        "val_mse": "val/mse",
                        "val_auroc": "val/auroc",
                        "val_auprc": "val/auprc",
                        "val_topk_hit_rate": f"val/topk_hit_rate@{args.val_topk}",
                        "val_degree_mean": "val/tf_degree_mean",
                        "val_degree_median": "val/tf_degree_median",
                        "val_degree_p95": "val/tf_degree_p95",
                        "val_degree_max": "val/tf_degree_max",
                        "val_pvalue_min": "val/tf_pvalue_min",
                        "val_pvalue_median": "val/tf_pvalue_median",
                    }
                    for key, wandb_key in wandb_metric_map.items():
                        value = val_metrics.get(key)
                        if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                            payload[wandb_key] = float(value)
                    wandb_run.log(payload, step=global_step)

                if val_stats.get("val_pu_loss") is not None:
                    def _disp(value: Optional[float]) -> str:
                        if value is None:
                            return "NA"
                        value = float(value)
                        if math.isnan(value) or math.isinf(value):
                            return "NA"
                        return f"{value:.4f}"

                    topk_label = (
                        f"top{args.val_topk}" if args.val_topk and args.val_topk > 0 else "topK"
                    )
                    print(
                        "[val] epoch {epoch}: pu_loss={pu}, mse={mse}, auroc={auroc}, auprc={auprc}, "
                        "{topk_label}={topk}, pos_logit={pos_logit}, unl_logit={unl_logit}, "
                        "tf_deg_mean={deg_mean}, tf_deg_p95={deg_p95}, tf_pval_min={pval_min}".format(
                            epoch=epoch,
                            pu=_disp(val_stats.get("val_pu_loss")),
                            mse=_disp(val_stats.get("val_mse")),
                            auroc=_disp(val_stats.get("val_auroc")),
                            auprc=_disp(val_stats.get("val_auprc")),
                            topk_label=topk_label,
                            topk=_disp(val_stats.get("val_topk_hit_rate")),
                            pos_logit=_disp(val_stats.get("val_pos_logit_mean")),
                            unl_logit=_disp(val_stats.get("val_unl_logit_mean")),
                            deg_mean=_disp(val_stats.get("val_degree_mean")),
                            deg_p95=_disp(val_stats.get("val_degree_p95")),
                            pval_min=_disp(val_stats.get("val_pvalue_min")),
                        )
                    )

                ckpt_path = output_dir / f"head_epoch_{epoch}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "head_state": head.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                final_path = output_dir / "grn_head.pth"
                torch.save(head.state_dict(), final_path)
                print(f"[done] Saved fine-tuned GRN head to {final_path}")
                if wandb_run is not None:
                    wandb_run.log({"artifact/grn_head_path": str(final_path)}, step=global_step)
                if args.export_grn_each_epoch and args.export_grn_csv:
                    try:
                        export_grn_to_csv(
                            args=args,
                            output_dir=output_dir,
                            backbone=backbone,
                            head=head,
                            dataset=dataset,
                            tf_names=tf_names,
                            positive_links=positive_links,
                            device=device,
                            epoch=epoch,
                        )
                    except Exception as err:
                        print(f"[warn] Failed to export GRN CSV for epoch {epoch}: {err}")
        else:
            eval_stats = run_validation_pass(
                args=args,
                backbone=backbone,
                head=head,
                val_loader=val_loader,
                val_sampler=val_sampler,
                tf_names=tf_names,
                tf_gene_lookup=tf_gene_lookup,
                device=device,
            )
            val_metrics = {key: eval_stats.get(key) for key in VAL_METRIC_HEADERS}
            append_metric(metrics_path, 0, None, val_metrics)
            if wandb_run is not None:
                payload = {"epoch": 0}
                for field, wandb_field in (
                    ("val_pos_logit_mean", "val/pos_logit_mean"),
                    ("val_unl_logit_mean", "val/unl_logit_mean"),
                    ("val_pos_prob_mean", "val/pos_prob_mean"),
                    ("val_unl_prob_mean", "val/unl_prob_mean"),
                ):
                    if eval_stats.get(field) is not None:
                        payload[wandb_field] = eval_stats[field]
                for key, wandb_key in {
                    "val_pu_loss": "val/pu_loss",
                    "val_mse": "val/mse",
                    "val_auroc": "val/auroc",
                    "val_auprc": "val/auprc",
                    "val_topk_hit_rate": f"val/topk_hit_rate@{args.val_topk}",
                    "val_degree_mean": "val/tf_degree_mean",
                    "val_degree_median": "val/tf_degree_median",
                    "val_degree_p95": "val/tf_degree_p95",
                    "val_degree_max": "val/tf_degree_max",
                    "val_pvalue_min": "val/tf_pvalue_min",
                    "val_pvalue_median": "val/tf_pvalue_median",
                }.items():
                    value = val_metrics.get(key)
                    if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                        payload[wandb_key] = float(value)
                wandb_run.log(payload, step=global_step)
            if eval_stats.get("val_pu_loss") is not None:
                def _disp_eval(value: Optional[float]) -> str:
                    if value is None:
                        return "NA"
                    value = float(value)
                    if math.isnan(value) or math.isinf(value):
                        return "NA"
                    return f"{value:.4f}"

                topk_label = (
                    f"top{args.val_topk}" if args.val_topk and args.val_topk > 0 else "topK"
                )
                print(
                    "[eval] pu_loss={pu}, mse={mse}, auroc={auroc}, auprc={auprc}, {topk_label}={topk}, "
                    "pos_logit={pos_logit}, unl_logit={unl_logit}, tf_deg_mean={deg_mean}, tf_deg_p95={deg_p95}, "
                    "tf_pval_min={pval_min}".format(
                        pu=_disp_eval(eval_stats.get("val_pu_loss")),
                        mse=_disp_eval(eval_stats.get("val_mse")),
                        auroc=_disp_eval(eval_stats.get("val_auroc")),
                        auprc=_disp_eval(eval_stats.get("val_auprc")),
                        topk_label=topk_label,
                        topk=_disp_eval(eval_stats.get("val_topk_hit_rate")),
                        pos_logit=_disp_eval(eval_stats.get("val_pos_logit_mean")),
                        unl_logit=_disp_eval(eval_stats.get("val_unl_logit_mean")),
                        deg_mean=_disp_eval(eval_stats.get("val_degree_mean")),
                        deg_p95=_disp_eval(eval_stats.get("val_degree_p95")),
                        pval_min=_disp_eval(eval_stats.get("val_pvalue_min")),
                    )
                )
            else:
                print("[eval] Validation metrics unavailable; ensure --val_fraction > 0 or provide val data.")
        if args.export_grn_csv:
            try:
                export_grn_to_csv(
                    args=args,
                    output_dir=output_dir,
                    backbone=backbone,
                    head=head,
                    dataset=dataset,
                    tf_names=tf_names,
                    positive_links=positive_links,
                    device=device,
                )
            except Exception as err:
                print(f"[warn] Failed to export GRN CSV: {err}")
    finally:
        if output_dir is not None and start_time is not None:
            runtime = {
                "elapsed_sec": float(time.time() - start_time),
                "device": str(device) if device is not None else None,
                "cuda_max_memory_allocated_bytes": None,
                "cuda_max_memory_reserved_bytes": None,
            }
            if device is not None and device.type == "cuda":
                try:
                    runtime["cuda_max_memory_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated(device)
                    )
                    runtime["cuda_max_memory_reserved_bytes"] = int(
                        torch.cuda.max_memory_reserved(device)
                    )
                except Exception:
                    runtime["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                    runtime["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            try:
                (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2))
            except Exception as err:
                print(f"[warn] Failed to write runtime.json: {err}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    train()
