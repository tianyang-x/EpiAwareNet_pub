import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:  # Optional dependency for experiment tracking
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from epiaware import RNADataset
from epiaware.data import PositiveUnlabeledSampler, load_positive_links, load_tf_list, tf_gene_lookup
from epiaware.models import GRNPredictorHead, GeneOnlyTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 fine-tuning for GRNs (RNA-only ablation: gene self-attention backbone)"
    )
    parser.add_argument("--rna_path", type=str, required=True)
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
    parser.add_argument("--backbone_checkpoint", type=str, required=True, help="Stage 1 RNA-only checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="output/epiaware_grn_rna_only")

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
        help="Weight applied to the non-negative negative-risk term in nnPU loss.",
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
            "Speeds up runs for large gene counts at the cost of a fixed neighbor graph (approximation)."
        ),
    )
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use_faiss_knn",
        action="store_true",
        help="Use FAISS for kNN gene attention",
        default=False,
    )
    parser.add_argument(
        "--self_attn_chunk_size",
        type=int,
        default=1024,
        help="Process genes in this many-sized chunks inside gene self-attention.",
    )

    parser.add_argument(
        "--head_hidden",
        type=int,
        default=256,
        help="Hidden dimension used in the GRN predictor head.",
    )
    parser.add_argument("--head_depth", type=int, default=2)

    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of cells reserved for validation (set to 0 to disable).",
    )
    parser.add_argument("--val_cell_batch_size", type=int, default=None)
    parser.add_argument("--val_pos_batch_size", type=int, default=None)
    parser.add_argument("--val_unl_batch_size", type=int, default=None)
    parser.add_argument(
        "--val_prob_threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold used by some Stage2 runners for discrete validation metrics. "
            "This RNA-only script currently reports threshold-free metrics (AUROC/AUPRC), but accepts the flag "
            "for compatibility with shared ablation scripts."
        ),
    )
    parser.add_argument(
        "--val_topk",
        type=int,
        default=50,
        help="Top-K used when computing validation hit-rate and enrichment metrics.",
    )

    parser.add_argument(
        "--head_checkpoint",
        type=str,
        default=None,
        help="Optional head checkpoint to load (used with --eval_only or for export-only runs).",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only run validation/export using a provided head checkpoint.",
    )

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
        "--export_logit_path",
        type=str,
        default=None,
        help="Optional path to store averaged TF-target logits; when present and file exists, reuse instead of recomputing.",
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
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
    )

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def init_wandb_run(args: argparse.Namespace) -> Optional[object]:
    if not args.enable_wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError(
            "Weights & Biases is not installed. Install it with `pip install wandb` "
            "or run without --enable_wandb."
        )
    if not args.wandb_project:
        raise ValueError("Specify --wandb_project when using --enable_wandb.")

    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=config,
    )
    return run


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    rank_sum = ranks[pos_mask].sum() + n_pos
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    total_pos = float(labels_sorted.sum())
    if total_pos <= 0:
        return float("nan")
    cum_pos = 0.0
    precision_sum = 0.0
    for idx, label in enumerate(labels_sorted, start=1):
        if label <= 0:
            continue
        cum_pos += 1.0
        precision_sum += cum_pos / float(idx)
    return float(precision_sum / total_pos)


def nn_pu_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    prior: float,
    negative_risk_weight: float = 1.0,
) -> torch.Tensor:
    # y_true: 1 for positive, 0 for unlabeled
    y_true = y_true.float()
    pos_mask = y_true == 1.0
    unl_mask = y_true == 0.0

    if pos_mask.any():
        pos_logits = logits[pos_mask]
        pos_loss = torch.nn.functional.softplus(-pos_logits).mean()
        pos_as_neg_loss = torch.nn.functional.softplus(pos_logits).mean()
    else:
        pos_loss = logits.new_zeros(())
        pos_as_neg_loss = logits.new_zeros(())

    if unl_mask.any():
        unl_logits = logits[unl_mask]
        unl_neg_loss = torch.nn.functional.softplus(unl_logits).mean()
    else:
        unl_neg_loss = logits.new_zeros(())

    negative_risk = unl_neg_loss - prior * pos_as_neg_loss
    negative_risk = torch.clamp(negative_risk, min=0.0) * float(negative_risk_weight)
    risk = prior * pos_loss + negative_risk
    return risk


def gather_logits(head: GRNPredictorHead, gene_states: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    """Return mean logits over cells for each TF-target pair.

    For large pair batches (notably GRN export), compute in chunks to avoid
    excessive peak GPU memory from advanced indexing.
    """

    cells = int(gene_states.size(0))
    pair_count = int(pair_idx.size(0))
    if pair_count == 0:
        return torch.empty((0,), device=gene_states.device, dtype=gene_states.dtype)

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

    chunks: List[torch.Tensor] = []
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


def load_geneonly_backbone(args: argparse.Namespace, dataset: RNADataset, device: torch.device) -> GeneOnlyTransformer:
    backbone = GeneOnlyTransformer(
        num_genes=len(dataset.gene_names),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        k_neighbors=args.k_neighbors,
        ffn_hidden=args.ffn_dim,
        dropout=args.dropout,
        use_faiss_knn=args.use_faiss_knn,
        self_chunk_size=args.self_attn_chunk_size,
        reuse_knn_indices=args.reuse_knn_indices,
    )
    payload = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=True)
    # Stage1 RNA-only currently saves either:
    # 1) a raw state_dict / OrderedDict (most common)
    # 2) a wrapped dict with a nested state dict under common keys
    if isinstance(payload, dict):
        if "model_state" in payload and isinstance(payload["model_state"], dict):
            state = payload["model_state"]
        elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state = payload["state_dict"]
        elif "model" in payload and isinstance(payload["model"], dict):
            state = payload["model"]
        else:
            # Assume the payload itself is already a state_dict-like mapping.
            state = payload
    else:
        state = payload
    if not isinstance(state, dict):
        raise ValueError(
            f"Unexpected checkpoint format at {args.backbone_checkpoint}: {type(payload)}"
        )

    # Stage1 RNA-only saves GeneOnlyPretrainingModel state dict, where backbone params may be under "backbone.".
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            filtered[key[len("backbone.") :]] = value
        elif key.startswith("transformer."):
            filtered[key[len("transformer.") :]] = value
        else:
            # allow direct loading when keys match GeneOnlyTransformer
            filtered[key] = value

    missing = backbone.load_state_dict(filtered, strict=False)
    if missing.missing_keys:
        print(f"[warn] Missing keys when loading checkpoint: {missing.missing_keys[:10]}")
    if missing.unexpected_keys:
        print(f"[warn] Unexpected keys when loading checkpoint: {missing.unexpected_keys[:10]}")

    backbone = backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    return backbone


def export_grn_to_csv(
    args: argparse.Namespace,
    output_dir: Path,
    backbone: GeneOnlyTransformer,
    head: GRNPredictorHead,
    dataset: RNADataset,
    tf_names: Sequence[str],
    positive_links: Sequence[Tuple[str, str]],
    device: torch.device,
) -> None:
    if not args.export_grn_csv:
        return

    export_path = Path(args.export_grn_csv)
    if not export_path.is_absolute():
        try:
            already_under_outdir = export_path.is_relative_to(output_dir)
        except AttributeError:  # pragma: no cover - py<3.9
            already_under_outdir = export_path.parts[: len(output_dir.parts)] == output_dir.parts
        if not already_under_outdir:
            export_path = output_dir / export_path
    export_path.parent.mkdir(parents=True, exist_ok=True)

    gene_to_idx = {name: idx for idx, name in enumerate(dataset.gene_names)}
    valid_tf_indices: List[int] = []
    valid_tf_names: List[str] = []
    for name in tf_names:
        idx = gene_to_idx.get(name)
        if idx is None:
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
        cache_path = Path(args.export_logit_path)
        if not cache_path.is_absolute():
            try:
                already_under_outdir = cache_path.is_relative_to(output_dir)
            except AttributeError:  # pragma: no cover - py<3.9
                already_under_outdir = cache_path.parts[: len(output_dir.parts)] == output_dir.parts
            if not already_under_outdir:
                cache_path = output_dir / cache_path
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
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
            for batch in tqdm(export_loader, desc="Exporting GRN (RNA-only)", leave=False):
                rna = batch["rna"].to(device)
                gene_states = backbone(rna)
                cells_in_batch = gene_states.size(0)
                total_cells += cells_in_batch

                for tf_start in range(0, num_tfs, tf_chunk):
                    tf_end = min(tf_start + tf_chunk, num_tfs)
                    tf_idx_chunk = tf_indices_tensor[tf_start:tf_end]
                    if tf_idx_chunk.numel() == 0:
                        continue
                    for tgt_start in range(0, num_targets, tgt_chunk):
                        tgt_end = min(tgt_start + tgt_chunk, num_targets)
                        tgt_idx_chunk = torch.arange(tgt_start, tgt_end, dtype=torch.long, device=device)
                        if tgt_idx_chunk.numel() == 0:
                            continue
                        try:
                            tf_grid, tgt_grid = torch.meshgrid(tf_idx_chunk, tgt_idx_chunk, indexing="ij")
                        except TypeError:
                            tf_grid, tgt_grid = torch.meshgrid(tf_idx_chunk, tgt_idx_chunk)
                        pair_idx = torch.stack((tf_grid.reshape(-1), tgt_grid.reshape(-1)), dim=1)
                        logits = gather_logits(head, gene_states, pair_idx)
                        chunk_logits = logits.reshape(tf_end - tf_start, tgt_end - tgt_start)
                        logit_sums[tf_start:tf_end, tgt_start:tgt_end] += chunk_logits.double().cpu() * cells_in_batch

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
    training_targets = {target for _, target in positive_links if target in gene_name_set}

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
    wandb_run: Optional[object] = None
    output_dir: Optional[Path] = None
    start_time: Optional[float] = None
    device: Optional[torch.device] = None
    try:
        wandb_run = init_wandb_run(args)
        set_seed(args.seed)
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

        dataset = RNADataset(rna_path=args.rna_path, gene_names_path=args.gene_names)

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
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
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

        backbone = load_geneonly_backbone(args, dataset, device)

        head = GRNPredictorHead(
            embed_dim=args.embed_dim,
            hidden_dim=args.head_hidden,
            depth=args.head_depth,
            dropout=args.dropout,
        ).to(device)

        if args.head_checkpoint:
            payload = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("head_state") if isinstance(payload, dict) else payload
            missing = head.load_state_dict(state, strict=False)
            if missing.missing_keys:
                print(f"[warn] Missing keys when loading head checkpoint: {missing.missing_keys[:10]}")

        tf_names = load_tf_list(args.tf_list, dataset.gene_names, prefix=args.tf_prefix)
        positive_links = load_positive_links(
            args.positive_links,
            tf_prefix=args.positive_tf_prefix,
            target_prefix=args.positive_target_prefix,
        )
        val_links_path = args.val_positive_links or args.positive_links
        val_positive_links = load_positive_links(
            val_links_path,
            tf_prefix=args.positive_tf_prefix,
            target_prefix=args.positive_target_prefix,
        )

        _ = tf_gene_lookup(tf_names, dataset.gene_names)
        sampler = PositiveUnlabeledSampler(tf_names, dataset.gene_names, positive_links, seed=args.seed)
        val_sampler = PositiveUnlabeledSampler(tf_names, dataset.gene_names, val_positive_links, seed=args.seed + 1)
        
        def sample_pairs(
            pu_sampler: PositiveUnlabeledSampler,
            pos_batch_size: int,
            unl_batch_size: int,
            device: torch.device,
        ) -> Tuple[torch.LongTensor, torch.Tensor]:
            pos_bs = max(1, int(pos_batch_size))
            unl_bs = max(1, int(unl_batch_size))
            pos_pairs = pu_sampler.sample_positive(pos_bs)
            unl_pairs = pu_sampler.sample_unlabeled(unl_bs)
            pair_idx = torch.cat([pos_pairs, unl_pairs], dim=0).to(device)
            labels = torch.cat(
                [
                    torch.ones(pos_pairs.size(0), dtype=torch.float32),
                    torch.zeros(unl_pairs.size(0), dtype=torch.float32),
                ],
                dim=0,
            ).to(device)
            perm = torch.randperm(pair_idx.size(0), device=device)
            return pair_idx[perm], labels[perm]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

        start_time = time.time()
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                torch.cuda.reset_peak_memory_stats()

        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        global_step = 0

        def run_validation() -> Dict[str, float]:
            metrics: Dict[str, float] = {}
            if val_loader is None:
                return metrics
            head.eval()
            all_labels: List[float] = []
            all_scores: List[float] = []
            mse_sum = 0.0
            mse_count = 0
            loss_sum = 0.0
            steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    rna = batch["rna"].to(device)
                    gene_states = backbone(rna)
                    pair_idx, labels_f = sample_pairs(
                        val_sampler,
                        args.val_pos_batch_size or args.pos_batch_size,
                        args.val_unl_batch_size or args.unl_batch_size,
                        device,
                    )
                    logits = gather_logits(head, gene_states, pair_idx)
                    if args.use_bce_loss:
                        loss = bce_loss(logits, labels_f)
                    else:
                        loss = nn_pu_loss(
                            logits=logits,
                            y_true=labels_f,
                            prior=float(args.positive_prior),
                            negative_risk_weight=float(args.negative_risk_weight),
                        )
                    loss_sum += float(loss.item())
                    steps += 1
                    probs = torch.sigmoid(logits)
                    all_labels.extend(labels_f.detach().cpu().numpy().tolist())
                    all_scores.extend(probs.detach().cpu().numpy().tolist())
                    mse_sum += float(torch.nn.functional.mse_loss(probs, labels_f).item())
                    mse_count += 1

            labels_np = np.asarray(all_labels, dtype=np.float64)
            scores_np = np.asarray(all_scores, dtype=np.float64)
            metrics["val_loss"] = loss_sum / max(steps, 1)
            metrics["val_mse"] = mse_sum / max(mse_count, 1)
            metrics["val_auroc"] = _compute_auc(labels_np, scores_np)
            metrics["val_auprc"] = _average_precision(labels_np, scores_np)
            head.train()
            return metrics

        if args.eval_only:
            if not args.head_checkpoint:
                raise ValueError("--eval_only requires --head_checkpoint")
            val_metrics = run_validation()
            print(
                "[eval] loss={loss}, mse={mse}, auroc={auroc}, auprc={auprc}".format(
                    loss=f"{val_metrics.get('val_loss', float('nan')):.4f}",
                    mse=f"{val_metrics.get('val_mse', float('nan')):.4f}",
                    auroc=f"{val_metrics.get('val_auroc', float('nan')):.4f}",
                    auprc=f"{val_metrics.get('val_auprc', float('nan')):.4f}",
                )
            )
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
            return

        head.train()
        metrics_path = output_dir / "training_metrics.csv"
        if not metrics_path.exists():
            metrics_path.write_text("epoch,train_loss,val_loss,val_mse,val_auroc,val_auprc\n")

        for epoch in range(1, args.epochs + 1):
            epoch_loss_sum = 0.0
            epoch_steps = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                rna = batch["rna"].to(device)
                with torch.no_grad():
                    gene_states = backbone(rna)
                pair_idx, labels_f = sample_pairs(
                    sampler,
                    args.pos_batch_size,
                    args.unl_batch_size,
                    device,
                )
                logits = gather_logits(head, gene_states, pair_idx)
                if args.use_bce_loss:
                    loss = bce_loss(logits, labels_f)
                else:
                    loss = nn_pu_loss(
                        logits=logits,
                        y_true=labels_f,
                        prior=float(args.positive_prior),
                        negative_risk_weight=float(args.negative_risk_weight),
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.item())
                epoch_steps += 1
                global_step += 1

                if args.save_every_steps and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    ckpt_path = output_dir / f"head_step_{global_step}.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "head_state": head.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                        },
                        ckpt_path,
                    )

            train_loss = epoch_loss_sum / max(epoch_steps, 1)
            val_metrics = run_validation()
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch},{train_loss:.6f},{val_metrics.get('val_loss', float('nan')):.6f},"
                    f"{val_metrics.get('val_mse', float('nan')):.6f},{val_metrics.get('val_auroc', float('nan')):.6f},"
                    f"{val_metrics.get('val_auprc', float('nan')):.6f}\n"
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train()
