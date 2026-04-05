import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # Optional dependency for experiment tracking
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import numpy as np

from epiaware import CandidateSet, MultiOmicDataset
from epiaware.models import (
    EpiAwarePretrainingModel,
    EpiAwareTransformer,
    MaskedNegativeBinomialHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 pretraining for EpiAwareNet")
    parser.add_argument("--rna_path", type=str, required=True, help="RNA matrix path (.npy/.npz/.h5ad)")
    parser.add_argument("--atac_path", type=str, required=True, help="ATAC matrix path (.npy/.npz/.h5ad)")
    parser.add_argument("--candidate_file", type=str, help="Candidate peak set JSON/TSV")
    parser.add_argument("--gene_names", type=str, default=None, help="Optional gene name list (txt/json)")
    parser.add_argument("--peak_names", type=str, default=None, help="Optional peak name list (txt/json)")
    parser.add_argument("--output_dir", type=str, default="output/epiaware_backbone", help="Directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--topk_peaks", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--recon_hidden", type=int, default=512)
    parser.add_argument("--recon_depth", type=int, default=12)
    parser.add_argument("--use_faiss_knn", action="store_true", help="Use FAISS for kNN gene attention", default=False)
    parser.add_argument(
        "--reuse_knn_indices",
        action="store_true",
        default=False,
        help=(
            "Reuse gene-gene kNN indices across all transformer blocks during eval/validation. "
            "Speeds up runs for large gene counts (approximation; training is unchanged unless you also evaluate with it)."
        ),
    )
    parser.add_argument(
        "--self_attn_chunk_size",
        type=int,
        default=1024,
        help="Process genes in this many-sized chunks inside gene self-attention.",
    )
    parser.add_argument(
        "--cross_chunk_size",
        type=int,
        default=1024,
        help="Process genes in this many-sized chunks inside cross-attention to limit memory.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of cells reserved for validation (set to 0 to disable).",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Validation batch size (defaults to train batch size).",
    )
    parser.add_argument(
        "--val_mask_ratio",
        type=float,
        default=None,
        help="Mask ratio used during validation (defaults to train mask_ratio).",
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
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="Number of micro-batches to accumulate before each optimizer step",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0")
    return parser.parse_args()


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
    # W&B specific fields are still useful for reference but not when logging metrics
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
    run.define_metric("epoch")
    run.define_metric("train/epoch_loss", step_metric="epoch")
    run.define_metric("val/nb_loss", step_metric="epoch")
    run.define_metric("val/mse", step_metric="epoch")
    run.define_metric("val/mask_coverage", step_metric="epoch")
    run.define_metric("val/mae", step_metric="epoch")
    run.define_metric("val/pearson", step_metric="epoch")
    run.define_metric("val/spearman", step_metric="epoch")
    run.define_metric("val/r2", step_metric="epoch")
    run.define_metric("val/zero_auc", step_metric="epoch")
    run.define_metric("val/zero_ap", step_metric="epoch")
    return run


VAL_METRIC_HEADERS = [
    "val_nb_loss",
    "val_mse",
    "val_mae",
    "val_pearson",
    "val_spearman",
    "val_r2",
    "val_zero_auc",
    "val_zero_ap",
    "val_mask_coverage",
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mask(batch: int, genes: int, ratio: float, device: torch.device) -> torch.Tensor:
    mask = (torch.rand(batch, genes, device=device) < ratio).float()
    empty = mask.sum(dim=-1) == 0
    if empty.any():
        num_empty = empty.sum().item()
        replacement = torch.zeros_like(mask[empty])
        rand_idx = torch.randint(0, genes, (num_empty,), device=device)
        replacement[torch.arange(num_empty, device=device), rand_idx] = 1.0
        mask[empty] = replacement
    return mask


def save_checkpoint(state: Dict, output_dir: Path, tag: str) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / f"{tag}.pth")


def append_metric(
    path: Path,
    epoch: int,
    train_loss: Optional[float],
    val_metrics: Dict[str, Optional[float]],
) -> None:
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8") as handle:
        headers = VAL_METRIC_HEADERS
        if write_header:
            handle.write(",".join(["epoch", "train_loss", *headers]) + "\n")

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
        for key in headers:
            row.append(fmt(val_metrics.get(key)))
        handle.write(",".join(row) + "\n")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _batch_correlations(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Tuple[float, int, float, int]:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    pearson_sum = 0.0
    pearson_count = 0
    spearman_sum = 0.0
    spearman_count = 0
    for i in range(mask_np.shape[0]):
        idx = mask_np[i]
        if idx.sum() < 2:
            continue
        x = pred_np[i, idx]
        y = target_np[i, idx]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denom > 0:
            pearson_sum += float(np.dot(x_centered, y_centered) / denom)
            pearson_count += 1
        x_ranks = _rankdata(x)
        y_ranks = _rankdata(y)
        xr_centered = x_ranks - x_ranks.mean()
        yr_centered = y_ranks - y_ranks.mean()
        denom_r = np.linalg.norm(xr_centered) * np.linalg.norm(yr_centered)
        if denom_r > 0:
            spearman_sum += float(np.dot(xr_centered, yr_centered) / denom_r)
            spearman_count += 1
    return pearson_sum, pearson_count, spearman_sum, spearman_count


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0:
        return float("nan")
    labels = labels.astype(bool)
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(order.size)
    rank_sum = ranks[labels].sum() + pos  # convert to 1-based rank
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0:
        return float("nan")
    labels = labels.astype(bool)
    total_pos = labels.sum()
    if total_pos == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    cum_pos = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx]:
            cum_pos += 1
            precision_sum += cum_pos / rank
    return float(precision_sum / total_pos)


def train() -> None:
    args = parse_args()
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
    output_dir: Optional[Path] = None
    start_time: Optional[float] = None
    device: Optional[torch.device] = None
    try:
        wandb_run = init_wandb_run(args)
        set_seed(args.seed)
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
            # generate a dummy candidate set with all peaks
            candidate_set = CandidateSet.generate_full_candidate_set(
                gene_names=dataset.gene_names,
                peak_names=dataset.peak_names,
            )                                                               
        total_cells = len(dataset)
        val_fraction = max(0.0, min(1.0, args.val_fraction))
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
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False,
        )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_batch_size = args.val_batch_size or args.batch_size
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
        if wandb_run is not None:
            wandb_run.config.update(
                {
                    "train_cell_count": len(train_dataset),
                    "val_cell_count": len(val_dataset) if val_dataset is not None else 0,
                },
                allow_val_change=True,
            )
        device = torch.device(
            args.device
            if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
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
            use_faiss_knn=args.use_faiss_knn,
            cross_chunk_size=args.cross_chunk_size,
            self_chunk_size=args.self_attn_chunk_size,
            reuse_knn_indices=args.reuse_knn_indices,
        )
        recon_head = MaskedNegativeBinomialHead(
            embed_dim=args.embed_dim,
            hidden_dim=args.recon_hidden,
            depth=args.recon_depth,
            dropout=args.dropout,
        )
        model = EpiAwarePretrainingModel(backbone, recon_head).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * max(len(train_loader), 1)
        )
        accum_steps = max(1, args.grad_accum_steps)
        amp_enabled = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast_context = torch.cuda.amp.autocast if amp_enabled else nullcontext
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(vars(args), indent=2))

        start_time = time.time()
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                torch.cuda.reset_peak_memory_stats()

        metrics_path = output_dir / "training_metrics.csv"
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            epoch_total_loss = 0.0
            epoch_steps = 0
            optimizer.zero_grad(set_to_none=True)
            with tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}") as progress:
                for batch in progress:
                    rna = batch["rna"].to(device)
                    atac = batch["atac"].to(device)
                    mask = make_mask(rna.size(0), rna.size(1), args.mask_ratio, device)
                    rna_chunks = torch.chunk(rna, accum_steps, dim=0)
                    atac_chunks = torch.chunk(atac, accum_steps, dim=0)
                    mask_chunks = torch.chunk(mask, accum_steps, dim=0)
                    micro_batches = [
                        (rna_micro, atac_micro, mask_micro)
                        for rna_micro, atac_micro, mask_micro in zip(rna_chunks, atac_chunks, mask_chunks)
                        if rna_micro.size(0) > 0
                    ]
                    if not micro_batches:
                        continue
                    micro_count = len(micro_batches)
                    step_loss_value = 0.0
                    for rna_micro, atac_micro, mask_micro in micro_batches:
                        with autocast_context():
                            outputs = model.compute_loss(rna_micro, atac_micro, mask_micro)
                            micro_loss = outputs["loss"]
                        step_loss_value += micro_loss.item()
                        scaled_loss = micro_loss / micro_count
                        scaler.scale(scaled_loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    running_loss += step_loss_value
                    epoch_total_loss += step_loss_value
                    epoch_steps += 1
                    global_step += 1
                    if wandb_run is not None:
                        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                        metrics = {
                            "train/global_step": global_step,
                            "train/step_loss": step_loss_value / micro_count,
                            "train/step_loss_sum": step_loss_value,
                            "train/lr": lr,
                        }
                        if amp_enabled:
                            metrics["train/grad_scale"] = scaler.get_scale()
                        wandb_run.log(metrics, step=global_step)
                    if global_step % args.log_every == 0:
                        progress.set_postfix(loss=running_loss / args.log_every)
                        running_loss = 0.0
            train_epoch_loss = (
                epoch_total_loss / epoch_steps if epoch_steps > 0 else None
            )
            val_metrics: Dict[str, Optional[float]] = {key: None for key in VAL_METRIC_HEADERS}
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                val_steps = 0
                val_mse_sum = 0.0
                val_abs_sum = 0.0
                val_mask_tokens = 0.0
                val_total_tokens = 0.0
                val_sst_sum = 0.0
                val_pearson_sum = 0.0
                val_pearson_count = 0
                val_spearman_sum = 0.0
                val_spearman_count = 0
                zero_labels: List[np.ndarray] = []
                zero_scores: List[np.ndarray] = []
                val_mask_ratio = (
                    args.val_mask_ratio if args.val_mask_ratio is not None else args.mask_ratio
                )
                with torch.no_grad():
                    for batch in val_loader:
                        rna = batch["rna"].to(device)
                        atac = batch["atac"].to(device)
                        val_mask = make_mask(
                            rna.size(0), rna.size(1), val_mask_ratio, device
                        )
                        with autocast_context():
                            outputs = model.compute_loss(rna, atac, val_mask)
                            val_loss = outputs["loss"]
                        val_loss_sum += val_loss.item()
                        val_steps += 1
                        mean = outputs["mean"]
                        dispersion = outputs["dispersion"]
                        diff = (mean - rna) * val_mask
                        val_mse_sum += diff.pow(2).sum().item()
                        val_abs_sum += diff.abs().sum().item()
                        val_mask_tokens += val_mask.sum().item()
                        val_total_tokens += float(val_mask.numel())
                        pearson_batch, pearson_count, spearman_batch, spearman_count = _batch_correlations(
                            mean, rna, val_mask
                        )
                        val_pearson_sum += pearson_batch
                        val_pearson_count += pearson_count
                        val_spearman_sum += spearman_batch
                        val_spearman_count += spearman_count
                        mask_np = val_mask.detach().cpu().numpy().astype(bool)
                        rna_np = rna.detach().cpu().numpy()
                        for i in range(mask_np.shape[0]):
                            idx = mask_np[i]
                            if idx.sum() == 0:
                                continue
                            target_vals = rna_np[i, idx]
                            centered = target_vals - target_vals.mean()
                            val_sst_sum += float((centered ** 2).sum())
                        zero_mask = val_mask > 0
                        if zero_mask.any():
                            zero_scores.append(
                                (
                                    (dispersion / (dispersion + mean))
                                    .pow(dispersion)
                                    .masked_select(zero_mask)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                            )
                            zero_labels.append(
                                (
                                    rna.eq(0)
                                    .masked_select(zero_mask)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                            )
                if val_steps > 0:
                    val_metrics["val_nb_loss"] = val_loss_sum / val_steps
                if val_mask_tokens > 0:
                    val_metrics["val_mse"] = val_mse_sum / val_mask_tokens
                    val_metrics["val_mae"] = val_abs_sum / val_mask_tokens
                if val_total_tokens > 0:
                    val_metrics["val_mask_coverage"] = val_mask_tokens / val_total_tokens
                if val_sst_sum > 0:
                    val_metrics["val_r2"] = 1.0 - (val_mse_sum / val_sst_sum)
                if val_pearson_count > 0:
                    val_metrics["val_pearson"] = val_pearson_sum / val_pearson_count
                if val_spearman_count > 0:
                    val_metrics["val_spearman"] = val_spearman_sum / val_spearman_count
                if zero_scores:
                    labels_concat = np.concatenate(zero_labels)
                    scores_concat = np.concatenate(zero_scores)
                    val_metrics["val_zero_auc"] = _compute_auc(labels_concat, scores_concat)
                    val_metrics["val_zero_ap"] = _average_precision(labels_concat, scores_concat)
                if val_total_tokens > 0:
                    val_metrics.setdefault("val_mask_coverage", val_mask_tokens / val_total_tokens)
                model.train()
            append_metric(metrics_path, epoch, train_epoch_loss, val_metrics)
            if wandb_run is not None:
                log_payload = {
                    "epoch": epoch,
                }
                if train_epoch_loss is not None:
                    log_payload["train/epoch_loss"] = train_epoch_loss
                wandb_metric_map = {
                    "val_nb_loss": "val/nb_loss",
                    "val_mse": "val/mse",
                    "val_mae": "val/mae",
                    "val_mask_coverage": "val/mask_coverage",
                    "val_pearson": "val/pearson",
                    "val_spearman": "val/spearman",
                    "val_r2": "val/r2",
                    "val_zero_auc": "val/zero_auc",
                    "val_zero_ap": "val/zero_ap",
                }
                for key, wandb_key in wandb_metric_map.items():
                    value = val_metrics.get(key)
                    if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                        log_payload[wandb_key] = float(value)
                wandb_run.log(log_payload, step=global_step)
            if val_metrics.get("val_nb_loss") is not None:
                def _disp(val: Optional[float]) -> str:
                    if val is None:
                        return "NA"
                    try:
                        val_f = float(val)
                    except (TypeError, ValueError):
                        return "NA"
                    if math.isnan(val_f) or math.isinf(val_f):
                        return "NA"
                    return f"{val_f:.4f}"

                print(
                    "[val] epoch {epoch}: nb_loss={nb}, mse={mse}, mae={mae}, "
                    "pearson={pearson}, spearman={spearman}, r2={r2}, "
                    "zero_auc={auc}, zero_ap={ap}".format(
                        epoch=epoch,
                        nb=_disp(val_metrics.get("val_nb_loss")),
                        mse=_disp(val_metrics.get("val_mse")),
                        mae=_disp(val_metrics.get("val_mae")),
                        pearson=_disp(val_metrics.get("val_pearson")),
                        spearman=_disp(val_metrics.get("val_spearman")),
                        r2=_disp(val_metrics.get("val_r2")),
                        auc=_disp(val_metrics.get("val_zero_auc")),
                        ap=_disp(val_metrics.get("val_zero_ap")),
                    )
                )
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    output_dir,
                    tag=f"epoch_{epoch}",
                )
        final_path = output_dir / "epiaware_backbone.pth"
        torch.save(model.state_dict(), final_path)
        print(f"[done] Saved pretrained backbone to {final_path}")
        if wandb_run is not None:
            wandb_run.log({"artifact/backbone_path": str(final_path)}, step=global_step)
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
