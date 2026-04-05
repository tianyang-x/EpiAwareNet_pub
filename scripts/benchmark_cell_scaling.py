#!/usr/bin/env python3
"""Benchmark preprocessing / train-step / inference scaling over cell counts.

This script keeps the gene/peak vocabulary and candidate set fixed while
subsampling different numbers of cells from an existing dataset.  For each
requested cell count it measures:
  * Preprocessing time (loading + log1p + tensorization)
  * Average training-step latency (forward + backward) for the Stage 1 model
  * Average inference latency (forward-only) for the Stage 1 model

To avoid multi-hour runs, the timing loops can cap the number of measured
mini-batches and extrapolate to a full epoch.  Results are written to CSV
and emitted as JSON lines so downstream notebooks can plot them easily.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epiaware import CandidateSet, EpiAwarePretrainingModel, EpiAwareTransformer, MaskedNegativeBinomialHead
from epiaware.data.multiomic_dataset import load_feature_names


def parse_int_list(text: str) -> List[int]:
    return [int(token) for token in text.replace(",", " ").split() if token.strip()]


def load_subset(
    *,
    rna_memmap: np.memmap,
    atac_memmap: np.memmap,
    cell_count: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Load + log1p a subset of cells and return tensors plus elapsed seconds."""
    total_cells = rna_memmap.shape[0]
    actual = min(cell_count, total_cells)
    rng = np.random.default_rng(seed)
    if actual >= total_cells:
        indices = np.arange(total_cells)
    else:
        indices = np.sort(rng.choice(total_cells, size=actual, replace=False))
    start = time.perf_counter()
    rna_np = np.asarray(rna_memmap[indices], dtype=np.float32)
    atac_np = np.asarray(atac_memmap[indices], dtype=np.float32)
    np.log1p(rna_np, out=rna_np)
    np.log1p(atac_np, out=atac_np)
    rna_tensor = torch.from_numpy(rna_np)
    atac_tensor = torch.from_numpy(atac_np)
    elapsed = time.perf_counter() - start
    return rna_tensor, atac_tensor, elapsed


def make_mask(batch_size: int, num_genes: int, ratio: float, device: torch.device) -> torch.Tensor:
    mask = (torch.rand(batch_size, num_genes, device=device) < ratio).float()
    empty = mask.sum(dim=-1) == 0
    if empty.any():
        num_empty = empty.sum().item()
        replacement = torch.zeros_like(mask[empty])
        rand_idx = torch.randint(0, num_genes, (num_empty,), device=device)
        replacement[torch.arange(num_empty, device=device), rand_idx] = 1.0
        mask[empty] = replacement
    return mask


def build_model(
    *,
    num_genes: int,
    num_peaks: int,
    candidate_set: CandidateSet,
    args: argparse.Namespace,
    device: torch.device,
) -> EpiAwarePretrainingModel:
    backbone = EpiAwareTransformer(
        num_genes=num_genes,
        num_peaks=num_peaks,
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
        self_chunk_size=args.self_chunk_size,
        reuse_knn_indices=args.reuse_knn_indices,
    )
    recon_head = MaskedNegativeBinomialHead(
        embed_dim=args.embed_dim,
        hidden_dim=args.recon_hidden,
        depth=args.recon_depth,
        dropout=args.dropout,
    )
    return EpiAwarePretrainingModel(backbone, recon_head).to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_training(
    model: EpiAwarePretrainingModel,
    loader: DataLoader,
    *,
    mask_ratio: float,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_batches: int,
    amp: bool,
) -> Dict[str, float]:
    total_batches = max(1, math.ceil(len(loader.dataset) / loader.batch_size))
    limit = total_batches if max_batches <= 0 else min(total_batches, max_batches)
    limit = max(1, limit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    steps = 0
    model.train()
    _sync(device)
    start = time.perf_counter()
    for rna, atac in loader:
        rna = rna.to(device, non_blocking=True)
        atac = atac.to(device, non_blocking=True)
        mask = make_mask(rna.size(0), rna.size(1), mask_ratio, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model.compute_loss(rna, atac, mask)
            loss = outputs["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        if steps >= limit:
            break
    _sync(device)
    elapsed = time.perf_counter() - start
    per_step = elapsed / max(1, steps)
    est_epoch = per_step * total_batches
    return {
        "steps_measured": steps,
        "measured_seconds": elapsed,
        "per_step_seconds": per_step,
        "estimated_epoch_seconds": est_epoch,
        "total_batches": total_batches,
    }


def measure_inference(
    model: EpiAwarePretrainingModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    amp: bool,
) -> Dict[str, float]:
    total_batches = max(1, math.ceil(len(loader.dataset) / loader.batch_size))
    limit = total_batches if max_batches <= 0 else min(total_batches, max_batches)
    limit = max(1, limit)
    model.eval()
    steps = 0
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for rna, atac in loader:
            rna = rna.to(device, non_blocking=True)
            atac = atac.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                _ = model(rna, atac)
            steps += 1
            if steps >= limit:
                break
    _sync(device)
    elapsed = time.perf_counter() - start
    per_step = elapsed / max(1, steps)
    est_epoch = per_step * total_batches
    return {
        "steps_measured": steps,
        "measured_seconds": elapsed,
        "per_step_seconds": per_step,
        "estimated_epoch_seconds": est_epoch,
        "total_batches": total_batches,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Stage1 scaling over cell counts.")
    parser.add_argument("--rna_path", required=True, help="Path to rna.npy")
    parser.add_argument("--atac_path", required=True, help="Path to atac.npy")
    parser.add_argument("--candidate_file", required=True, help="Candidate peaks JSON/TSV")
    parser.add_argument("--gene_names", required=True, help="Gene names txt/json")
    parser.add_argument("--peak_names", required=True, help="Peak names txt/json")
    parser.add_argument("--cell_counts", type=str, default="5000 10000", help="Space/comma separated list")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--max_train_batches", type=int, default=100, help="0 -> run all steps")
    parser.add_argument("--max_infer_batches", type=int, default=200, help="0 -> run all steps")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--topk_peaks", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--recon_hidden", type=int, default=512)
    parser.add_argument("--recon_depth", type=int, default=12)
    parser.add_argument("--cross_chunk_size", type=int, default=1024)
    parser.add_argument("--self_chunk_size", type=int, default=1024)
    parser.add_argument("--use_faiss_knn", action="store_true", default=False)
    parser.add_argument("--reuse_knn_indices", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None, help="Override torch device string")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--output_csv", type=str, default="output/scaling_benchmark/cell_scaling.csv")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rna_path = Path(args.rna_path)
    atac_path = Path(args.atac_path)
    rna_mm = np.load(rna_path, mmap_mode="r")
    total_cells, num_genes = rna_mm.shape
    atac_mm = np.load(atac_path, mmap_mode="r")
    num_peaks = atac_mm.shape[1]
    if atac_mm.shape[0] != total_cells:
        raise ValueError(f"RNA/ATAC cell mismatch: {total_cells} vs {atac_mm.shape[0]}")
    gene_names = load_feature_names(args.gene_names, num_genes, "gene")
    peak_names = load_feature_names(args.peak_names, num_peaks, "peak")
    candidate_set = CandidateSet(
        candidate_file=args.candidate_file,
        gene_names=gene_names,
        peak_names=peak_names,
    )
    rows: List[Dict[str, object]] = []
    cell_counts = parse_int_list(args.cell_counts)
    for idx, target in enumerate(cell_counts):
        actual = min(target, total_cells)
        if target > total_cells:
            print(
                f"[warn] Requested {target} cells but dataset only has {total_cells}; using {actual}.",
                file=sys.stderr,
            )
        subset_seed = args.seed + idx
        rna_tensor, atac_tensor, prep_sec = load_subset(
            rna_memmap=rna_mm,
            atac_memmap=atac_mm,
            cell_count=actual,
            seed=subset_seed,
        )
        actual = int(rna_tensor.size(0))
        dataset = TensorDataset(rna_tensor, atac_tensor)
        pin_memory = device.type == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        model = build_model(
            num_genes=num_genes,
            num_peaks=num_peaks,
            candidate_set=candidate_set,
            args=args,
            device=device,
        )
        train_stats = measure_training(
            model,
            loader,
            mask_ratio=args.mask_ratio,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_batches=args.max_train_batches,
            amp=args.amp,
        )
        # Rebuild a fresh model to avoid any optimizer state affecting inference timing.
        model_infer = build_model(
            num_genes=num_genes,
            num_peaks=num_peaks,
            candidate_set=candidate_set,
            args=args,
            device=device,
        )
        infer_stats = measure_inference(
            model_infer,
            loader,
            device=device,
            max_batches=args.max_infer_batches,
            amp=args.amp,
        )
        row: Dict[str, object] = {
            "cell_count_requested": target,
            "cell_count_actual": actual,
            "preprocess_seconds": prep_sec,
            "train_steps_total": train_stats["total_batches"],
            "train_steps_measured": train_stats["steps_measured"],
            "train_measured_seconds": train_stats["measured_seconds"],
            "train_per_step_seconds": train_stats["per_step_seconds"],
            "train_estimated_epoch_seconds": train_stats["estimated_epoch_seconds"],
            "inference_steps_total": infer_stats["total_batches"],
            "inference_steps_measured": infer_stats["steps_measured"],
            "inference_measured_seconds": infer_stats["measured_seconds"],
            "inference_per_step_seconds": infer_stats["per_step_seconds"],
            "inference_estimated_epoch_seconds": infer_stats["estimated_epoch_seconds"],
            "batch_size": args.batch_size,
            "mask_ratio": args.mask_ratio,
            "device": str(device),
            "amp": bool(args.amp and device.type == "cuda"),
            "num_genes": num_genes,
            "num_peaks": num_peaks,
            "candidate_max": candidate_set.max_candidates,
            "dataset_total_cells": total_cells,
        }
        rows.append(row)
        print(json.dumps(row))
        del model, model_infer, dataset, loader, rna_tensor, atac_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
