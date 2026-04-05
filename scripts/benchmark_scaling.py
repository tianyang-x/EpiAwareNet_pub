#!/usr/bin/env python3
"""Benchmark runtime + GPU memory as G/P scale.

This benchmark focuses on *model-side* complexity:
- Stage1 multi-omic backbone: gene sparse self-attn + gene-peak cross-attn
- RNA-only backbone: gene sparse self-attn only

Notes on candidate sets / priors:
- The implementation represents candidate sets as a dense (G, max_candidates) index tensor.
  Therefore, a true "full" candidate set of size P per gene is not feasible for large P.
- To study scalability, we benchmark with a bounded max_candidates per gene (e.g., 50, 200, 1000)
  and optionally apply Top-K routing within those candidates.

Outputs a CSV (or prints JSON lines) with runtime and peak GPU memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from epiaware.models import EpiAwareTransformer, GeneOnlyTransformer  # noqa: E402


@dataclass
class BenchResult:
    tag: str
    model: str
    G: int
    P: int
    batch: int
    embed_dim: int
    num_heads: int
    depth: int
    k_neighbors: int
    max_candidates: int
    topk_peaks: int
    cross_chunk_size: int
    self_chunk_size: int
    dtype: str
    device: str
    warmup: int
    steps: int
    avg_infer_ms: float
    infer_peak_alloc_mb: Optional[float]
    infer_peak_reserved_mb: Optional[float]
    avg_train_ms: Optional[float] = None
    train_peak_alloc_mb: Optional[float] = None
    train_peak_reserved_mb: Optional[float] = None


def parse_int_list(text: str) -> List[int]:
    items = []
    for part in text.replace(",", " ").split():
        if part.strip():
            items.append(int(part))
    return items


def make_candidate_tensors(
    G: int,
    P: int,
    max_candidates: int,
    mode: str,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return candidate_index (G, M) int64 and candidate_mask (G, M) bool."""
    M = max(1, int(max_candidates))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    if mode == "fixed":
        # Deterministic, same candidates for all genes: [0..M-1] clipped to P.
        base = torch.arange(min(P, M), dtype=torch.long)
        if base.numel() < M:
            pad = base.new_full((M - base.numel(),), int(base[-1]) if base.numel() else 0)
            base = torch.cat([base, pad], dim=0)
        index = base.unsqueeze(0).expand(G, -1).contiguous()
        mask = torch.ones((G, M), dtype=torch.bool)
        mask[:, min(P, M) :] = False
        return index.to(device), mask.to(device)

    if mode == "random":
        # Random candidates per gene.
        # Note: sampling without replacement per gene would be expensive; we allow repeats
        # (it doesn't change compute/memory much).
        index = torch.randint(low=0, high=max(P, 1), size=(G, M), generator=gen, dtype=torch.long)
        mask = torch.ones((G, M), dtype=torch.bool)
        return index.to(device), mask.to(device)

    raise ValueError(f"Unknown candidate mode: {mode} (expected fixed|random)")


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dummy_loss(output: Tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
    if isinstance(output, tuple):
        loss = None
        for tensor in output:
            tensor_loss = tensor.float().mean()
            loss = tensor_loss if loss is None else (loss + tensor_loss)
        assert loss is not None
        return loss
    return output.float().mean()


def measure_forward(
    *,
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int,
    steps: int,
    use_amp: bool,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Return avg_ms, peak_alloc_mb, peak_reserved_mb."""
    model.eval()

    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            torch.cuda.reset_peak_memory_stats()

    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
    )

    # Warmup
    with torch.inference_mode(), autocast_ctx:
        for _ in range(max(0, warmup)):
            _ = model(*inputs)
    _cuda_sync(device)

    # Timed
    start = time.perf_counter()
    with torch.inference_mode(), autocast_ctx:
        for _ in range(max(1, steps)):
            _ = model(*inputs)
    _cuda_sync(device)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / max(1, steps)

    peak_alloc_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        try:
            peak_alloc_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            peak_reserved_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
        except Exception:
            peak_alloc_mb = float(torch.cuda.max_memory_allocated()) / (1024.0**2)
            peak_reserved_mb = float(torch.cuda.max_memory_reserved()) / (1024.0**2)

    return avg_ms, peak_alloc_mb, peak_reserved_mb


def measure_train(
    *,
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int,
    steps: int,
    use_amp: bool,
    learning_rate: float,
) -> Tuple[float, Optional[float], Optional[float]]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
    )

    def run_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            output = model(*inputs)
            loss = _dummy_loss(output)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    with torch.enable_grad():
        for _ in range(max(0, warmup)):
            run_step()
        _cuda_sync(device)
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(max(1, steps)):
            run_step()
        _cuda_sync(device)
        end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / max(1, steps)
    peak_alloc_mb = None
    peak_reserved_mb = None
    if device.type == "cuda":
        try:
            peak_alloc_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            peak_reserved_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
        except Exception:
            peak_alloc_mb = float(torch.cuda.max_memory_allocated()) / (1024.0**2)
            peak_reserved_mb = float(torch.cuda.max_memory_reserved()) / (1024.0**2)

    return avg_ms, peak_alloc_mb, peak_reserved_mb


from contextlib import nullcontext


def build_multiomic_backbone(
    *,
    G: int,
    P: int,
    candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    embed_dim: int,
    num_heads: int,
    depth: int,
    k_neighbors: int,
    topk_peaks: int,
    cross_chunk_size: int,
    self_chunk_size: int,
    use_faiss_knn: bool,
    device: torch.device,
) -> EpiAwareTransformer:
    model = EpiAwareTransformer(
        num_genes=G,
        num_peaks=P,
        candidate_index=candidate_index,
        candidate_mask=candidate_mask,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        k_neighbors=k_neighbors,
        topk_peaks=topk_peaks,
        ffn_hidden=max(4 * embed_dim, 512),
        dropout=0.0,
        use_faiss_knn=use_faiss_knn,
        cross_chunk_size=cross_chunk_size,
        self_chunk_size=self_chunk_size,
    ).to(device)
    return model


def build_geneonly_backbone(
    *,
    G: int,
    embed_dim: int,
    num_heads: int,
    depth: int,
    k_neighbors: int,
    self_chunk_size: int,
    use_faiss_knn: bool,
    device: torch.device,
) -> GeneOnlyTransformer:
    model = GeneOnlyTransformer(
        num_genes=G,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        k_neighbors=k_neighbors,
        ffn_hidden=max(4 * embed_dim, 512),
        dropout=0.0,
        use_faiss_knn=use_faiss_knn,
        self_chunk_size=self_chunk_size,
    ).to(device)
    return model


def write_csv(path: Path, rows: List[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark scalability over G (genes), P (peaks), and top-K routing")
    p.add_argument("--tag", type=str, default="", help="Optional label recorded with every row")
    p.add_argument("--G", type=str, default="2000 10000", help="Gene counts, e.g. '2000 5000 10000'")
    p.add_argument("--P", type=str, default="50000 300000", help="Peak counts, e.g. '50000 100000 300000'")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--k_neighbors", type=int, default=16)

    p.add_argument("--max_candidates", type=int, default=50, help="Candidates per gene for cross-attn")
    p.add_argument(
        "--topk_peaks",
        type=str,
        default="8 0",
        help="Stage1 cross-attn routing K values. Use 0 for no routing (attend all candidates).",
    )
    p.add_argument(
        "--candidate_mode",
        choices=("fixed", "random"),
        default="random",
        help="How to build candidate peak lists.",
    )
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--cross_chunk_size", type=int, default=1024)
    p.add_argument("--self_chunk_size", type=int, default=1024)
    p.add_argument("--use_faiss_knn", action="store_true", default=False)

    p.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu")
    p.add_argument("--amp", action="store_true", default=True, help="Use autocast fp16 on CUDA")
    p.add_argument("--no-amp", dest="amp", action="store_false")

    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)

    p.add_argument("--train_warmup", type=int, default=0, help="Backward warmup steps (0 to skip training benchmark)")
    p.add_argument("--train_steps", type=int, default=0, help="Timed backward steps (0 to skip training benchmark)")
    p.add_argument("--train_lr", type=float, default=1e-4, help="Learning rate for synthetic optimizer")

    p.add_argument("--out_csv", type=str, default="output/scaling_benchmark/results.csv")
    p.add_argument("--also_geneonly", action="store_true", default=True)
    p.add_argument("--no-geneonly", dest="also_geneonly", action="store_false")

    args = p.parse_args()

    G_list = parse_int_list(args.G)
    P_list = parse_int_list(args.P)
    topk_list = parse_int_list(args.topk_peaks)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    rows: List[BenchResult] = []

    for G in G_list:
        # Pre-build RNA-only backbone once per G
        geneonly_model = None
        rna = torch.randn((args.batch, G), device=device, dtype=torch.float32)
        if args.also_geneonly:
            geneonly_model = build_geneonly_backbone(
                G=G,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                depth=args.depth,
                k_neighbors=args.k_neighbors,
                self_chunk_size=args.self_chunk_size,
                use_faiss_knn=args.use_faiss_knn,
                device=device,
            )
            avg_ms, peak_a, peak_r = measure_forward(
                model=geneonly_model,
                inputs=(rna,),
                device=device,
                warmup=args.warmup,
                steps=args.steps,
                use_amp=args.amp,
            )
            train_ms = None
            train_peak_a = None
            train_peak_r = None
            if args.train_steps > 0:
                train_ms, train_peak_a, train_peak_r = measure_train(
                    model=geneonly_model,
                    inputs=(rna,),
                    device=device,
                    warmup=args.train_warmup,
                    steps=args.train_steps,
                    use_amp=args.amp,
                    learning_rate=args.train_lr,
                )
            rows.append(
                BenchResult(
                    tag=args.tag,
                    model="gene_only",
                    G=G,
                    P=0,
                    batch=args.batch,
                    embed_dim=args.embed_dim,
                    num_heads=args.num_heads,
                    depth=args.depth,
                    k_neighbors=args.k_neighbors,
                    max_candidates=0,
                    topk_peaks=0,
                    cross_chunk_size=0,
                    self_chunk_size=args.self_chunk_size,
                    dtype="amp_fp16" if (args.amp and device.type == "cuda") else "fp32",
                    device=str(device),
                    warmup=args.warmup,
                    steps=args.steps,
                    avg_infer_ms=avg_ms,
                    infer_peak_alloc_mb=peak_a,
                    infer_peak_reserved_mb=peak_r,
                    avg_train_ms=train_ms,
                    train_peak_alloc_mb=train_peak_a,
                    train_peak_reserved_mb=train_peak_r,
                )
            )
            print(json.dumps(asdict(rows[-1]), ensure_ascii=False))

        for P in P_list:
            atac = torch.randn((args.batch, P), device=device, dtype=torch.float32)
            candidate_index, candidate_mask = make_candidate_tensors(
                G=G,
                P=P,
                max_candidates=args.max_candidates,
                mode=args.candidate_mode,
                seed=args.seed,
                device=device,
            )

            for topk in topk_list:
                model = build_multiomic_backbone(
                    G=G,
                    P=P,
                    candidate_index=candidate_index,
                    candidate_mask=candidate_mask,
                    embed_dim=args.embed_dim,
                    num_heads=args.num_heads,
                    depth=args.depth,
                    k_neighbors=args.k_neighbors,
                    topk_peaks=int(topk),
                    cross_chunk_size=args.cross_chunk_size,
                    self_chunk_size=args.self_chunk_size,
                    use_faiss_knn=args.use_faiss_knn,
                    device=device,
                )

                avg_ms, peak_a, peak_r = measure_forward(
                    model=model,
                    inputs=(rna, atac),
                    device=device,
                    warmup=args.warmup,
                    steps=args.steps,
                    use_amp=args.amp,
                )
                train_ms = None
                train_peak_a = None
                train_peak_r = None
                if args.train_steps > 0:
                    train_ms, train_peak_a, train_peak_r = measure_train(
                        model=model,
                        inputs=(rna, atac),
                        device=device,
                        warmup=args.train_warmup,
                        steps=args.train_steps,
                        use_amp=args.amp,
                        learning_rate=args.train_lr,
                    )

                rows.append(
                    BenchResult(
                        tag=args.tag,
                        model="multiomic_backbone",
                        G=G,
                        P=P,
                        batch=args.batch,
                        embed_dim=args.embed_dim,
                        num_heads=args.num_heads,
                        depth=args.depth,
                        k_neighbors=args.k_neighbors,
                        max_candidates=args.max_candidates,
                        topk_peaks=int(topk),
                        cross_chunk_size=args.cross_chunk_size,
                        self_chunk_size=args.self_chunk_size,
                        dtype="amp_fp16" if (args.amp and device.type == "cuda") else "fp32",
                        device=str(device),
                        warmup=args.warmup,
                        steps=args.steps,
                        avg_infer_ms=avg_ms,
                        infer_peak_alloc_mb=peak_a,
                        infer_peak_reserved_mb=peak_r,
                        avg_train_ms=train_ms,
                        train_peak_alloc_mb=train_peak_a,
                        train_peak_reserved_mb=train_peak_r,
                    )
                )
                print(json.dumps(asdict(rows[-1]), ensure_ascii=False))

    out_path = Path(args.out_csv)
    if rows:
        write_csv(out_path, rows)
        print(f"[done] Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
