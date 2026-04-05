#!/usr/bin/env python3
"""Preprocess a 10x *multiome* Matrix Market directory for EpiAwareNet.

This script handles the common 10x layout where Gene Expression and Peaks share the
same `filtered_feature_bc_matrix/` directory:
  - matrix.mtx[.gz]
  - features.tsv[.gz]
  - barcodes.tsv[.gz]

It exports the same file layout used by the training/ablation scripts:
  - RNA dir:  rna.npy, genes.txt, cells.txt, genes_metadata.tsv, candidate_peaks.json
  - ATAC dir: atac.npy, peaks.txt, cells.txt

Optionally, it can build `candidate_peaks.json` from a BED-like gene↔peak table
(e.g. `datasets/mouse_brain/gene_peaks_5kb.txt`). When such a table is provided,
ATAC peaks are restricted to the union of selected candidate peaks to keep the
output dense ATAC matrix reasonably sized.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from scipy.io import mmread
from scipy.sparse import csr_matrix

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):
        return iterable


def _find_file(directory: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {list(candidates)} found in {directory}")


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _write_list(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(values))


def _read_features(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            rows.append(parts)
    return rows


def _read_barcodes(path: Path) -> List[str]:
    values: List[str] = []
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values.append(line.split()[0])
    return values


def _save_dense_matrix(
    sparse: csr_matrix,
    output_path: Path,
    chunk_size: int,
) -> Tuple[int, int]:
    # sparse is (features, cells)
    num_features, num_cells = sparse.shape
    dense = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_cells, num_features),
    )
    for start in tqdm(range(0, num_cells, chunk_size), desc=f"Writing {output_path.name}", unit="cells"):
        end = min(num_cells, start + chunk_size)
        block = sparse[:, start:end].transpose()  # (chunk, features)
        dense[start:end] = block.toarray().astype(np.float32, copy=False)
    del dense
    return num_cells, num_features


def _build_gene_names(
    symbols: Sequence[str],
    ensembl_ids: Sequence[str],
) -> Tuple[List[str], Mapping[str, List[str]]]:
    base_names: List[str] = []
    for symbol, ens in zip(symbols, ensembl_ids):
        symbol = symbol.strip()
        ens = ens.strip()
        if symbol:
            base_names.append(symbol)
        elif ens:
            base_names.append(ens)
        else:
            base_names.append("UNKNOWN")

    counts = Counter(base_names)
    exported: List[str] = []
    lookup: DefaultDict[str, List[str]] = defaultdict(list)

    for base, symbol, ens in zip(base_names, symbols, ensembl_ids):
        base = base.strip()
        symbol = symbol.strip()
        ens = ens.strip()
        tag = base
        if counts[base] > 1 and ens:
            tag = f"{base}|{ens}"
        exported_name = f"gene:{tag}"
        exported.append(exported_name)
        if symbol:
            lookup[symbol].append(exported_name)
        if ens:
            lookup[ens].append(exported_name)
    return exported, lookup


def _write_gene_metadata(
    path: Path,
    exported_names: Sequence[str],
    symbols: Sequence[str],
    ensembl_ids: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene_name", "symbol", "ensembl_id"])
        for name, sym, ens in zip(exported_names, symbols, ensembl_ids):
            writer.writerow([name, sym.strip(), ens.strip()])


def _peak_name(chrom: str, start: str, end: str) -> str:
    return f"{chrom}:{start}-{end}"


def _build_candidates_from_gene_peak_table(
    gene_peak_path: Path,
    gene_lookup: Mapping[str, List[str]],
    valid_peaks: Sequence[str],
    max_peaks_per_gene: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    peak_set = set(valid_peaks)
    counts: DefaultDict[str, Dict[str, int]] = defaultdict(dict)

    with open(gene_peak_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 7:
                continue
            gene_ens = parts[3]
            peak = _peak_name(parts[4], parts[5], parts[6])
            if peak not in peak_set:
                continue
            exported_genes = gene_lookup.get(gene_ens)
            if not exported_genes:
                continue
            for exported in exported_genes:
                bucket = counts.setdefault(exported, {})
                bucket[peak] = bucket.get(peak, 0) + 1

    candidates: List[Dict[str, object]] = []
    used_peaks: List[str] = []
    used_set = set()

    for gene_name, peak_counts in counts.items():
        ordered = sorted(peak_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        limited = [peak for peak, _ in ordered[:max_peaks_per_gene]]
        if not limited:
            continue
        candidates.append({"gene": gene_name, "peaks": limited})
        for peak in limited:
            if peak not in used_set:
                used_set.add(peak)
                used_peaks.append(peak)

    if not candidates:
        raise RuntimeError(
            "No candidate peaks could be built. Check that gene ids in the gene-peak table "
            "match the ENSEMBL ids in the 10x features file, and that the peak coordinates use the same build."
        )

    return candidates, used_peaks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a 10x multiome mtx directory into EpiAwareNet inputs.")
    parser.add_argument(
        "--multiome_dir",
        type=Path,
        required=True,
        help="Path to 10x directory containing matrix.mtx(.gz), features.tsv(.gz), barcodes.tsv(.gz)",
    )
    parser.add_argument("--rna_dir", type=Path, required=True, help="Output directory for RNA files")
    parser.add_argument("--atac_dir", type=Path, required=True, help="Output directory for ATAC files")
    parser.add_argument(
        "--gene_peaks",
        type=Path,
        default=None,
        help="Optional BED-like gene↔peak table (e.g. gene_peaks_5kb.txt) to build candidate_peaks.json",
    )
    parser.add_argument("--max_peaks_per_gene", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.rna_dir.mkdir(parents=True, exist_ok=True)
    args.atac_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = _find_file(args.multiome_dir, ["matrix.mtx.gz", "matrix.mtx"])
    features_path = _find_file(args.multiome_dir, ["features.tsv.gz", "features.tsv"])
    barcodes_path = _find_file(args.multiome_dir, ["barcodes.tsv.gz", "barcodes.tsv"])

    print(f"[info] Reading features: {features_path}")
    feature_rows = _read_features(features_path)
    if not feature_rows:
        raise RuntimeError(f"No features found in {features_path}")

    # 10x multiome `features.tsv.gz` typically has >= 3 columns; in this repo it has 6.
    # We only require: id, name, feature_type
    for i, row in enumerate(feature_rows[:5]):
        if len(row) < 3:
            raise RuntimeError(
                f"Expected at least 3 columns in features file, but row {i} has {len(row)} columns: {row!r}"
            )

    feature_ids = np.array([row[0] for row in feature_rows], dtype=object)
    feature_names = np.array([row[1] for row in feature_rows], dtype=object)
    feature_types = np.array([row[2] for row in feature_rows], dtype=object)

    gene_mask = feature_types == "Gene Expression"
    peak_mask = feature_types == "Peaks"
    if gene_mask.sum() == 0 or peak_mask.sum() == 0:
        raise RuntimeError(
            "Expected both Gene Expression and Peaks modalities in the features file. "
            f"Found Gene Expression={int(gene_mask.sum())}, Peaks={int(peak_mask.sum())}."
        )

    barcodes = _read_barcodes(barcodes_path)
    print(f"[info] Loaded {len(barcodes)} barcodes")

    print(f"[info] Loading sparse matrix: {matrix_path}")
    sparse_all = mmread(str(matrix_path)).tocsr().astype(np.float32)
    if sparse_all.shape[0] != len(feature_rows):
        raise RuntimeError(
            f"Matrix rows ({sparse_all.shape[0]}) do not match features rows ({len(feature_rows)})"
        )
    if sparse_all.shape[1] != len(barcodes):
        raise RuntimeError(
            f"Matrix cols ({sparse_all.shape[1]}) do not match barcodes rows ({len(barcodes)})"
        )

    gene_indices = np.flatnonzero(gene_mask)
    peak_indices_all = np.flatnonzero(peak_mask)

    gene_symbols = feature_names[gene_indices].astype(str)
    gene_ensembl = feature_ids[gene_indices].astype(str)
    exported_gene_names, gene_lookup = _build_gene_names(gene_symbols, gene_ensembl)

    peak_names_all = feature_names[peak_indices_all].astype(str).tolist()

    # Build candidates first (if provided), so we can restrict peaks to the used subset.
    candidates = None
    used_peak_names = None
    peak_indices = peak_indices_all
    peak_names = peak_names_all

    if args.gene_peaks is not None:
        print(f"[info] Building candidate peaks from {args.gene_peaks}")
        candidates, used_peak_names = _build_candidates_from_gene_peak_table(
            args.gene_peaks,
            gene_lookup,
            peak_names_all,
            max_peaks_per_gene=args.max_peaks_per_gene,
        )
        used_peak_set = set(used_peak_names)
        peak_name_to_old_idx = {name: idx for idx, name in zip(peak_indices_all, peak_names_all)}
        peak_indices = np.array(
            sorted([peak_name_to_old_idx[name] for name in used_peak_set]), dtype=np.int64
        )
        # Preserve old ordering
        peak_names = [feature_names[i] for i in peak_indices]
        peak_names = [str(x) for x in peak_names]
        print(
            f"[info] Restricting ATAC peaks to candidate union: {len(peak_indices_all)} -> {len(peak_indices)}"
        )

    print("[info] Slicing matrices …")
    gene_matrix = sparse_all[gene_indices, :]
    peak_matrix = sparse_all[peak_indices, :]

    print("[info] Writing dense RNA matrix …")
    rna_path = args.rna_dir / "rna.npy"
    num_cells, num_genes = _save_dense_matrix(gene_matrix, rna_path, chunk_size=args.chunk_size)
    print(f"[info]   Saved RNA matrix: {rna_path} (cells={num_cells}, genes={num_genes})")

    print("[info] Writing dense ATAC matrix …")
    atac_path = args.atac_dir / "atac.npy"
    atac_cells, num_peaks = _save_dense_matrix(peak_matrix, atac_path, chunk_size=args.chunk_size)
    if atac_cells != num_cells:
        raise RuntimeError(f"RNA and ATAC cell counts differ (RNA={num_cells}, ATAC={atac_cells}).")
    print(f"[info]   Saved ATAC matrix: {atac_path} (cells={atac_cells}, peaks={num_peaks})")

    print("[info] Writing vocabularies …")
    _write_list(args.rna_dir / "genes.txt", exported_gene_names)
    _write_list(args.rna_dir / "cells.txt", barcodes)
    _write_list(args.atac_dir / "peaks.txt", peak_names)
    _write_list(args.atac_dir / "cells.txt", barcodes)

    _write_gene_metadata(args.rna_dir / "genes_metadata.tsv", exported_gene_names, gene_symbols, gene_ensembl)

    if candidates is not None:
        candidate_path = args.rna_dir / "candidate_peaks.json"
        with open(candidate_path, "w", encoding="utf-8") as handle:
            json.dump({"candidates": candidates}, handle, indent=2)
        print(f"[info] Wrote candidate set: {candidate_path} (genes={len(candidates)}, peaks_union={len(set(peak_names))})")

    print("[done] Preprocessing complete.")


if __name__ == "__main__":
    main()
