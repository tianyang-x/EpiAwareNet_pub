#!/usr/bin/env python3
"""
Convert 10x-style matrices (matrix.mtx[.gz], features.tsv[.gz], barcodes.tsv[.gz])
to the dense .npy + plain-text metadata files expected by the EpiAware pipeline.

The script streams sparse blocks into an on-disk memmap to avoid holding the
entire dense matrix in memory at once.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from scipy.io import mmread
from scipy.sparse import csr_matrix

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable


def _find_file(directory: Path, candidates: List[str]) -> Path:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the files {candidates} found inside {directory}")


def _read_tsv_column(path: Path) -> List[List[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: List[List[str]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if parts:
                rows.append(parts)
    return rows


def _write_list(path: Path, values: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(values))


def _save_dense_matrix(
    sparse: csr_matrix,
    output_path: Path,
    chunk_size: int = 512,
) -> Tuple[int, int]:
    n_genes, n_cells = sparse.shape
    dense = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_cells, n_genes),
    )
    for start in tqdm(
        range(0, n_cells, chunk_size),
        desc=f"Writing {output_path.name}",
        unit="cells",
    ):
        end = min(n_cells, start + chunk_size)
        block = sparse[:, start:end].transpose()  # (chunk, genes)
        dense[start:end] = block.toarray().astype(np.float32, copy=False)
    del dense  # flush to disk
    return n_cells, n_genes


def convert_modality(
    input_dir: Path,
    matrix_out: Path,
    features_out: Path,
    barcodes_out: Path,
    feature_type: str,
    chunk_size: int,
) -> None:
    matrix_path = _find_file(input_dir, ["matrix.mtx.gz", "matrix.mtx"])
    features_path = _find_file(input_dir, ["features.tsv.gz", "features.tsv"])
    barcodes_path = _find_file(input_dir, ["barcodes.tsv.gz", "barcodes.tsv"])

    print(f"[info] Loading sparse matrix from {matrix_path}")
    sparse = mmread(str(matrix_path)).tocsr().astype(np.float32)
    cells, genes = _save_dense_matrix(sparse, matrix_out, chunk_size=chunk_size)
    print(f"[info] Saved dense matrix to {matrix_out} (cells={cells}, genes={genes})")

    feature_rows = _read_tsv_column(features_path)
    if feature_rows and len(feature_rows[0]) >= 2:
        feature_names = [row[1] if len(row) > 1 and row[1] else row[0] for row in feature_rows]
    else:
        feature_names = [row[0] for row in feature_rows]
        feature_rows = [[name, name, feature_type] for name in feature_names]
        print(
            f"[warn] Feature file {features_path} had <3 columns; "
            "generated standard 10x columns with repeated names."
        )
        tmp_path = features_path.with_name(features_path.name + ".tmp")
        with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
            for gene_id, gene_symbol, ftype in feature_rows:
                handle.write(f"{gene_id}\t{gene_symbol}\t{ftype}\n")
        os.replace(tmp_path, features_path)
    _write_list(features_out, feature_names)
    print(f"[info] Wrote feature names to {features_out}")

    barcode_rows = _read_tsv_column(barcodes_path)
    barcodes = [row[0] for row in barcode_rows]
    _write_list(barcodes_out, barcodes)
    print(f"[info] Wrote {len(barcodes)} barcodes to {barcodes_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 10x-formatted RNA/ATAC datasets to dense .npy arrays."
    )
    parser.add_argument("--rna_dir", type=Path, required=True, help="Directory containing RNA 10x files")
    parser.add_argument("--atac_dir", type=Path, required=True, help="Directory containing ATAC 10x files")
    parser.add_argument(
        "--rna_out",
        type=Path,
        default=None,
        help="Output path for RNA .npy (default: <rna_dir>/rna.npy)",
    )
    parser.add_argument(
        "--atac_out",
        type=Path,
        default=None,
        help="Output path for ATAC .npy (default: <atac_dir>/atac.npy)",
    )
    parser.add_argument(
        "--genes_out",
        type=Path,
        default=None,
        help="File to store gene names (default: <rna_dir>/genes.txt)",
    )
    parser.add_argument(
        "--peaks_out",
        type=Path,
        default=None,
        help="File to store peak names (default: <atac_dir>/peaks.txt)",
    )
    parser.add_argument(
        "--rna_barcodes_out",
        type=Path,
        default=None,
        help="File to store RNA barcodes (default: <rna_dir>/cells.txt)",
    )
    parser.add_argument(
        "--atac_barcodes_out",
        type=Path,
        default=None,
        help="File to store ATAC barcodes (default: <atac_dir>/cells.txt)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Number of cells to densify per chunk when writing .npy ",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rna_out = args.rna_out or (args.rna_dir / "rna.npy")
    atac_out = args.atac_out or (args.atac_dir / "atac.npy")
    genes_out = args.genes_out or (args.rna_dir / "genes.txt")
    peaks_out = args.peaks_out or (args.atac_dir / "peaks.txt")
    rna_barcodes_out = args.rna_barcodes_out or (args.rna_dir / "cells.txt")
    atac_barcodes_out = args.atac_barcodes_out or (args.atac_dir / "cells.txt")

    convert_modality(
        args.rna_dir,
        rna_out,
        genes_out,
        rna_barcodes_out,
        feature_type="Gene Expression",
        chunk_size=args.chunk_size,
    )
    convert_modality(
        args.atac_dir,
        atac_out,
        peaks_out,
        atac_barcodes_out,
        feature_type="Peaks",
        chunk_size=args.chunk_size,
    )
    print("[done] Conversion finished. New files ready for the EpiAware pipeline.")


if __name__ == "__main__":
    main()
