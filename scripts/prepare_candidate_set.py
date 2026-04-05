#!/usr/bin/env python3
"""
Convert the plusNshare gene-peak table into the JSON/TSV format
consumed by EpiAware's CandidateSet loader.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate peak JSON from gene_peak_10k.txt"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input TSV (e.g., datasets/plusNshare/gene_peak_10k.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON for CandidateSet",
    )
    parser.add_argument(
        "--gene_column",
        default="gene_id_1",
        help="Column containing gene IDs (default: gene_id_1)",
    )
    parser.add_argument(
        "--peak_column",
        default="peak_name_1",
        help="Column containing peak names (default: peak_name_1)",
    )
    parser.add_argument(
        "--peaks_txt",
        type=Path,
        default=None,
        help="Optional peaks.txt file; if provided, peaks not present there are skipped",
    )
    parser.add_argument(
        "--max_peaks_per_gene",
        type=int,
        default=None,
        help="Cap the number of peaks per gene (keep highest-frequency entries first)",
    )
    return parser.parse_args()


def load_peak_whitelist(path: Path | None) -> Set[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        peaks = {line.strip() for line in handle if line.strip()}
    return peaks


def build_candidate_map(
    input_path: Path,
    gene_column: str,
    peak_column: str,
    peak_whitelist: Set[str] | None,
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    with open(input_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = row.get(gene_column)
            peak = row.get(peak_column)
            if not gene or not peak:
                continue
            if peak_whitelist is not None and peak not in peak_whitelist:
                continue
            bucket = counts.setdefault(gene, {})
            bucket[peak] = bucket.get(peak, 0) + 1
    return counts


def dump_candidate_json(
    counts: Dict[str, Dict[str, int]],
    output_path: Path,
    max_peaks_per_gene: int | None,
) -> None:
    payload: Dict[str, List[Dict[str, object]]] = {"candidates": []}
    for gene, peaks in counts.items():
        ordered = sorted(peaks.items(), key=lambda kv: kv[1], reverse=True)
        if max_peaks_per_gene is not None:
            ordered = ordered[:max_peaks_per_gene]
        payload["candidates"].append(
            {
                "gene": gene,
                "peaks": [peak for peak, _ in ordered],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[done] Wrote {len(payload['candidates'])} gene entries to {output_path}")


def main() -> None:
    args = parse_args()
    peak_whitelist = load_peak_whitelist(args.peaks_txt)
    counts = build_candidate_map(
        args.input,
        args.gene_column,
        args.peak_column,
        peak_whitelist,
    )
    dump_candidate_json(counts, args.output, args.max_peaks_per_gene)


if __name__ == "__main__":
    main()
