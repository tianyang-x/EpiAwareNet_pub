#!/usr/bin/env python3
"""Generate a random CandidateSet JSON for weak-prior ablations.

Outputs the JSON format consumed by epiaware.data.CandidateSet:
  {"candidates": [{"gene": "...", "peaks": ["...", ...]}, ...]}

This keeps the file size manageable by sampling up to K peaks per gene.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random candidate peak JSON")
    parser.add_argument("--genes_txt", type=Path, required=True, help="Path to genes.txt")
    parser.add_argument("--peaks_txt", type=Path, required=True, help="Path to peaks.txt")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--max_peaks_per_gene",
        type=int,
        default=50,
        help="Number of random peaks per gene (<=0 uses all peaks)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    genes = load_lines(args.genes_txt)
    peaks = load_lines(args.peaks_txt)
    if not genes:
        raise ValueError(f"No genes found in {args.genes_txt}")
    if not peaks:
        raise ValueError(f"No peaks found in {args.peaks_txt}")

    rng = random.Random(int(args.seed))
    k = int(args.max_peaks_per_gene)

    payload = {"candidates": []}
    for gene in genes:
        if k <= 0 or k >= len(peaks):
            chosen = list(peaks)
        else:
            chosen = rng.sample(peaks, k)
        payload["candidates"].append({"gene": gene, "peaks": chosen})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[done] Wrote random candidate set for {len(genes)} genes to {args.output}")


if __name__ == "__main__":
    main()
