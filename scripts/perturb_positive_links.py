#!/usr/bin/env python3
"""
Create corrupted versions of bulk prior link files by randomly dropping links
and/or flipping a fraction into false positives.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from epiaware.data import load_positive_links, load_tf_list


def read_plain_list(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            items.append(tokens[0])
    return items


def ensure_prefix(name: str, prefix: str) -> str:
    if not prefix:
        return name
    if name.startswith(prefix):
        return name
    return f"{prefix}{name}"


def unique_preserve_order(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def sample_indices(rng: random.Random, population_size: int, k: int) -> List[int]:
    if k <= 0:
        return []
    k = min(k, population_size)
    return rng.sample(range(population_size), k)


def write_links(path: str, pairs: Iterable[Tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for tf, tgt in pairs:
            handle.write(f"{tf}\t{tgt}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrupt bulk prior TF-target pairs.")
    parser.add_argument("--positive_links", required=True, help="Original bulk priors file")
    parser.add_argument("--genes_txt", required=True, help="Gene name list for the dataset")
    parser.add_argument(
        "--tf_list",
        required=True,
        help="TF list used during training (before prefixing)",
    )
    parser.add_argument("--output", required=True, help="Path to write corrupted priors")
    parser.add_argument(
        "--metadata_out",
        default=None,
        help="Optional JSON file summarizing the corruption statistics",
    )
    parser.add_argument(
        "--drop_frac",
        type=float,
        default=0.0,
        help="Fraction (0-1) of links to remove entirely",
    )
    parser.add_argument(
        "--flip_frac",
        type=float,
        default=0.0,
        help="Fraction (0-1) of links to convert into false positives",
    )
    parser.add_argument(
        "--tf_prefix",
        type=str,
        default="gene:",
        help="Prefix expected by training scripts (default: gene:)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.drop_frac <= 1.0):
        raise ValueError("--drop_frac must be within [0, 1]")
    if not (0.0 <= args.flip_frac <= 1.0):
        raise ValueError("--flip_frac must be within [0, 1]")

    rng = random.Random(args.seed)

    pos_links = load_positive_links(
        args.positive_links,
        tf_prefix=args.tf_prefix,
        target_prefix=args.tf_prefix,
    )
    pos_links = unique_preserve_order(pos_links)
    original_count = len(pos_links)
    if original_count == 0:
        raise ValueError("Positive link file is empty.")

    gene_names = read_plain_list(args.genes_txt)
    tf_names = load_tf_list(args.tf_list, gene_names, prefix=args.tf_prefix)
    if not tf_names:
        raise ValueError("TF list is empty after prefixing.")

    # Step 1: drop links entirely
    drop_count = int(round(original_count * args.drop_frac))
    drop_indices = set(sample_indices(rng, original_count, drop_count))
    retained: List[Tuple[str, str]] = [
        pair for idx, pair in enumerate(pos_links) if idx not in drop_indices
    ]

    # Step 2: flip a subset into random (false) positives
    flip_count = int(round(original_count * args.flip_frac))
    flip_count = min(flip_count, len(retained))
    flip_indices = set(sample_indices(rng, len(retained), flip_count))
    retained_after_flip: List[Tuple[str, str]] = []
    flipped_pairs: List[Tuple[str, str]] = []
    for idx, pair in enumerate(retained):
        if idx in flip_indices:
            flipped_pairs.append(pair)
        else:
            retained_after_flip.append(pair)

    positive_set = set(retained_after_flip)
    generated_false_pairs: List[Tuple[str, str]] = []
    max_attempts = flip_count * 50 if flip_count > 0 else 0
    attempts = 0
    while len(generated_false_pairs) < flip_count and attempts < max_attempts:
        attempts += 1
        tf_name = ensure_prefix(rng.choice(tf_names), args.tf_prefix)
        tgt_name = rng.choice(gene_names)
        pair = (tf_name, tgt_name)
        if pair in positive_set:
            continue
        positive_set.add(pair)
        generated_false_pairs.append(pair)
    if len(generated_false_pairs) < flip_count:
        raise RuntimeError(
            f"Unable to generate {flip_count} unique false positives "
            f"(only produced {len(generated_false_pairs)})."
        )

    final_pairs = retained_after_flip + generated_false_pairs
    rng.shuffle(final_pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_links(str(out_path), final_pairs)

    if args.metadata_out:
        metadata = {
            "input": args.positive_links,
            "output": str(out_path),
            "seed": args.seed,
            "drop_frac": args.drop_frac,
            "flip_frac": args.flip_frac,
            "original_count": original_count,
            "dropped": drop_count,
            "flipped": flip_count,
            "remaining_after_drop": len(retained),
            "final_count": len(final_pairs),
        }
        Path(args.metadata_out).write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
