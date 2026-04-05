"""Utilities for per-TF edge summaries and degree-matched random-network p-values."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import math
import numpy as np


def _unique_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_gold_map(gold_pairs: Sequence[Tuple[str, str]]) -> Dict[str, set]:
    gold_map: Dict[str, set] = defaultdict(set)
    for tf, tgt in gold_pairs:
        gold_map[tf].add(tgt)
    return gold_map


def summarize_edge_map(
    edge_map: Mapping[str, Sequence[str]],
    gold_map: Mapping[str, Sequence[str]],
    total_gold: int,
) -> Dict[str, float]:
    total_edges = 0
    total_tp = 0
    tf_precisions: List[float] = []
    tf_recalls: List[float] = []

    for tf, targets in edge_map.items():
        unique_targets = _unique_preserve(targets)
        if not unique_targets:
            continue
        total_edges += len(unique_targets)
        gold_targets = set(gold_map.get(tf, ()))
        tp = sum(1 for tgt in unique_targets if tgt in gold_targets)
        total_tp += tp
        tf_precisions.append(tp / len(unique_targets))
        tf_recalls.append(tp / len(gold_targets) if gold_targets else 0.0)

    precision = (total_tp / total_edges) if total_edges else 0.0
    recall = (total_tp / total_gold) if total_gold else 0.0
    macro_precision = (sum(tf_precisions) / len(tf_precisions)) if tf_precisions else 0.0
    macro_recall = (sum(tf_recalls) / len(tf_recalls)) if tf_recalls else 0.0

    return {
        "edges": int(total_edges),
        "true_positives": int(total_tp),
        "precision": float(precision),
        "recall": float(recall),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
    }


def degree_matched_pvalue(
    edge_map: Mapping[str, Sequence[str]],
    gold_map: Mapping[str, Sequence[str]],
    candidate_genes: Sequence[str],
    *,
    num_trials: int = 1000,
    seed: int = 13,
    max_edges: int | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute empirical p-value via degree-matched random TF-target assignments."""

    stats: Dict[str, float] = {
        "observed_overlap": 0.0,
        "random_hits_mean": math.nan,
        "random_hits_std": math.nan,
        "random_hits_min": math.nan,
        "random_hits_max": math.nan,
        "random_trials": float(max(num_trials, 0)),
        "edge_count": 0.0,
        "tf_count": 0.0,
    }

    if not edge_map:
        stats["skip_reason"] = "no_edges"
        return math.nan, stats

    gene_list = _unique_preserve(candidate_genes)
    gene_set = set(gene_list)
    for targets in gold_map.values():
        for tgt in targets:
            if tgt not in gene_set:
                gene_set.add(tgt)
                gene_list.append(tgt)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
    gene_count = len(gene_list)
    if gene_count == 0:
        stats["skip_reason"] = "empty_gene_universe"
        return math.nan, stats

    gold_idx_by_tf: Dict[str, set] = {}
    for tf, targets in gold_map.items():
        gold_idx_by_tf[tf] = {gene_to_idx[tgt] for tgt in targets if tgt in gene_to_idx}

    filtered_edge_map: Dict[str, List[int]] = {}
    observed = 0
    total_edges = 0

    for tf, targets in edge_map.items():
        unique_targets = _unique_preserve(targets)
        idx_targets = [gene_to_idx[tgt] for tgt in unique_targets if tgt in gene_to_idx]
        if not idx_targets:
            continue
        filtered_edge_map[tf] = idx_targets
        total_edges += len(idx_targets)
        gold_idx = gold_idx_by_tf.get(tf)
        if gold_idx:
            observed += sum(1 for idx in idx_targets if idx in gold_idx)

    stats["observed_overlap"] = float(observed)
    stats["edge_count"] = float(total_edges)
    stats["tf_count"] = float(len(filtered_edge_map))

    if total_edges == 0 or not filtered_edge_map:
        stats["skip_reason"] = "no_edges_after_filter"
        return math.nan, stats

    if max_edges is not None and total_edges > max_edges:
        stats["skip_reason"] = f"edge_count_exceeds_{max_edges}"
        return math.nan, stats

    if num_trials <= 0:
        stats["skip_reason"] = "zero_trials"
        return math.nan, stats

    rng = np.random.default_rng(int(seed))
    random_hits = np.zeros(num_trials, dtype=np.int32)
    gene_indices = np.arange(gene_count, dtype=np.int32)

    for trial in range(num_trials):
        total = 0
        for tf, idx_targets in filtered_edge_map.items():
            deg = len(idx_targets)
            if deg >= gene_count:
                choices = gene_indices
            else:
                choices = rng.choice(gene_count, size=deg, replace=False)
            gold_idx = gold_idx_by_tf.get(tf)
            if gold_idx:
                total += sum(1 for idx in choices if idx in gold_idx)
        random_hits[trial] = total

    stats["random_hits_mean"] = float(random_hits.mean())
    stats["random_hits_std"] = float(random_hits.std(ddof=0))
    stats["random_hits_min"] = float(random_hits.min())
    stats["random_hits_max"] = float(random_hits.max())

    greater_equal = int((random_hits >= observed).sum())
    p_value = (1 + greater_equal) / (num_trials + 1)
    return float(p_value), stats
