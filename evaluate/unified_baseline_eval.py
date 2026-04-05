#!/usr/bin/env python
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

DEFAULT_K = (100, 500, 1000, 5000, 10000)

# -----------------------------
# IO helpers
# -----------------------------
def load_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def _strip_prefix(name: str, prefix: str = "gene:") -> str:
    return name[len(prefix):] if prefix and name.startswith(prefix) else name

def load_genes(path: Path, prefix: str = "gene:") -> List[str]:
    return [_strip_prefix(x, prefix) for x in load_list(path)]

def load_tf_list(path: Path, fallback: Iterable[str] | None = None, prefix: str = "gene:") -> List[str]:
    if path.exists():
        return [_strip_prefix(x, prefix) for x in load_list(path)]
    return list(fallback or [])

def _normalise_entry(entry: Dict[str, str], tf_prefix: str, target_prefix: str) -> Optional[Tuple[str, str, float]]:
    lowered = {str(k).lower(): v for k, v in entry.items()}
    tf = lowered.get("tf") or lowered.get("regulator") or lowered.get("gene_i") or lowered.get("source") or lowered.get("from")
    tgt = lowered.get("target") or lowered.get("gene_j") or lowered.get("gene") or lowered.get("sink") or lowered.get("to")
    score = lowered.get("score") or lowered.get("probability") or lowered.get("logit") or lowered.get("value") or lowered.get("importance") or lowered.get("weight")
    if tf is None or tgt is None or score is None:
        return None
    try:
        return _strip_prefix(str(tf), tf_prefix), _strip_prefix(str(tgt), target_prefix), float(score)
    except ValueError:
        return None

def load_predictions(path: Path, tf_filter: Optional[set] = None, tf_prefix: str = "gene:", target_prefix: str = "gene:") -> List[Tuple[str, str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    entries: List[Tuple[str, str, float]] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        iterable = raw.get("predictions") or raw.get("predicted_directed_links") or raw
        if isinstance(iterable, dict):
            iterable = iterable.values()
        for item in iterable:
            if not isinstance(item, dict):
                continue
            norm = _normalise_entry(item, tf_prefix, target_prefix)
            if norm:
                if tf_filter and norm[0] not in tf_filter:
                    continue
                entries.append(norm)
        return entries
    import csv
    delimiter = "," if path.suffix.lower().endswith("csv") else "\t"
    with open(path, "r", encoding="utf-8") as handle:
        sniff = handle.read(4096)
        handle.seek(0)
        if "\t" in sniff and path.suffix.lower().endswith("csv"):
            delimiter = "\t"
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            norm = _normalise_entry(row, tf_prefix, target_prefix)
            if norm:
                if tf_filter and norm[0] not in tf_filter:
                    continue
                entries.append(norm)
    return entries

def load_gold(path: Path, tf_prefix: str = "gene:", target_prefix: str = "gene:") -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    positives: List[Tuple[str, str]] = []
    import csv
    delimiter = "," if path.suffix.lower().endswith("csv") else "\t"
    with open(path, "r", encoding="utf-8") as handle:
        sniff = handle.read(4096)
        handle.seek(0)
        if "\t" in sniff and path.suffix.lower().endswith("csv"):
            delimiter = "\t"
        reader = csv.DictReader(handle, delimiter=delimiter)
        tf_col = None
        tgt_col = None
        for column in reader.fieldnames or []:
            c = column.lower()
            if c in {"tf", "regulator", "gene_i", "source"}:
                tf_col = column
            if c in {"target", "gene_j", "gene", "sink"}:
                tgt_col = column
        if tf_col is None or tgt_col is None:
            tf_col, tgt_col = reader.fieldnames[:2]
        for row in reader:
            tf = row.get(tf_col)
            tgt = row.get(tgt_col)
            if tf is None or tgt is None:
                continue
            positives.append((_strip_prefix(str(tf), tf_prefix), _strip_prefix(str(tgt), target_prefix)))
    return positives

def build_gold_map(gold_pairs: Sequence[Tuple[str, str]]) -> Dict[str, set]:
    gmap: Dict[str, set] = defaultdict(set)
    for tf, tgt in gold_pairs:
        gmap[tf].add(tgt)
    return gmap

# -----------------------------
# Metrics
# -----------------------------

def summarize_edge_map(edge_map: Dict[str, List[str]], gold_map: Dict[str, set], total_pos: int) -> Dict[str, float]:
    tp = 0
    edges = 0
    tf_prec = []
    tf_rec = []
    for tf, tgts in edge_map.items():
        tgts_set = set(tgts)
        edges += len(tgts_set)
        gold_tgts = gold_map.get(tf, set())
        hits = len(tgts_set & gold_tgts)
        tp += hits
        tf_prec.append(hits / len(tgts_set) if tgts_set else 0.0)
        tf_rec.append(hits / len(gold_tgts) if gold_tgts else 0.0)
    macro_prec = float(np.mean(tf_prec)) if tf_prec else float("nan")
    macro_rec = float(np.mean(tf_rec)) if tf_rec else float("nan")
    precision = tp / edges if edges else 0.0
    recall = tp / total_pos if total_pos else 0.0
    return {
        "edges": edges,
        "true_positives": tp,
        "precision": precision,
        "recall": recall,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }


def degree_matched_pvalue(edge_map: Dict[str, List[str]], gold_map: Dict[str, set], genes: List[str], num_trials: int, seed: int, max_edges: Optional[int]) -> Tuple[float, Dict[str, float]]:
    edges = sum(len(v) for v in edge_map.values())
    if max_edges is not None and edges > max_edges:
        return float("nan"), {}
    rng = np.random.default_rng(seed)
    tf_degrees = {tf: len(tgts) for tf, tgts in edge_map.items()}
    gold_list = [(tf, tgt) for tf, tgts in gold_map.items() for tgt in tgts]
    gold_set = set(gold_list)
    hits = []
    gene_array = np.array(genes)
    for _ in range(num_trials):
        rand_hits = 0
        for tf, deg in tf_degrees.items():
            if deg == 0:
                continue
            # sample without replacement
            choices = rng.choice(gene_array, size=min(deg, len(gene_array)), replace=False)
            for tgt in choices:
                if (tf, tgt) in gold_set:
                    rand_hits += 1
        hits.append(rand_hits)
    hits = np.array(hits, dtype=float)
    real_tp = sum(len(set(tgts) & gold_map.get(tf, set())) for tf, tgts in edge_map.items())
    p = float((hits >= real_tp).mean()) if hits.size else float("nan")
    return p, {
        "random_hits_mean": float(hits.mean()) if hits.size else float("nan"),
        "random_hits_std": float(hits.std()) if hits.size else float("nan"),
        "random_hits_max": float(hits.max()) if hits.size else float("nan"),
        "random_hits_min": float(hits.min()) if hits.size else float("nan"),
        "random_trials": int(num_trials),
        "random_hits_observed": float(real_tp),
    }


def compute_metrics(scores: np.ndarray, y_true: np.ndarray, tf_count: int, gene_count: int, k_values: Sequence[int] = DEFAULT_K) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    auprc = float(average_precision_score(y_true, scores))
    auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    metrics["auprc"] = auprc
    metrics["auroc"] = auroc
    total_positives = float(y_true.sum())
    # Global precision/recall@K
    precision_at = {}
    recall_at = {}
    max_k = min(max(k_values), scores.size)
    top_idx = np.argpartition(-scores, max_k - 1)[:max_k]
    tf_idx = top_idx // gene_count
    gene_idx = top_idx % gene_count
    order = np.lexsort((gene_idx, tf_idx, -scores[top_idx]))
    top_idx_sorted = top_idx[order]
    for k in k_values:
        k = min(k, scores.size)
        sel = top_idx_sorted[:k]
        if sel.size == 0:
            precision_at[f"precision@{k}"] = float("nan")
            recall_at[f"recall@{k}"] = float("nan")
            continue
        tp = int(y_true[sel].sum())
        prec = tp / sel.size if sel.size else 0.0
        rec = tp / total_positives if total_positives else 0.0
        precision_at[f"precision@{k}"] = float(prec)
        recall_at[f"recall@{k}"] = float(rec)

    # macro per-TF precision@K
    macro_precision = {}
    macro_precision_nonempty = {}
    for k in k_values:
        k_use = min(k, gene_count)
        tf_precisions = []
        tf_precisions_nonempty = []
        for t in range(tf_count):
            start = t * gene_count
            end = start + gene_count
            scores_tf = scores[start:end]
            y_tf = y_true[start:end]
            if k_use >= gene_count:
                chosen = np.arange(gene_count)
            else:
                topk_idx = np.argpartition(-scores_tf, k_use - 1)[:k_use]
                order_tf = np.lexsort((topk_idx, -scores_tf[topk_idx]))
                chosen = topk_idx[order_tf][:k_use]
            tp = float(y_tf[chosen].sum())
            tf_precisions.append(tp / len(chosen) if len(chosen) else 0.0)
            if y_tf.sum() > 0:
                tf_precisions_nonempty.append(tp / len(chosen) if len(chosen) else 0.0)
        macro_precision[f"macro_precision@{k_use}"] = float(np.mean(tf_precisions)) if tf_precisions else 0.0
        macro_precision_nonempty[f"macro_precision_nonempty@{k_use}"] = float(np.mean(tf_precisions_nonempty)) if tf_precisions_nonempty else float("nan")

    metrics.update(precision_at)
    metrics.update(recall_at)
    metrics.update(macro_precision)
    metrics.update(macro_precision_nonempty)

    # Non-empty TF overall metrics
    tf_has_pos = np.zeros(tf_count, dtype=bool)
    for t in range(tf_count):
        if y_true[t * gene_count : (t + 1) * gene_count].sum() > 0:
            tf_has_pos[t] = True
    if tf_has_pos.any():
        idx_keep = np.concatenate([np.arange(t * gene_count, (t + 1) * gene_count) for t in range(tf_count) if tf_has_pos[t]])
        y_keep = y_true[idx_keep]
        s_keep = scores[idx_keep]
        auprc_ne = float(average_precision_score(y_keep, s_keep))
        auroc_ne = float(roc_auc_score(y_keep, s_keep)) if len(np.unique(y_keep)) > 1 else float("nan")
        pi_ne = float(y_keep.mean())
        metrics.update(
            {
                "auprc_nonempty": auprc_ne,
                "auroc_nonempty": auroc_ne,
                "positive_ratio_nonempty": pi_ne,
                "pr_lift_nonempty": auprc_ne / pi_ne if pi_ne > 0 else float("nan"),
            }
        )

    return metrics


# -----------------------------
# Core evaluation
# -----------------------------

def evaluate_method(preds, gold_pairs, gold_map, tf_list, genes, k_values, per_tf_topk, random_trials, random_seed, pvalue_max_edges):
    tf_index = {t: i for i, t in enumerate(tf_list)}
    gene_index = {g: i for i, g in enumerate(genes)}
    tf_count, gene_count = len(tf_list), len(genes)
    size = tf_count * gene_count
    tf_edge_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    min_score = math.inf
    for tf, tgt, score in preds:
        ti = tf_index.get(tf)
        gi = gene_index.get(tgt)
        if ti is None or gi is None:
            continue
        min_score = min(min_score, score)
        if score > tf_edge_scores[tf].get(tgt, -math.inf):
            tf_edge_scores[tf][tgt] = score
    fill_value = (min_score - 1e-6) if min_score is not math.inf else -1.0
    scores = np.full(size, fill_value, dtype=np.float32)
    coverage = 0
    for tf, tgts in tf_edge_scores.items():
        base = tf_index[tf] * gene_count
        for tgt, sc in tgts.items():
            scores[base + gene_index[tgt]] = sc
        coverage += len(tgts)
    y_true = np.zeros(size, dtype=np.uint8)
    for tf, tgt in gold_pairs:
        ti = tf_index.get(tf)
        gi = gene_index.get(tgt)
        if ti is None or gi is None:
            continue
        y_true[ti * gene_count + gi] = 1
    metrics = compute_metrics(scores, y_true, tf_count, gene_count, k_values)
    total_pos = int(y_true.sum())
    metrics.update(
        {
            "positives_in_scored_pairs": total_pos,
            "total_scored_pairs": int(size),
            "coverage_edges": int(coverage),
            "coverage_ratio": float(coverage / size) if size else 0.0,
        }
    )
    per_tf_topk = max(int(per_tf_topk), 1)
    per_tf_edge_map: Dict[str, List[str]] = {}
    for tf, tgts in tf_edge_scores.items():
        sorted_targets = sorted(tgts.items(), key=lambda kv: kv[1], reverse=True)
        selected = [t for t, _ in sorted_targets[:per_tf_topk]]
        if selected:
            per_tf_edge_map[tf] = selected
    topk_summary = summarize_edge_map(per_tf_edge_map, gold_map, total_pos)
    metrics.update(
        {
            "per_tf_topk": per_tf_topk,
            "per_tf_topk_edges": topk_summary["edges"],
            "per_tf_topk_true_positives": topk_summary["true_positives"],
            "per_tf_topk_precision": topk_summary["precision"],
            "per_tf_topk_recall": topk_summary["recall"],
            "per_tf_topk_macro_precision": topk_summary["macro_precision"],
            "per_tf_topk_macro_recall": topk_summary["macro_recall"],
        }
    )
    p_val, rand_stats = degree_matched_pvalue(per_tf_edge_map, gold_map, genes, random_trials, random_seed, pvalue_max_edges)
    metrics["per_tf_topk_p_value"] = p_val
    for k, v in rand_stats.items():
        metrics[f"per_tf_topk_{k}"] = v
    return metrics


def run_for_dataset(name, config, methods, k_values, output_root, per_tf_topk, random_trials, random_seed, pvalue_max_edges):
    print(f"[eval] Dataset {name}")
    genes = load_genes(Path(config["genes"]))
    tf_list_full = load_tf_list(Path(config["tfs"]))
    gold = load_gold(Path(config["gold"]))
    gold_map = build_gold_map(gold)
    gold_tf_set = {tf for tf, _ in gold}
    (output_root / name).mkdir(parents=True, exist_ok=True)
    tf_orders = {
        "full": tf_list_full,
        "gold": [t for t in tf_list_full if t in gold_tf_set] or tf_list_full,
    }
    results: Dict[str, Dict[str, float]] = {}
    for method, path in methods.items():
        if path is None or not Path(path).exists():
            print(f"[eval][{name}] skip {method}: missing {path}")
            continue
        preds_all = load_predictions(Path(path))
        if not preds_all:
            print(f"[eval][{name}] skip {method}: no predictions.")
            continue
        method_res = {}
        for tag, tf_order in tf_orders.items():
            tf_filter = set(tf_order)
            preds = [p for p in preds_all if p[0] in tf_filter]
            if not preds:
                continue
            metrics = evaluate_method(
                preds,
                gold,
                gold_map,
                tf_order,
                genes,
                k_values,
                per_tf_topk=per_tf_topk,
                random_trials=random_trials,
                random_seed=random_seed,
                pvalue_max_edges=pvalue_max_edges,
            )
            method_res[tag] = metrics
            out_file = output_root / name / f"{method}_{tag}_metrics.json"
            out_file.write_text(json.dumps(metrics, indent=2))
            msg = f"[eval][{name}] {method} [{tag} TFs]: AUPRC={metrics['auprc']:.4g} AUROC={metrics['auroc']:.4g}"
            if "auprc_nonempty" in metrics:
                msg += f" AUPRC_nonempty={metrics['auprc_nonempty']:.4g}"
            if "auroc_nonempty" in metrics:
                auroc_ne = metrics['auroc_nonempty']
                msg += f" AUROC_nonempty={auroc_ne:.4g}" if auroc_ne == auroc_ne else " AUROC_nonempty=nan"
            msg += f" coverage={metrics['coverage_ratio']:.4g}"
            print(msg)
        if method_res:
            results[method] = method_res
    (output_root / name / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Unified TF->gene GRN evaluation across methods on a shared edge space.")
    p.add_argument("--output_root", default="output/evaluate/unified_metrics", help="Where to write per-dataset metrics.")
    p.add_argument("--k", default="100,500,1000,5000,10000", help="Comma-separated K values for Precision@K and macro Precision@K.")
    p.add_argument("--per_tf_topk", type=int, default=200, help="Per-TF Top-k edges used for network-level metrics and p-values.")
    p.add_argument("--random_trials", type=int, default=1000, help="Random degree-matched networks per method for the p-value.")
    p.add_argument("--random_seed", type=int, default=13, help="Seed for TF-target permutations in p-value estimation.")
    p.add_argument("--pvalue_max_edges", type=int, default=200_000, help="Skip p-value calculation if edges exceed this.")
    p.add_argument("--dataset_genes", type=str, default=None, help="Gene list for single-dataset eval.")
    p.add_argument("--dataset_tfs", type=str, default=None, help="TF list for single-dataset eval.")
    p.add_argument("--dataset_gold", type=str, default=None, help="Gold links for single-dataset eval.")
    p.add_argument("--methods_json", type=str, default=None, help="JSON mapping method_name -> grn_path (for ablations).")
    return p.parse_args()


def main():
    args = parse_args()
    k_values = [int(x) for x in args.k.split(",") if x.strip()]
    output_root = Path(args.output_root)
    if not (args.dataset_genes and args.dataset_tfs and args.dataset_gold and args.methods_json):
        print("ERROR: --dataset_genes, --dataset_tfs, --dataset_gold, and --methods_json are all required.")
        raise SystemExit(1)
    cfg = {"genes": Path(args.dataset_genes), "tfs": Path(args.dataset_tfs), "gold": Path(args.dataset_gold)}
    methods = json.loads(Path(args.methods_json).read_text())
    run_for_dataset(
        "single",
        cfg,
        methods,
        k_values,
        output_root,
        per_tf_topk=int(args.per_tf_topk),
        random_trials=int(args.random_trials),
        random_seed=int(args.random_seed),
        pvalue_max_edges=int(args.pvalue_max_edges),
    )


if __name__ == "__main__":
    main()
