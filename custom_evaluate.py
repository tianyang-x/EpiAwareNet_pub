import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent / "evaluate"))
import epiaware_metrics  # noqa: E402
from random_network_pvalue import (  # noqa: E402
    build_gold_map,
    degree_matched_pvalue,
    summarize_edge_map,
)


def _strip_prefix(name: str, prefix: str) -> str:
    if prefix and isinstance(name, str) and name.startswith(prefix):
        return name[len(prefix) :]
    return str(name)


def load_list_file(path: Path, prefix: str = "") -> List[str]:
    values: List[str] = []
    if path is None or not path.exists():
        return values
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item:
                continue
            values.append(_strip_prefix(item, prefix))
    return values


def unique_preserving_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        ordered.append(val)
    return ordered

def load_predictions_pandas(path, tf_prefix="gene:", target_prefix="gene:"):
    """
    Load predictions using pandas for faster processing of large files.
    """
    print(f"Reading CSV with pandas: {path}")
    # Use C engine for speed, assume header exists
    df = pd.read_csv(path, engine='c')
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Identify columns
    tf_col = next((c for c in df.columns if c in ["tf", "regulator", "gene_i", "source", "from"]), None)
    target_col = next((c for c in df.columns if c in ["target", "gene_j", "gene", "sink", "to"]), None)
    score_col = next((c for c in df.columns if c in ["score", "probability", "logit", "p", "value", "importance", "weight"]), None)
    
    if not all([tf_col, target_col, score_col]):
        raise ValueError(f"Could not identify tf, target, and score columns. Found: {df.columns.tolist()}")
    
    # Rename columns to standard names
    df = df.rename(columns={tf_col: 'tf', target_col: 'target', score_col: 'score'})
    
    # Strip prefixes
    if tf_prefix:
        # Vectorized string operation is much faster
        df['tf'] = df['tf'].astype(str).str.replace(f"^{tf_prefix}", "", regex=True)
    if target_prefix:
        df['target'] = df['target'].astype(str).str.replace(f"^{target_prefix}", "", regex=True)
        
    return df[['tf', 'target', 'score']]

def deduplicate_pandas(df):
    """
    Deduplicate predictions using pandas, keeping the highest score for each TF-Target pair.
    """
    # Sort by score descending so that drop_duplicates(keep='first') keeps the highest score
    df = df.sort_values('score', ascending=False)
    df = df.drop_duplicates(subset=['tf', 'target'], keep='first')
    return df


def build_topk_edge_map(df: pd.DataFrame, topk: int) -> Dict[str, List[str]]:
    topk = max(int(topk), 1)
    edge_map: Dict[str, List[str]] = {}
    grouped = df.groupby("tf", sort=False)
    for tf, group in grouped:
        selected = group.sort_values("score", ascending=False).head(topk)["target"].tolist()
        if selected:
            edge_map[tf] = selected
    return edge_map


def build_edge_map_from_df(df: pd.DataFrame) -> Dict[str, List[str]]:
    edge_map: Dict[str, List[str]] = {}
    grouped = df.groupby("tf", sort=False)
    for tf, group in grouped:
        edge_map[tf] = group.sort_values("score", ascending=False)["target"].tolist()
    return edge_map

def save_curves_with_baseline(
    output_dir: Path,
    precision: np.ndarray,
    recall: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    baseline: float,
    global_baseline: float = None,
) -> None:
    plt.figure(figsize=(5, 4))
    plt.step(recall, precision, where="post", color="#2C7BB6", label='Model')
    plt.plot([0, 1], [baseline, baseline], linestyle='--', color='gray', label=f'Random Baseline ({baseline:.4f})')

    if global_baseline is not None:
        plt.plot([0, 1], [global_baseline, global_baseline], linestyle=':', color='green', label=f'Global Random ({global_baseline:.4f})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    (output_dir / "pr_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=200)
    plt.close()

    if fpr is not None and tpr is not None:
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color="#D7191C", label='Model')
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", label='Random Baseline')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close()

def export_and_evaluate_network(
    df: pd.DataFrame,
    threshold: float,
    gold_pairs,
    gold_map,
    candidate_genes: Sequence[str],
    total_gold: int,
    output_dir: Path,
    slug: str,
    label: str,
    global_baseline: float,
    random_trials: int,
    random_seed: int,
    pvalue_max_edges: Optional[int],
    subset_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, float]]:
    if subset_df is None:
        subset_df = df[df["score"] >= threshold].copy()
    else:
        subset_df = subset_df.copy()
    csv_path = output_dir / f"{slug}.csv"
    subset_df.to_csv(csv_path, index=False)
    print(f"Exported {len(subset_df)} edges to {csv_path}")

    if subset_df.empty:
        print(f"Skipping evaluation for {label} because no edges met the threshold.")
        return None

    preds_subset = list(subset_df.itertuples(index=False, name=None))
    try:
        (
            metrics_subset,
            prec_subset,
            rec_subset,
            fpr_subset,
            tpr_subset,
        ) = epiaware_metrics.compute_metrics(preds_subset, gold_pairs)
    except ValueError as exc:
        # For thresholded exports it's valid to have 0 gold overlaps.
        # Still write out metrics for bookkeeping, but skip curves.
        logger.warning("%s: %s", label, exc)
        metrics_subset = {
            "positives_in_predictions": 0,
            "total_predictions": int(len(preds_subset)),
            "positive_ratio": 0.0,
            "auprc": 0.0,
            "auroc": float("nan"),
        }
        prec_subset = rec_subset = fpr_subset = tpr_subset = None

    metrics_subset["precision"] = (
        metrics_subset["positives_in_predictions"] / metrics_subset["total_predictions"]
        if metrics_subset["total_predictions"]
        else 0.0
    )
    metrics_subset["recall"] = (
        metrics_subset["positives_in_predictions"] / total_gold
        if total_gold
        else 0.0
    )

    edge_map = build_edge_map_from_df(subset_df)
    summary = summarize_edge_map(edge_map, gold_map, total_gold)
    metrics_subset["network_edges"] = summary["edges"]
    metrics_subset["network_true_positives"] = summary["true_positives"]
    metrics_subset["macro_precision"] = summary["macro_precision"]
    metrics_subset["macro_recall"] = summary["macro_recall"]

    p_value, rand_stats = degree_matched_pvalue(
        edge_map,
        gold_map,
        candidate_genes,
        num_trials=random_trials,
        seed=random_seed,
        max_edges=pvalue_max_edges,
    )
    metrics_subset["p_value"] = p_value
    for key, value in rand_stats.items():
        metrics_subset[f"p_value_{key}"] = value

    subset_dir = output_dir / slug
    epiaware_metrics.save_metrics_json(metrics_subset, subset_dir)
    if prec_subset is not None and rec_subset is not None:
        save_curves_with_baseline(
            subset_dir,
            prec_subset,
            rec_subset,
            fpr_subset,
            tpr_subset,
            metrics_subset["positive_ratio"],
            global_baseline=global_baseline,
        )

    print(f"\nSubset Metrics ({label}):")
    print(json.dumps(metrics_subset, indent=2))
    return metrics_subset


def evaluate_grn(
    predictions_path: Path,
    gold_path: Path,
    gene_path: Optional[Path],
    output_dir: Path,
    tf_prefix: str = "gene:",
    target_prefix: str = "gene:",
    per_tf_topk: int = 200,
    pvalue_trials: int = 1000,
    pvalue_seed: int = 13,
    pvalue_max_edges: int = 200_000,
    precision_targets: Sequence[float] = (0.20, 0.25, 0.30, 0.35),
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_tf_topk = max(int(per_tf_topk), 1)

    print(f"Loading predictions from {predictions_path}...")
    df_preds = load_predictions_pandas(str(predictions_path), tf_prefix, target_prefix)
    print(f"Loaded {len(df_preds)} predictions.")

    print("Deduplicating predictions...")
    df_preds = deduplicate_pandas(df_preds)
    print(f"Unique predictions: {len(df_preds)}")

    preds = list(df_preds.itertuples(index=False, name=None))

    print(f"Loading gold set from {gold_path}...")
    gold_pairs = epiaware_metrics.load_gold_file(Path(gold_path), tf_prefix, target_prefix)
    gold_set = set(gold_pairs)
    total_gold = len(gold_set)
    print(f"Loaded {len(gold_pairs)} gold pairs.")

    candidate_genes: List[str] = []
    if gene_path is not None and gene_path.exists():
        candidate_genes = load_list_file(gene_path, target_prefix)
        if candidate_genes:
            print(f"Loaded {len(candidate_genes)} candidate genes from {gene_path}")
        else:
            logger.warning("Candidate gene list %s is empty.", gene_path)

    global_baseline = None
    if candidate_genes:
        n_genes = len(candidate_genes)
        total_space = n_genes * n_genes
        n_gold = len(gold_set)
        global_baseline = n_gold / total_space
        print(
            f"Global Random Baseline: {global_baseline:.6f} (Genes: {n_genes}, Total Space: {total_space}, Positives: {n_gold})"
        )

    if not candidate_genes:
        inferred = unique_preserving_order(df_preds["target"].tolist())
        if not inferred:
            inferred = unique_preserving_order([tgt for _, tgt in gold_pairs])
        candidate_genes = inferred
        logger.warning(
            "Falling back to %d unique targets from predictions/gold as candidate genes.",
            len(candidate_genes),
        )

    print("Computing metrics...")
    metrics, precision, recall, fpr, tpr = epiaware_metrics.compute_metrics(preds, gold_pairs)
    metrics["auprc_baseline"] = metrics["positive_ratio"]
    if global_baseline is not None:
        metrics["auprc_global_baseline"] = global_baseline
    metrics["precision"] = (
        metrics["positives_in_predictions"] / metrics["total_predictions"]
        if metrics["total_predictions"]
        else 0.0
    )
    metrics["recall"] = (
        metrics["positives_in_predictions"] / total_gold if total_gold else 0.0
    )

    gold_map = build_gold_map(gold_pairs)
    per_tf_edge_map = build_topk_edge_map(df_preds, per_tf_topk)
    print("Computing per-TF Top-k overlap and p-value...")
    topk_summary = summarize_edge_map(per_tf_edge_map, gold_map, total_gold)
    metrics["per_tf_topk"] = int(per_tf_topk)
    metrics["per_tf_topk_edges"] = topk_summary["edges"]
    metrics["per_tf_topk_true_positives"] = topk_summary["true_positives"]
    metrics["per_tf_topk_precision"] = topk_summary["precision"]
    metrics["per_tf_topk_recall"] = topk_summary["recall"]
    metrics["per_tf_topk_macro_precision"] = topk_summary["macro_precision"]
    metrics["per_tf_topk_macro_recall"] = topk_summary["macro_recall"]

    p_value, rand_stats = degree_matched_pvalue(
        per_tf_edge_map,
        gold_map,
        candidate_genes,
        num_trials=pvalue_trials,
        seed=pvalue_seed,
        max_edges=pvalue_max_edges,
    )
    metrics["per_tf_topk_p_value"] = p_value
    for key, value in rand_stats.items():
        metrics[f"per_tf_topk_{key}"] = value

    scores = df_preds["score"].to_numpy(dtype=np.float32)
    y_true = np.array(
        [1 if (tf, tgt) in gold_set else 0 for tf, tgt, _ in preds],
        dtype=np.int32,
    )

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    print(f"Saving results to {output_dir}...")
    epiaware_metrics.save_metrics_json(metrics, output_dir)
    save_curves_with_baseline(
        output_dir,
        precision,
        recall,
        fpr,
        tpr,
        metrics["positive_ratio"],
        global_baseline=global_baseline,
    )

    print("\n--- Threshold Analysis ---")
    prec_curve, rec_curve, thresholds = epiaware_metrics.precision_recall_curve(y_true, scores)

    usable_prec = prec_curve[:-1]
    usable_thresholds = thresholds

    if len(usable_thresholds) == 0:
        print("Skipping threshold analysis because no valid thresholds were produced.")
        return

    threshold_jobs = []
    max_prec_idx = int(np.argmax(usable_prec))
    threshold_jobs.append(
        (
            "network_max_precision",
            "Max Precision",
            float(usable_thresholds[max_prec_idx]),
            float(usable_prec[max_prec_idx]),
        )
    )

    for target in precision_targets:
        candidates = np.where(usable_prec >= float(target))[0]
        if len(candidates) == 0:
            idx = int(np.argmax(usable_prec))
            fallback = True
        else:
            idx = int(candidates[0])
            fallback = False
        threshold_jobs.append(
            (
                f"network_precision_{str(target).replace('.', '_')}",
                f"Precision ≥ {target:.2f}" + (" (fallback)" if fallback else ""),
                float(usable_thresholds[idx]),
                float(usable_prec[idx]),
            )
        )

    threshold_results = []
    for slug, label, thresh_value, curve_prec in threshold_jobs:
        subset_df = df_preds[df_preds["score"] >= thresh_value].copy()
        if subset_df.empty:
            print(f"\nNetwork Cut: {label} -> no edges above threshold {thresh_value:.6f}. Skipping.")
            continue

        subset_metrics = export_and_evaluate_network(
            df_preds,
            thresh_value,
            gold_pairs,
            gold_map,
            candidate_genes,
            total_gold,
            output_dir,
            slug,
            label,
            global_baseline,
            random_trials=pvalue_trials,
            random_seed=pvalue_seed,
            pvalue_max_edges=pvalue_max_edges,
            subset_df=subset_df,
        )

        if subset_metrics is None:
            continue

        n_selected = subset_metrics["total_predictions"]
        n_tp = subset_metrics["positives_in_predictions"]
        precision_val = subset_metrics["precision"]
        recall_val = subset_metrics["recall"]
        f1 = (
            2 * precision_val * recall_val / (precision_val + recall_val)
            if (precision_val + recall_val)
            else 0.0
        )

        print(f"\nNetwork Cut: {label}")
        print(f"  Threshold: {thresh_value:.6f}")
        print(f"  Edges Selected: {n_selected}")
        print(f"  True Positives: {n_tp}")
        print(f"  Precision: {precision_val:.6f}")
        print(f"  Recall:    {recall_val:.6f}")
        print(f"  F1 Score:  {f1:.6f}")

        threshold_results.append(
            {
                "name": label,
                "slug": slug,
                "threshold": thresh_value,
                "curve_precision": curve_prec,
                "edges": n_selected,
                "true_positives": n_tp,
                "precision": precision_val,
                "recall": recall_val,
                "f1": f1,
            }
        )

    if threshold_results:
        with open(output_dir / "threshold_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(threshold_results, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GRN CSV against a gold set and export clipped networks at precision targets."
    )
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV (tf,target,score).")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold TF-target file.")
    parser.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Optional genes.txt path used to compute global random baseline (|gold| / N_genes^2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to <predictions parent>/custom_eval).",
    )
    parser.add_argument("--tf_prefix", type=str, default="gene:")
    parser.add_argument("--target_prefix", type=str, default="gene:")
    parser.add_argument(
        "--per_tf_topk",
        type=int,
        default=200,
        help="Per-TF Top-k edges used for network-level reporting and p-value.",
    )
    parser.add_argument(
        "--pvalue_trials",
        type=int,
        default=1000,
        help="Number of random degree-matched networks for the empirical p-value.",
    )
    parser.add_argument(
        "--pvalue_seed",
        type=int,
        default=13,
        help="Seed for random TF-target permutations.",
    )
    parser.add_argument(
        "--pvalue_max_edges",
        type=int,
        default=200_000,
        help="Skip p-value calculation if the network has more edges than this limit.",
    )
    parser.add_argument(
        "--precision_targets",
        type=float,
        nargs="*",
        default=[0.20, 0.25, 0.30, 0.35],
        help="Precision targets used for clipped network exports.",
    )
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    predictions_path = Path(args.predictions)
    gold_path = Path(args.gold)
    gene_path = Path(args.genes) if args.genes else None
    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent / "custom_eval"

    evaluate_grn(
        predictions_path=predictions_path,
        gold_path=gold_path,
        gene_path=gene_path,
        output_dir=output_dir,
        tf_prefix=args.tf_prefix,
        target_prefix=args.target_prefix,
        per_tf_topk=int(args.per_tf_topk),
        pvalue_trials=int(args.pvalue_trials),
        pvalue_seed=int(args.pvalue_seed),
        pvalue_max_edges=int(args.pvalue_max_edges),
        precision_targets=tuple(float(x) for x in args.precision_targets),
    )

    print("Done.")

if __name__ == "__main__":
    main()
