#!/usr/bin/env python3
"""
Plot PR/ROC/Precision@K/Hit@K curves for non-empty TFs across datasets/methods.

This script reuses the unified evaluation inputs (shared gene lists, TF lists,
gold edges, and GRN predictions) and restricts all computations to TFs that
have at least one positive target in the gold standard.  The curves are useful
for visualizing how each method behaves on the TFs that matter to reviewers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from hashlib import blake2s
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - fallback if tqdm missing
    def tqdm(iterable=None, **kwargs):
        return iterable
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FONT_PATH = REPO_ROOT / "fonts" / "helvetica.ttf"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Helvetica"
plt.rcParams["mathtext.it"] = "Helvetica:italic"
plt.rcParams["mathtext.bf"] = "Helvetica:bold"
plt.rcParams["figure.figsize"] = (2.7, 2.0)
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

from evaluate.unified_baseline_eval import (  # noqa: E402
    build_gold_map,
    load_genes,
    load_gold,
    load_predictions,
    load_tf_list,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_CONFIG = {
    "mN": {
        "genes": REPO_ROOT / "datasets/mN_combined/genes.txt",
        "tfs": REPO_ROOT / "datasets/plusNshare/regulators.txt",
        "gold": REPO_ROOT / "datasets/TF_Target_Validation_inSingleCell_val.txt",
    },
    "pN": {
        "genes": REPO_ROOT / "datasets/pN_combined/genes.txt",
        "tfs": REPO_ROOT / "datasets/plusNshare/regulators.txt",
        "gold": REPO_ROOT / "datasets/TF_Target_Validation_inSingleCell_val.txt",
    },
    "pbmc": {
        "genes": REPO_ROOT / "datasets/pbmc_preprocessed/genes.txt",
        "tfs": REPO_ROOT / "datasets/pbmc_preprocessed/tf_list.txt",
        "gold": REPO_ROOT / "datasets/dorothea_high_confidence_links_val.csv",
    },
    "mouse_brain": {
        "genes": REPO_ROOT / "datasets/mouse_brain_combined/genes.txt",
        "tfs": REPO_ROOT / "datasets/mouse_brain_combined/tfs_trrust_mouse.txt",
        "gold": REPO_ROOT / "datasets/mouse_brain_combined/positive_links_trrust_mouse_val_1266.tsv",
    },
}

METHOD_PATHS: Dict[str, Dict[str, Path]] = {}

METHOD_COLORS = {
    "EpiAwareNet": "#d62728",
    "EpiAwareNet (nnPU)": "#e377c2",
}

DEFAULT_K = [100, 500, 1000, 5000, 10000]
CACHE_DIR = REPO_ROOT / "cache" / "tf_curves"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def select_tf_order(tf_list: Sequence[str], gold_pairs: Sequence[tuple[str, str]]) -> List[str]:
    gold_tf = {tf for tf, _ in gold_pairs}
    ordered = [tf for tf in tf_list if tf in gold_tf]
    if ordered:
        return ordered
    return sorted(gold_tf)


def build_score_arrays(
    predictions: Sequence[tuple[str, str, float]],
    gold_pairs: Sequence[tuple[str, str]],
    tf_order: Sequence[str],
    genes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    tf_index = {tf: i for i, tf in enumerate(tf_order)}
    gene_index = {g: i for i, g in enumerate(genes)}
    tf_count, gene_count = len(tf_order), len(genes)
    if tf_count == 0 or gene_count == 0:
        raise ValueError("Empty TF or gene list.")
    min_score = math.inf
    for _, _, score in predictions:
        if score < min_score:
            min_score = score
    fill_value = (min_score - 1e-6) if predictions else -1.0
    scores = np.full((tf_count, gene_count), fill_value, dtype=np.float32)
    for tf, tgt, score in predictions:
        ti = tf_index.get(tf)
        gi = gene_index.get(tgt)
        if ti is None or gi is None:
            continue
        # retain the highest score if duplicates exist
        if score > scores[ti, gi]:
            scores[ti, gi] = score
    y_true = np.zeros((tf_count, gene_count), dtype=np.uint8)
    for tf, tgt in gold_pairs:
        ti = tf_index.get(tf)
        gi = gene_index.get(tgt)
        if ti is None or gi is None:
            continue
        y_true[ti, gi] = 1
    return scores, y_true


def flatten_nonempty(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tf_mask = labels.sum(axis=1) > 0
    scores_sel = scores[tf_mask]
    labels_sel = labels[tf_mask]
    return scores_sel.reshape(-1), labels_sel.reshape(-1)


def compute_precision_at_k(sorted_labels: np.ndarray, ks: Sequence[int]) -> Dict[int, Dict[str, float]]:
    total_pos = float(sorted_labels.sum())
    cumsum = np.cumsum(sorted_labels, dtype=np.float64)
    result: Dict[int, Dict[str, float]] = {}
    n = sorted_labels.size
    for k in ks:
        k_eff = min(int(k), n)
        if k_eff <= 0:
            result[k] = {"precision": float("nan"), "recall": float("nan")}
            continue
        tp = float(cumsum[k_eff - 1])
        precision = tp / k_eff
        recall = tp / total_pos if total_pos > 0 else float("nan")
        result[k] = {"precision": precision, "recall": recall}
    return result


def compute_hit_at_k(scores: np.ndarray, labels: np.ndarray, ks: Sequence[int]) -> Dict[int, float]:
    tf_count, gene_count = scores.shape
    hits: Dict[int, float] = {}
    for k in ks:
        k_eff = min(int(k), gene_count)
        if k_eff <= 0:
            hits[k] = float("nan")
            continue
        success = 0
        for tf_idx in range(tf_count):
            label_row = labels[tf_idx]
            if label_row.sum() == 0:
                continue
            score_row = scores[tf_idx]
            if k_eff >= gene_count:
                chosen = np.arange(gene_count)
            else:
                idx = np.argpartition(score_row, -k_eff)[-k_eff:]
                chosen = idx
            if label_row[chosen].any():
                success += 1
        hits[k] = success / tf_count if tf_count else float("nan")
    return hits


def normalize_metric_payload(metrics: Dict[str, object]) -> Dict[str, object]:
    """Ensure cached metric keys are ints so downstream plotting is consistent."""
    prec = metrics.get("precision_at")
    if isinstance(prec, dict):
        metrics["precision_at"] = {int(k): v for k, v in prec.items()}
    hits = metrics.get("hit_at")
    if isinstance(hits, dict):
        metrics["hit_at"] = {int(k): v for k, v in hits.items()}
    return metrics


def summarize_method(
    scores: np.ndarray,
    labels: np.ndarray,
    ks: Sequence[int],
) -> Dict[str, object]:
    scores_vec, labels_vec = flatten_nonempty(scores, labels)
    if labels_vec.sum() == 0:
        raise ValueError("No positives remain after restricting to non-empty TFs.")
    order = np.argsort(scores_vec)[::-1]
    sorted_labels = labels_vec[order]
    pr_prec, pr_rec, _ = precision_recall_curve(labels_vec, scores_vec)
    fpr, tpr, _ = roc_curve(labels_vec, scores_vec)
    precision_at = compute_precision_at_k(sorted_labels, ks)
    hits_at = compute_hit_at_k(scores[labels.sum(axis=1) > 0], labels[labels.sum(axis=1) > 0], ks)
    return {
        "pr_curve": {"precision": pr_prec.tolist(), "recall": pr_rec.tolist()},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "precision_at": precision_at,
        "hit_at": hits_at,
        "total_positives": int(labels_vec.sum()),
        "total_edges": int(labels_vec.size),
    }


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))


# --------------------------------------------------------------------------- #
# Caching helpers
# --------------------------------------------------------------------------- #


def _fingerprint(path: Path) -> str:
    """Cheap fingerprint based on mtime and size (avoids hashing huge CSVs)."""
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def build_signature(
    dataset: str,
    method: str,
    pred_path: Path,
    gold_path: Path,
    genes_path: Path,
    tf_path: Path,
    ks: Sequence[int],
) -> str:
    payload = "|".join(
        [
            dataset,
            method,
            _fingerprint(pred_path),
            _fingerprint(gold_path),
            _fingerprint(genes_path),
            _fingerprint(tf_path),
            ",".join(str(k) for k in ks),
        ]
    )
    return blake2s(payload.encode("utf-8"), digest_size=8).hexdigest()


def load_cached_metrics(cache_dir: Path, dataset: str, method: str, signature: str) -> Dict[str, object] | None:
    cache_file = cache_dir / f"{dataset}__{method}__{signature}.json"
    if not cache_file.exists():
        return None
    try:
        metrics = json.loads(cache_file.read_text())
        return normalize_metric_payload(metrics)
    except json.JSONDecodeError:
        return None


def save_cached_metrics(cache_dir: Path, dataset: str, method: str, signature: str, metrics: Dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset}__{method}__{signature}.json"
    cache_file.write_text(json.dumps(metrics))


# --------------------------------------------------------------------------- #
# Axis helpers
# --------------------------------------------------------------------------- #


def make_fixed_main_axes(
    *,
    figsize: tuple[float, float] = (2.7, 2.0),
    margins_in: tuple[float, float, float, float] = (0.62, 0.12, 0.52, 0.18),
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a figure whose *main plotting box* has a fixed physical size.

    Why this exists:
    - `constrained_layout` / `tight_layout` make the axes box vary with labels.
    - Review figures often require identical internal plotting area across panels.

    Parameters
    - figsize: (width, height) of the whole figure in inches.
    - margins_in: (left, right, bottom, top) margins in inches.

    Returns
    - (fig, ax) where `ax` occupies the exact same rectangle (in inches) for all calls.
    """

    fig_w, fig_h = figsize
    left_in, right_in, bottom_in, top_in = margins_in
    if left_in + right_in >= fig_w or bottom_in + top_in >= fig_h:
        raise ValueError(
            f"Invalid margins {margins_in} for figsize={figsize}: no space left for axes."
        )

    left = left_in / fig_w
    bottom = bottom_in / fig_h
    width = 1.0 - (left_in + right_in) / fig_w
    height = 1.0 - (bottom_in + top_in) / fig_h

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([left, bottom, width, height])
    return fig, ax


def auto_ylim(values: Sequence[float], lower: float = 0.0, upper: float = 1.0, min_span: float = 0.05) -> tuple[float, float]:
    """Choose a tight y-range using quantiles; avoids rare spikes forcing ylim to 1."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return lower, upper
    if len(vals) > 5:
        q_low = float(np.quantile(vals, 0.01))
        q_high = float(np.quantile(vals, 0.99))
    else:
        q_low = min(vals)
        q_high = max(vals)
    span = q_high - q_low
    pad = 0.1 * span if span > 0 else 0.02
    y0 = max(lower, q_low - pad)
    y1 = min(upper, q_high + pad)
    if y1 - y0 < min_span:
        mid = 0.5 * (y0 + y1)
        y0 = max(lower, mid - min_span / 2)
        y1 = min(upper, mid + min_span / 2)
    return y0, y1


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _save_figure(fig: matplotlib.figure.Figure, path_stem: Path, formats: Sequence[str]) -> None:
    for ext in formats:
        fig.savefig(path_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def _first_available_image(path_stem: Path, formats: Sequence[str]) -> Path | None:
    preferred = ["png"] + [ext for ext in formats if ext != "png"]
    for ext in preferred:
        candidate = path_stem.with_suffix(f".{ext}")
        if candidate.exists():
            return candidate
    return None


def export_combined_pdf(output_dir: Path, datasets: Sequence[str], formats: Sequence[str], pdf_path: Path | None = None) -> None:
    if not datasets:
        return
    pdf_path = pdf_path or (output_dir / "tf_curves_combined.pdf")
    with PdfPages(pdf_path) as pdf:
        for dataset in datasets:
            stems = {
                "PR": output_dir / f"{dataset}_pr",
                "ROC": output_dir / f"{dataset}_roc",
                "P@K": output_dir / f"{dataset}_precision_at_k",
                "Hit@K": output_dir / f"{dataset}_hit_at_k",
            }
            images = {name: _first_available_image(stem, formats) for name, stem in stems.items()}
            if not any(images.values()):
                continue
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for ax, (title, img_path) in zip(axes.flat, images.items()):
                if img_path is None:
                    ax.axis("off")
                    continue
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"{dataset} - {title}", fontsize=12)
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[info] Combined PDF written to {pdf_path}")


def plot_dataset(
    dataset: str,
    results: Dict[str, Dict[str, object]],
    output_dir: Path,
    formats: Sequence[str],
    ks: Sequence[int],
    fixed_ylim: bool,
) -> None:
    colors = METHOD_COLORS
    k_ticks = [k for k in ks if k > 0]
    output_dir.mkdir(parents=True, exist_ok=True)
    # derive a dataset-level baseline precision if present in any metrics payload
    baseline_prec = None
    for metrics in results.values():
        tot_pos = metrics.get("total_positives")
        tot_edges = metrics.get("total_edges")
        if tot_pos is not None and tot_edges:
            baseline_prec = tot_pos / tot_edges
            break

    # Precision-Recall
    fig_pr, ax_pr = make_fixed_main_axes()
    pr_values: List[float] = []
    for method, metrics in results.items():
        color = colors.get(method, None)
        pr = metrics["pr_curve"]
        ax_pr.plot(pr["recall"], pr["precision"], label=method, color=color, linewidth=2)
        pr_values.extend(pr["precision"])
    if baseline_prec is not None:
        ax_pr.axhline(baseline_prec, color="gray", linestyle="--", linewidth=1, label="baseline")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    if fixed_ylim:
        ax_pr.set_ylim(0, 1)
    elif pr_values:
        y0, y1 = auto_ylim(pr_values, lower=0, upper=1, min_span=0.05)
        ax_pr.set_ylim(y0, y1)
    ax_pr.set_xlim(0, 1)
    ax_pr.grid(True, linestyle="--", linewidth=0.5)
    _save_figure(fig_pr, output_dir / f"{dataset}_pr", formats)

    # ROC
    fig_roc, ax_roc = make_fixed_main_axes()
    for method, metrics in results.items():
        color = colors.get(method, None)
        roc = metrics["roc_curve"]
        ax_roc.plot(roc["fpr"], roc["tpr"], label=method, color=color, linewidth=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.grid(True, linestyle="--", linewidth=0.5)
    _save_figure(fig_roc, output_dir / f"{dataset}_roc", formats)

    # Precision@K
    fig_prec, ax_prec = make_fixed_main_axes()
    prec_values: List[float] = []
    for method, metrics in results.items():
        color = colors.get(method, None)
        precision_points = metrics["precision_at"]
        xs = []
        ys = []
        for k in ks:
            if k <= 0:
                continue
            xs.append(k)
            ys.append(precision_points[k]["precision"])
            prec_values.append(precision_points[k]["precision"])
        ax_prec.plot(xs, ys, marker="o", label=method, color=color, linewidth=2)
    ax_prec.set_xlabel("K (global edges)")
    ax_prec.set_ylabel("Precision")
    ax_prec.set_xscale("log")
    if fixed_ylim:
        ax_prec.set_ylim(0, 1)
    elif prec_values:
        y0, y1 = auto_ylim(prec_values, lower=0, upper=1, min_span=0.05)
        ax_prec.set_ylim(y0, y1)
    ax_prec.grid(True, linestyle="--", linewidth=0.5)

    # Capture legend entries before the figure is closed.
    handles, labels = ax_prec.get_legend_handles_labels()
    _save_figure(fig_prec, output_dir / f"{dataset}_precision_at_k", formats)

    # Hit@K
    fig_hit, ax_hit = make_fixed_main_axes()
    hit_values: List[float] = []
    for method, metrics in results.items():
        color = colors.get(method, None)
        hit_points = metrics["hit_at"]
        xs = []
        ys = []
        for k in ks:
            if k <= 0:
                continue
            xs.append(k)
            ys.append(hit_points[k])
            hit_values.append(hit_points[k])
        ax_hit.plot(xs, ys, marker="o", label=method, color=color, linewidth=2)
    ax_hit.set_xlabel("K per TF")
    ax_hit.set_ylabel("Hit Rate")
    ax_hit.set_xscale("log")
    if fixed_ylim:
        ax_hit.set_ylim(0, 1)
    elif hit_values:
        y0, y1 = auto_ylim(hit_values, lower=0, upper=1, min_span=0.05)
        ax_hit.set_ylim(y0, y1)
    ax_hit.grid(True, linestyle="--", linewidth=0.5)
    _save_figure(fig_hit, output_dir / f"{dataset}_hit_at_k", formats)

    # Legend-only figure
    if handles:
        fig_leg, ax_leg = plt.subplots(figsize=(10.0, 0.3))
        ax_leg.axis("off")
        ax_leg.legend(handles, labels, loc="center", ncol=len(labels), frameon=False)
        _save_figure(fig_leg, output_dir / f"{dataset}_legend", formats)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(description="Plot TF-level curves for four datasets × multiple methods.")
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_CONFIG.keys()), help="Datasets to plot.")
    parser.add_argument("--methods", nargs="*", default=None, help="Methods to include (inferred from --methods_json if omitted).")
    parser.add_argument("--methods_json", type=Path, default=None, help="JSON file mapping method_name -> {dataset: grn_path}.")
    parser.add_argument("--k_values", type=str, default=",".join(str(k) for k in DEFAULT_K), help="Comma-separated K list.")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "figures" / "tf_curves")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"], help="Image formats to export.")
    parser.add_argument("--json_out", type=Path, default=REPO_ROOT / "output" / "figures" / "tf_curves" / "metrics.json")
    parser.add_argument("--metrics_in", type=Path, default=None, help="Optional metrics cache to reuse.")
    parser.add_argument("--no_cache", action="store_true", help="Disable incremental on-disk caching of metric computation.")
    parser.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--fixed_ylim", action="store_true", help="Use fixed [0,1] y-limits instead of auto-scaling.")
    parser.add_argument("--combined_pdf", type=Path, default=None, help="Optional combined PDF path (defaults to output_dir/tf_curves_combined.pdf).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.methods_json and args.methods_json.exists():
        raw = json.loads(args.methods_json.read_text())
        for method_name, paths in raw.items():
            METHOD_PATHS[method_name] = {ds: Path(p) for ds, p in paths.items()}
    ks = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    datasets = [ds for ds in args.datasets if ds in DATASET_CONFIG]
    methods_list = args.methods if args.methods else list(METHOD_PATHS.keys())
    methods = [m for m in methods_list if m in METHOD_PATHS]
    tqdm_fn = (lambda it, **kw: tqdm(it, **kw)) if not args.no_progress else (lambda it, **kw: it)
    ensure_paths_exist(
        [DATASET_CONFIG[ds]["genes"] for ds in datasets]
        + [DATASET_CONFIG[ds]["tfs"] for ds in datasets]
        + [DATASET_CONFIG[ds]["gold"] for ds in datasets]
    )
    cached_metrics: Dict[str, Dict[str, Dict[str, object]]] = {}
    if args.metrics_in and args.metrics_in.exists():
        try:
            cached_raw = json.loads(args.metrics_in.read_text())
            for dataset in datasets:
                ds_metrics = cached_raw.get(dataset, {})
                filtered = {m: normalize_metric_payload(ds_metrics[m]) for m in methods if m in ds_metrics}
                if filtered:
                    cached_metrics[dataset] = filtered
        except json.JSONDecodeError as exc:
            print(f"[warn] Failed to parse cached metrics ({args.metrics_in}): {exc}")

    all_metrics: Dict[str, Dict[str, object]] = {ds: dict(metrics) for ds, metrics in cached_metrics.items()}

    for dataset in tqdm_fn(datasets, desc="Datasets"):
        cfg = DATASET_CONFIG[dataset]
        genes = load_genes(cfg["genes"])
        tf_full = load_tf_list(cfg["tfs"], fallback=[])
        gold_pairs = load_gold(cfg["gold"])
        gold_pairs = [(tf, tgt) for tf, tgt in gold_pairs if tf and tgt]
        tf_order = select_tf_order(tf_full, gold_pairs)
        if not tf_order:
            print(f"[warn] Dataset {dataset}: no non-empty TFs found, skipping.")
            continue
        dataset_results: Dict[str, Dict[str, object]] = dict(all_metrics.get(dataset, {}))
        for method in tqdm_fn(methods, desc=f"{dataset} methods", leave=False):
            if method in dataset_results:
                continue
            pred_path = METHOD_PATHS[method].get(dataset)
            if pred_path is None or not pred_path.exists():
                print(f"[warn] {dataset} -> {method}: missing predictions ({pred_path}), skipping.")
                continue
            signature = build_signature(
                dataset,
                method,
                pred_path,
                cfg["gold"],
                cfg["genes"],
                cfg["tfs"],
                ks,
            )
            if not args.no_cache:
                cached = load_cached_metrics(CACHE_DIR, dataset, method, signature)
                if cached is not None:
                    dataset_results[method] = cached
                    continue
            preds = load_predictions(pred_path)
            if not preds:
                print(f"[warn] {dataset} -> {method}: no predictions parsed, skipping.")
                continue
            scores, labels = build_score_arrays(preds, gold_pairs, tf_order, genes)
            try:
                metrics = summarize_method(scores, labels, ks)
                dataset_results[method] = metrics
                if not args.no_cache:
                    save_cached_metrics(CACHE_DIR, dataset, method, signature, metrics)
            except ValueError as exc:
                print(f"[warn] {dataset} -> {method}: {exc}")
                continue
        if not dataset_results:
            print(f"[warn] Dataset {dataset}: no valid methods.")
            continue
        plot_dataset(dataset, dataset_results, args.output_dir, args.formats, ks, args.fixed_ylim)
        all_metrics[dataset] = dataset_results
        # release memory

    if all_metrics:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(all_metrics, indent=2))
        combined_path = args.combined_pdf or (args.output_dir / "tf_curves_combined.pdf")
        export_combined_pdf(args.output_dir, list(all_metrics.keys()), args.formats, combined_path)
    else:
        print("[warn] No metrics were generated.")


if __name__ == "__main__":
    main()
