import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute GRN evaluation metrics (AUPRC/AUROC) and plots for EpiAwareNet predictions."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction file (JSON or CSV/TSV). Must contain TF, target, and score columns.",
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Gold-standard link file (JSON or CSV/TSV) with TF and target columns.",
    )
    parser.add_argument(
        "--output_dir",
        default="output/epiaware_metrics",
        help="Directory to store metrics JSON and plots.",
    )
    parser.add_argument(
        "--tf_prefix",
        default="gene:",
        help="Prefix to strip from TF names (set to '' to disable).",
    )
    parser.add_argument(
        "--target_prefix",
        default="gene:",
        help="Prefix to strip from target names (set to '' to disable).",
    )
    parser.add_argument(
        "--min_predictions",
        type=int,
        default=10,
        help="Require at least this many prediction entries; otherwise raise an error.",
    )
    return parser.parse_args()


def _strip_prefix(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _normalise_entry(
    entry: Dict[str, str],
    tf_prefix: str,
    target_prefix: str,
) -> Optional[Tuple[str, str, float]]:
    lowered = {str(key).lower(): value for key, value in entry.items()}
    tf = (
        lowered.get("tf")
        or lowered.get("regulator")
        or lowered.get("gene_i")
        or lowered.get("source")
        or lowered.get("from")
    )
    target = (
        lowered.get("target")
        or lowered.get("gene_j")
        or lowered.get("gene")
        or lowered.get("sink")
        or lowered.get("to")
    )
    score = (
        lowered.get("score")
        or lowered.get("probability")
        or lowered.get("logit")
        or lowered.get("p")
        or lowered.get("value")
        or lowered.get("importance")
        or lowered.get("weight")
    )
    if tf is None or target is None:
        return None
    if score is None:
        return None
    tf_name = _strip_prefix(str(tf), tf_prefix)
    target_name = _strip_prefix(str(target), target_prefix)
    try:
        score_value = float(score)
    except ValueError:
        return None
    return tf_name, target_name, score_value


def load_prediction_file(
    path: Path, tf_prefix: str, target_prefix: str
) -> List[Tuple[str, str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    entries: List[Tuple[str, str, float]] = []
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            if "predicted_directed_links" in raw:
                iterable = raw["predicted_directed_links"]
            elif "predictions" in raw:
                iterable = raw["predictions"]
            else:
                iterable = raw.values()
        elif isinstance(raw, list):
            iterable = raw
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
        for item in iterable:
            if not isinstance(item, dict):
                continue
            norm = _normalise_entry(item, tf_prefix, target_prefix)
            if norm is not None:
                entries.append(norm)
        return entries

    # CSV/TSV loader
    delimiter = "," if path.suffix.lower().endswith("csv") else "\t"
    with open(path, "r", encoding="utf-8") as handle:
        sniff = handle.read(4096)
        handle.seek(0)
        if "\t" in sniff and path.suffix.lower().endswith("csv"):
            delimiter = "\t"
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            norm = _normalise_entry(row, tf_prefix, target_prefix)
            if norm is not None:
                entries.append(norm)
    return entries


def load_gold_file(path: Path, tf_prefix: str, target_prefix: str) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    positives: List[Tuple[str, str]] = []
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            if "links" in raw:
                iterable = raw["links"]
            else:
                iterable = raw.values()
        else:
            iterable = raw
        for item in iterable:
            if isinstance(item, dict):
                tf = (
                    item.get("tf")
                    or item.get("regulator")
                    or item.get("gene_i")
                    or item.get("source")
                )
                target = (
                    item.get("target")
                    or item.get("gene_j")
                    or item.get("gene")
                    or item.get("sink")
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                tf, target = item[:2]
            else:
                continue
            if tf is None or target is None:
                continue
            positives.append(
                (_strip_prefix(str(tf), tf_prefix), _strip_prefix(str(target), target_prefix))
            )
        return positives

    delimiter = "," if path.suffix.lower().endswith("csv") else "\t"
    with open(path, "r", encoding="utf-8") as handle:
        sniff = handle.read(4096)
        handle.seek(0)
        if "\t" in sniff and path.suffix.lower().endswith("csv"):
            delimiter = "\t"
        reader = csv.DictReader(handle, delimiter=delimiter)
        tf_col = None
        target_col = None
        for column in reader.fieldnames or []:
            lower = column.lower()
            if lower in {"tf", "transcription_factor", "regulator", "source"}:
                tf_col = column
            elif lower in {"target", "gene", "sink"}:
                target_col = column
        if tf_col is None or target_col is None:
            handle.seek(0)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split(delimiter)
                if len(tokens) < 2:
                    continue
                positives.append(
                    (
                        _strip_prefix(tokens[0], tf_prefix),
                        _strip_prefix(tokens[1], target_prefix),
                    )
                )
            return positives
        for row in reader:
            tf = row.get(tf_col)
            target = row.get(target_col)
            if not tf or not target:
                continue
            positives.append(
                (_strip_prefix(tf, tf_prefix), _strip_prefix(target, target_prefix))
            )
    return positives


def deduplicate_predictions(
    entries: Sequence[Tuple[str, str, float]]
) -> List[Tuple[str, str, float]]:
    pair_to_score: Dict[Tuple[str, str], float] = {}
    for tf, target, score in entries:
        key = (tf, target)
        if key not in pair_to_score or score > pair_to_score[key]:
            pair_to_score[key] = score
    deduped = [(tf, tgt, score) for (tf, tgt), score in pair_to_score.items()]
    deduped.sort(key=lambda x: x[2], reverse=True)
    return deduped


def compute_metrics(
    predictions: Sequence[Tuple[str, str, float]],
    gold_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, float]:
    gold_set = set(gold_pairs)
    scores = np.array([score for _, _, score in predictions], dtype=np.float32)
    y_true = np.array(
        [1 if (tf, tgt) in gold_set else 0 for tf, tgt, _ in predictions],
        dtype=np.int32,
    )
    metrics: Dict[str, float] = {}
    if y_true.sum() == 0:
        raise ValueError(
            "None of the prediction entries overlap with gold positives. "
            "Ensure TF/target identifiers use the same namespace."
        )
    metrics["positives_in_predictions"] = int(y_true.sum())
    metrics["total_predictions"] = int(len(predictions))
    metrics["positive_ratio"] = float(y_true.mean())
    metrics["auprc"] = float(average_precision_score(y_true, scores))
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["auroc"] = float("nan")
    precision, recall, _ = precision_recall_curve(y_true, scores)
    if math.isnan(metrics["auroc"]):
        fpr, tpr = None, None
    else:
        fpr, tpr, _ = roc_curve(y_true, scores)
    return metrics, precision, recall, fpr, tpr


def save_curves(
    output_dir: Path,
    precision: np.ndarray,
    recall: np.ndarray,
    fpr: Optional[np.ndarray],
    tpr: Optional[np.ndarray],
) -> None:
    plt.figure(figsize=(5, 4))
    plt.step(recall, precision, where="post", color="#2C7BB6")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.3)
    (output_dir / "pr_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=200)
    plt.close()

    if fpr is not None and tpr is not None:
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color="#D7191C")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close()


def save_metrics_json(metrics: Dict[str, float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    preds = load_prediction_file(Path(args.predictions), args.tf_prefix, args.target_prefix)
    if len(preds) < args.min_predictions:
        raise ValueError(
            f"Prediction file {args.predictions} only contained {len(preds)} pairs "
            f"(min required {args.min_predictions})."
        )
    preds = deduplicate_predictions(preds)
    gold_pairs = load_gold_file(Path(args.gold), args.tf_prefix, args.target_prefix)
    if not gold_pairs:
        raise ValueError(f"Gold file {args.gold} did not yield any positive pairs.")
    metrics, precision, recall, fpr, tpr = compute_metrics(preds, gold_pairs)
    save_metrics_json(metrics, output_dir)
    save_curves(output_dir, precision, recall, fpr, tpr)
    print(json.dumps(metrics, indent=2))
    print(f"[done] Saved metrics and plots to {output_dir}")


if __name__ == "__main__":
    main()
