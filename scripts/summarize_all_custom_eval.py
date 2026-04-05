#!/usr/bin/env python3

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class RunKey:
    dataset_root: str
    experiment: str
    auprc: Optional[float]
    auroc: Optional[float]
    positive_ratio: Optional[float]
    positives_in_predictions: Optional[int]
    total_predictions: Optional[int]


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _safe_load_json_list(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return None


def classify_experiment(run_dir: Path) -> str:
    p = run_dir.as_posix()
    if "/shared/stage2_baseline_pu" in p:
        return "EpiAwareNet (baseline nnPU)"
    if "/ablation_multiomics_rna_only/" in p:
        return "RNA-only Epi"
    if "/ablation_pu_learning/bce" in p:
        return "BCE head"
    if "/ablation_pu_learning/nnpu_export" in p:
        return "nnPU export"
    if "/ablation_stage1_topk_routing/" in p:
        m = re.search(r"topk_stage1_(\d+)", p)
        k = m.group(1) if m else "?"
        return f"cross-attention k={k}"
    if "/ablation_weak_priors/" in p:
        if "/prior" in p:
            return "Weak priors: prior"
        if "/random" in p:
            return "Weak priors: random"
        if "/full" in p:
            return "Weak priors: full"
        return "Weak priors: other"
    if "/stress_candidate_swap/" in p:
        if "/prior" in p:
            return "Stress swap: prior"
        if "/random" in p:
            return "Stress swap: random"
        if "/full" in p:
            return "Stress swap: full"
        return "Stress swap: other"
    return "Other"


def _dataset_root_from_metrics_path(metrics_path: Path) -> Optional[str]:
    parts = metrics_path.parts
    try:
        idx = parts.index("output")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def _run_rel_from_metrics_path(metrics_path: Path) -> Optional[str]:
    dataset_root = _dataset_root_from_metrics_path(metrics_path)
    if dataset_root is None:
        return None
    try:
        rel = metrics_path.relative_to(Path("output") / dataset_root)
    except Exception:
        return None
    # rel: <run_rel>/custom_eval/metrics.json
    if rel.parts[-2:] != ("custom_eval", "metrics.json"):
        return None
    run_rel = Path(*rel.parts[:-2]).as_posix()
    return run_rel


def _extract_slug_fields(prefix: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "name",
        "threshold",
        "curve_precision",
        "edges",
        "true_positives",
        "precision",
        "recall",
        "f1",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        out[f"{prefix}{k}"] = entry.get(k)
    return out


def _extract_cut_metrics_fields(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics computed on the clipped sub-network.

    These metrics are saved by custom_evaluate.py under custom_eval/<slug>/metrics.json.
    We flatten them into <slug>__<field> columns, alongside the threshold fields.
    """

    keys = [
        "auprc",
        "auroc",
        "positive_ratio",
        "positives_in_predictions",
        "total_predictions",
        "precision",
        "recall",
        "p_value",
        "auprc_baseline",
        "auprc_global_baseline",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        out[f"{prefix}{k}"] = metrics.get(k)
    return out


def build_table(output_root: Path) -> pd.DataFrame:
    metric_paths = list(output_root.rglob("custom_eval/metrics.json"))

    # First pass: collect records and compute union of threshold slugs.
    raw_records: List[Dict[str, Any]] = []
    all_slugs: set[str] = set()

    for mp in metric_paths:
        dataset_root = _dataset_root_from_metrics_path(mp)
        run_rel = _run_rel_from_metrics_path(mp)
        if dataset_root is None or run_rel is None:
            continue

        run_dir = Path("output") / dataset_root / run_rel
        custom_eval_dir = run_dir / "custom_eval"
        thresh_path = custom_eval_dir / "threshold_metrics.json"

        metrics = _safe_load_json(mp) or {}
        thresh_list = _safe_load_json_list(thresh_path) if thresh_path.exists() else None

        thresh_by_slug: Dict[str, Dict[str, Any]] = {}
        if thresh_list:
            for entry in thresh_list:
                slug = entry.get("slug")
                if isinstance(slug, str):
                    thresh_by_slug[slug] = entry
                    all_slugs.add(slug)

        # Second source of per-cut metrics: custom_eval/<slug>/metrics.json
        cut_metrics_by_slug: Dict[str, Dict[str, Any]] = {}
        if thresh_by_slug:
            for slug in thresh_by_slug.keys():
                cut_metrics_path = custom_eval_dir / slug / "metrics.json"
                if cut_metrics_path.exists():
                    cut_metrics = _safe_load_json(cut_metrics_path)
                    if isinstance(cut_metrics, dict):
                        cut_metrics_by_slug[slug] = cut_metrics

        record: Dict[str, Any] = {
            "dataset_root": dataset_root,
            "run_rel": run_rel,
            "experiment": classify_experiment(run_dir),
            "custom_eval_dir": custom_eval_dir.as_posix(),
            "metrics_path": mp.as_posix(),
            "threshold_metrics_path": thresh_path.as_posix() if thresh_path.exists() else "",
            "grn_csv_path": (run_dir / "grn.csv").as_posix() if (run_dir / "grn.csv").exists() else "",
            "grn_logits_path": (run_dir / "grn_logits.pt").as_posix() if (run_dir / "grn_logits.pt").exists() else "",
        }

        # Core metrics (flatten)
        for k in [
            "auprc",
            "auroc",
            "positive_ratio",
            "positives_in_predictions",
            "total_predictions",
            "precision",
            "recall",
            "p_value",
            "auprc_baseline",
            "auprc_global_baseline",
        ]:
            record[k] = metrics.get(k)

        # Keep threshold metrics for second pass to flatten with stable columns.
        record["_threshold_by_slug"] = thresh_by_slug
        record["_cut_metrics_by_slug"] = cut_metrics_by_slug
        raw_records.append(record)

    # Deduplicate nested output copies: keep shortest run_rel for identical metrics.
    dedup: Dict[RunKey, Dict[str, Any]] = {}
    for r in raw_records:
        key = RunKey(
            dataset_root=r.get("dataset_root", ""),
            experiment=r.get("experiment", ""),
            auprc=r.get("auprc"),
            auroc=r.get("auroc"),
            positive_ratio=r.get("positive_ratio"),
            positives_in_predictions=r.get("positives_in_predictions"),
            total_predictions=r.get("total_predictions"),
        )
        if key not in dedup or len(r.get("run_rel", "")) < len(dedup[key].get("run_rel", "")):
            dedup[key] = r

    records = list(dedup.values())

    # Second pass: flatten threshold metrics with consistent columns.
    slugs_sorted = sorted(all_slugs)
    flat_records: List[Dict[str, Any]] = []

    for r in records:
        thresh_by_slug: Dict[str, Dict[str, Any]] = r.pop("_threshold_by_slug", {})
        cut_metrics_by_slug: Dict[str, Dict[str, Any]] = r.pop("_cut_metrics_by_slug", {})
        out = dict(r)

        for slug in slugs_sorted:
            entry = thresh_by_slug.get(slug)
            prefix = f"{slug}__"
            if entry is None:
                # Fill empty columns for missing slug
                for k in [
                    "name",
                    "threshold",
                    "curve_precision",
                    "edges",
                    "true_positives",
                    "precision",
                    "recall",
                    "f1",
                ]:
                    out[f"{prefix}{k}"] = ""
                for k in [
                    "auprc",
                    "auroc",
                    "positive_ratio",
                    "positives_in_predictions",
                    "total_predictions",
                    "precision",
                    "recall",
                    "p_value",
                    "auprc_baseline",
                    "auprc_global_baseline",
                ]:
                    # Do not overwrite the empty threshold precision/recall fields above;
                    # these duplicates are fine and will remain empty.
                    out.setdefault(f"{prefix}{k}", "")
            else:
                out.update(_extract_slug_fields(prefix, entry))

            # Merge per-cut metrics computed on the clipped sub-network.
            cut_metrics = cut_metrics_by_slug.get(slug)
            if cut_metrics is None:
                for k in [
                    "auprc",
                    "auroc",
                    "positive_ratio",
                    "positives_in_predictions",
                    "total_predictions",
                    "precision",
                    "recall",
                    "p_value",
                    "auprc_baseline",
                    "auprc_global_baseline",
                ]:
                    out.setdefault(f"{prefix}{k}", "")
            else:
                out.update(_extract_cut_metrics_fields(prefix, cut_metrics))

        flat_records.append(out)

    df = pd.DataFrame(flat_records)

    # Stable ordering
    base_cols = [
        "dataset_root",
        "experiment",
        "run_rel",
        "custom_eval_dir",
        "metrics_path",
        "threshold_metrics_path",
        "grn_csv_path",
        "grn_logits_path",
        "auprc",
        "auroc",
        "positive_ratio",
        "positives_in_predictions",
        "total_predictions",
        "precision",
        "recall",
        "p_value",
        "auprc_baseline",
        "auprc_global_baseline",
    ]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + sorted(other_cols)]

    df = df.sort_values(by=["dataset_root", "experiment", "run_rel"], kind="mergesort")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan output/**/custom_eval metrics and produce one flattened TSV summary "
            "(including precision-target clipped network metrics)."
        )
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="output",
        help="Root directory to scan (default: output).",
    )
    p.add_argument(
        "--out_tsv",
        type=str,
        default="metrics_summary_all.tsv",
        help="Output TSV path (default: metrics_summary_all.tsv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise SystemExit(f"output_root does not exist: {output_root}")

    df = build_table(output_root)
    out_path = Path(args.out_tsv)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
