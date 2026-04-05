#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


FIGURE_ROWS = [
    ("EpiAwareNet", "EpiAwareNet (baseline nnPU)"),
    ("RNA-only Epi", "RNA-only Epi"),
    ("Full stage 1", "Weak priors: full"),
    ("Prior stage 1", "Weak priors: prior"),
    ("Random stage 1", "Weak priors: random"),
    ("BCE", "BCE head"),
    ("cross-attention k=0", "cross-attention k=0"),
    ("cross-attention k=2", "cross-attention k=2"),
    ("cross-attention k=4", "cross-attention k=4"),
    ("cross-attention k=8", "cross-attention k=8"),
    ("cross-attention k=16", "cross-attention k=16"),
]

DATASET_COLS = [
    ("mN", "mN_ablations_"),
    ("pN", "pN_ablations_"),
    ("mouse brain", "mouse_brain_ablations_"),
    ("PBMC", "pbmc_ablations_"),
]

THRESHOLD_SLUGS = [
    "network_max_precision",
    "network_precision_0_2",
    "network_precision_0_25",
    "network_precision_0_3",
    "network_precision_0_35",
]

SLUG_TO_TABLE_NAME = {
    "network_max_precision": "max_precision",
    "network_precision_0_2": "precision0.2",
    "network_precision_0_25": "precision0.25",
    "network_precision_0_3": "precision0.3",
    "network_precision_0_35": "precision0.35",
}

THRESHOLD_FIELDS = [
    "precision",
    "recall",
    "f1",
    "edges",
]


def _dataset_from_root(dataset_root: str) -> Optional[str]:
    for label, prefix in DATASET_COLS:
        if dataset_root.startswith(prefix):
            return label
    return None


def _pick_one(df: pd.DataFrame) -> Optional[pd.Series]:
    """Pick one row when there are duplicates (nested output copies).

    Preference order:
    1) shortest run_rel
    2) if tie, highest auprc
    """
    if df.empty:
        return None
    df2 = df.copy()
    df2["_run_len"] = df2["run_rel"].astype(str).map(len)
    df2 = df2.sort_values(by=["_run_len", "auprc"], ascending=[True, False], kind="mergesort")
    return df2.iloc[0]


def _format_cell(auprc: Optional[float], auroc: Optional[float]) -> str:
    if auprc is None or pd.isna(auprc):
        return ""
    if auroc is None or pd.isna(auroc):
        return f"{float(auprc):.4g}"
    return f"{float(auprc):.4g} ({float(auroc):.4g})"


def _format_threshold_cell(row: pd.Series, slug: str) -> str:
    prec = row.get(f"{slug}__precision")
    rec = row.get(f"{slug}__recall")
    f1 = row.get(f"{slug}__f1")
    edges = row.get(f"{slug}__edges")

    if pd.isna(prec) or prec == "":
        return ""

    def fmt(x: object) -> str:
        try:
            return f"{float(x):.4g}"
        except Exception:
            return ""

    edges_str = ""
    try:
        if edges != "" and not pd.isna(edges):
            edges_str = str(int(float(edges)))
    except Exception:
        edges_str = str(edges) if edges not in (None, "") else ""

    parts = [
        f"P={fmt(prec)}",
        f"R={fmt(rec)}",
        f"F1={fmt(f1)}",
    ]
    if edges_str:
        parts.append(f"E={edges_str}")
    return " ".join(parts)


def _format_overall_and_threshold_cell(row: pd.Series, slug: str) -> str:
    # For a clipped network table, use AUPRC/AUROC computed on that clipped subset.
    overall = _format_cell(row.get(f"{slug}__auprc"), row.get(f"{slug}__auroc"))
    thresh = _format_threshold_cell(row, slug)
    if not thresh:
        return overall
    if not overall:
        return thresh
    return f"{overall} | {thresh}"


def build_table(metrics_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_tsv, sep="\t")

    # Normalize expected columns
    for col in ["dataset_root", "experiment", "auprc", "auroc", "run_rel"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {metrics_tsv}")

    df["dataset"] = df["dataset_root"].astype(str).map(_dataset_from_root)

    out: Dict[str, Dict[str, str]] = {}
    for row_label, exp_label in FIGURE_ROWS:
        out[row_label] = {col_label: "" for col_label, _ in DATASET_COLS}

        sub = df[(df["experiment"] == exp_label) & (df["dataset"].notna())]
        for ds_label, _prefix in DATASET_COLS:
            ds_sub = sub[sub["dataset"] == ds_label]
            picked = _pick_one(ds_sub)
            if picked is None:
                continue
            out[row_label][ds_label] = _format_cell(picked.get("auprc"), picked.get("auroc"))

    table = pd.DataFrame.from_dict(out, orient="index")
    table.index.name = "method"
    table = table.reset_index()
    return table


def build_overall_table_with_auprc_auroc(metrics_tsv: Path) -> pd.DataFrame:
    """Overall (non precision-cut) table matching the *_with_auprc_auroc schema.

    We keep the same columns as the precision-cut tables:
        method,
        <dataset>_AUPRC, <dataset>_AUROC, <dataset>_P, <dataset>_R, <dataset>_F1, <dataset>_E
    but for overall-only we leave P/R/F1/E blank.
    """

    df = pd.read_csv(metrics_tsv, sep="\t")

    # Normalize expected columns
    for col in ["dataset_root", "experiment", "auprc", "auroc", "run_rel"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {metrics_tsv}")

    df["dataset"] = df["dataset_root"].astype(str).map(_dataset_from_root)

    rows: list[dict[str, str]] = []
    for method_label, exp_label in FIGURE_ROWS:
        row: dict[str, str] = {"method": method_label}
        sub = df[(df["experiment"] == exp_label) & (df["dataset"].notna())]

        for ds_label, _prefix in DATASET_COLS:
            ds_sub = sub[sub["dataset"] == ds_label]
            picked = _pick_one(ds_sub)
            auprc = ""
            auroc = ""
            prec = ""
            rec = ""
            f1 = ""
            edges = ""
            if picked is not None:
                try:
                    v = picked.get("auprc")
                    if v is not None and not pd.isna(v):
                        auprc = f"{float(v):.6g}"
                except Exception:
                    auprc = ""
                try:
                    v = picked.get("auroc")
                    if v is not None and not pd.isna(v):
                        auroc = f"{float(v):.6g}"
                except Exception:
                    auroc = ""

                # Full-network metrics (no precision cut): computed on the entire exported GRN.
                try:
                    v = picked.get("precision")
                    if v not in (None, "") and not pd.isna(v):
                        prec = f"{float(v):.6g}"
                except Exception:
                    prec = ""
                try:
                    v = picked.get("recall")
                    if v not in (None, "") and not pd.isna(v):
                        rec = f"{float(v):.6g}"
                except Exception:
                    rec = ""
                try:
                    p = float(picked.get("precision"))
                    r = float(picked.get("recall"))
                    if (p + r) > 0:
                        f1 = f"{(2 * p * r / (p + r)):.6g}"
                except Exception:
                    f1 = ""
                try:
                    v = picked.get("total_predictions")
                    if v not in (None, "") and not pd.isna(v):
                        edges = str(int(float(v)))
                except Exception:
                    edges = ""

            row[f"{ds_label}_AUPRC"] = auprc
            row[f"{ds_label}_AUROC"] = auroc
            row[f"{ds_label}_P"] = prec
            row[f"{ds_label}_R"] = rec
            row[f"{ds_label}_F1"] = f1
            row[f"{ds_label}_E"] = edges

        rows.append(row)

    return pd.DataFrame(rows)


def build_table_with_submetrics(metrics_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_tsv, sep="\t")

    for col in ["dataset_root", "experiment", "auprc", "auroc", "run_rel"]:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {metrics_tsv}")

    df["dataset"] = df["dataset_root"].astype(str).map(_dataset_from_root)

    rows: list[dict[str, str]] = []
    for method_label, exp_label in FIGURE_ROWS:
        sub = df[(df["experiment"] == exp_label) & (df["dataset"].notna())]

        # overall row
        base_row: dict[str, str] = {"method": method_label, "submetric": "overall (AUPRC (AUROC))"}
        for ds_label, _prefix in DATASET_COLS:
            ds_sub = sub[sub["dataset"] == ds_label]
            picked = _pick_one(ds_sub)
            base_row[ds_label] = "" if picked is None else _format_cell(picked.get("auprc"), picked.get("auroc"))
        rows.append(base_row)

        # threshold rows
        for slug in THRESHOLD_SLUGS:
            sm_row: dict[str, str] = {"method": method_label, "submetric": slug}
            for ds_label, _prefix in DATASET_COLS:
                ds_sub = sub[sub["dataset"] == ds_label]
                picked = _pick_one(ds_sub)
                sm_row[ds_label] = "" if picked is None else _format_threshold_cell(picked, slug)
            rows.append(sm_row)

    out_df = pd.DataFrame(rows)
    return out_df


def build_threshold_indicator_table(metrics_tsv: Path, slug: str, field: str) -> pd.DataFrame:
    if field not in THRESHOLD_FIELDS:
        raise SystemExit(f"Unsupported field: {field}")

    df = pd.read_csv(metrics_tsv, sep="\t")
    df["dataset"] = df["dataset_root"].astype(str).map(_dataset_from_root)

    out: Dict[str, Dict[str, str]] = {}
    for row_label, exp_label in FIGURE_ROWS:
        out[row_label] = {col_label: "" for col_label, _ in DATASET_COLS}
        sub = df[(df["experiment"] == exp_label) & (df["dataset"].notna())]
        for ds_label, _prefix in DATASET_COLS:
            ds_sub = sub[sub["dataset"] == ds_label]
            picked = _pick_one(ds_sub)
            if picked is None:
                continue
            val = picked.get(f"{slug}__{field}")
            if val == "" or pd.isna(val):
                continue
            if field == "edges":
                try:
                    out[row_label][ds_label] = str(int(float(val)))
                except Exception:
                    out[row_label][ds_label] = str(val)
            else:
                out[row_label][ds_label] = f"{float(val):.4g}"

    table = pd.DataFrame.from_dict(out, orient="index")
    table.index.name = "method"
    table = table.reset_index()
    return table


def build_threshold_table(metrics_tsv: Path, slug: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_tsv, sep="\t")
    df["dataset"] = df["dataset_root"].astype(str).map(_dataset_from_root)

    out: Dict[str, Dict[str, str]] = {}
    for row_label, exp_label in FIGURE_ROWS:
        out[row_label] = {col_label: "" for col_label, _ in DATASET_COLS}
        sub = df[(df["experiment"] == exp_label) & (df["dataset"].notna())]
        for ds_label, _prefix in DATASET_COLS:
            ds_sub = sub[sub["dataset"] == ds_label]
            picked = _pick_one(ds_sub)
            if picked is None:
                continue
            out[row_label][ds_label] = _format_threshold_cell(picked, slug)

    table = pd.DataFrame.from_dict(out, orient="index")
    table.index.name = "method"
    table = table.reset_index()
    return table


def build_threshold_table_with_overall(metrics_tsv: Path, slug: str) -> pd.DataFrame:
    df = pd.read_csv(metrics_tsv, sep="\t")
    df["dataset"] = df["dataset_root"].astype(str).map(_dataset_from_root)

    out: Dict[str, Dict[str, str]] = {}
    for row_label, exp_label in FIGURE_ROWS:
        out[row_label] = {col_label: "" for col_label, _ in DATASET_COLS}
        sub = df[(df["experiment"] == exp_label) & (df["dataset"].notna())]
        for ds_label, _prefix in DATASET_COLS:
            ds_sub = sub[sub["dataset"] == ds_label]
            picked = _pick_one(ds_sub)
            if picked is None:
                continue
            out[row_label][ds_label] = _format_overall_and_threshold_cell(picked, slug)

    table = pd.DataFrame.from_dict(out, orient="index")
    table.index.name = "method"
    table = table.reset_index()
    return table


def build_threshold_table_with_overall_split(metrics_tsv: Path, slug: str) -> pd.DataFrame:
    """Split overall AUPRC/AUROC and clipped-network metrics into separate columns.

    Output columns:
        method,
        <dataset>_AUPRC, <dataset>_AUROC,
        <dataset>_P, <dataset>_R, <dataset>_F1, <dataset>_E
    for each dataset.
    """

    df = pd.read_csv(metrics_tsv, sep="\t")
    df["dataset"] = df["dataset_root"].astype(str).map(_dataset_from_root)

    rows: list[dict[str, str]] = []
    for method_label, exp_label in FIGURE_ROWS:
        row: dict[str, str] = {"method": method_label}
        sub = df[(df["experiment"] == exp_label) & (df["dataset"].notna())]

        for ds_label, _prefix in DATASET_COLS:
            ds_sub = sub[sub["dataset"] == ds_label]
            picked = _pick_one(ds_sub)
            auprc = ""
            auroc = ""
            prec = ""
            rec = ""
            f1 = ""
            edges = ""

            if picked is not None:
                try:
                    v = picked.get(f"{slug}__auprc")
                    if v is not None and not pd.isna(v):
                        auprc = f"{float(v):.6g}"
                except Exception:
                    auprc = ""
                try:
                    v = picked.get(f"{slug}__auroc")
                    if v is not None and not pd.isna(v):
                        auroc = f"{float(v):.6g}"
                except Exception:
                    auroc = ""

                # clipped-network metrics for this slug
                try:
                    v = picked.get(f"{slug}__precision")
                    if v not in (None, "") and not pd.isna(v):
                        prec = f"{float(v):.6g}"
                except Exception:
                    prec = ""
                try:
                    v = picked.get(f"{slug}__recall")
                    if v not in (None, "") and not pd.isna(v):
                        rec = f"{float(v):.6g}"
                except Exception:
                    rec = ""
                try:
                    v = picked.get(f"{slug}__f1")
                    if v not in (None, "") and not pd.isna(v):
                        f1 = f"{float(v):.6g}"
                except Exception:
                    f1 = ""
                try:
                    v = picked.get(f"{slug}__edges")
                    if v not in (None, "") and not pd.isna(v):
                        edges = str(int(float(v)))
                except Exception:
                    edges = ""

            row[f"{ds_label}_AUPRC"] = auprc
            row[f"{ds_label}_AUROC"] = auroc
            row[f"{ds_label}_P"] = prec
            row[f"{ds_label}_R"] = rec
            row[f"{ds_label}_F1"] = f1
            row[f"{ds_label}_E"] = edges

        rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a figure-style metrics table (rows/methods, cols/datasets) from metrics_summary_all.tsv"
    )
    p.add_argument(
        "--metrics_tsv",
        type=str,
        default="metrics_summary_all.tsv",
        help="Input TSV generated by scripts/summarize_all_custom_eval.py",
    )
    p.add_argument(
        "--out_tsv",
        type=str,
        default="metrics_table_for_figure.tsv",
        help="Output TSV written to repo root",
    )
    p.add_argument(
        "--out_overall_with_auprc_auroc_tsv",
        type=str,
        default="metrics_table_for_figure_overall_with_auprc_auroc.tsv",
        help=(
            "Output TSV for overall metrics, with the same column schema as *_with_auprc_auroc tables "
            "(AUPRC/AUROC plus blank P/R/F1/E)."
        ),
    )
    p.add_argument(
        "--out_submetrics_tsv",
        type=str,
        default="metrics_table_for_figure_all_submetrics.tsv",
        help="Output TSV with method+submetric rows (overall + clipped-network metrics).",
    )
    p.add_argument(
        "--with_submetrics",
        action="store_true",
        help="Also write a method+submetric table that includes clipped-network metrics.",
    )
    p.add_argument(
        "--write_overall_table_with_auprc_auroc",
        action="store_true",
        help="Write an overall-only (non precision-cut) table with separate AUPRC/AUROC columns per dataset.",
    )
    p.add_argument(
        "--write_precision03_tables",
        action="store_true",
        help="Also write separate tables for precision=0.30 clipped network (f1/edges/precision/recall).",
    )
    p.add_argument(
        "--write_threshold_tables",
        action="store_true",
        help=(
            "Write 5 separate figure-style tables for clipped networks: max_precision, "
            "precision0.2/0.25/0.3/0.35. Cells contain P/R/F1/E."
        ),
    )
    p.add_argument(
        "--write_threshold_tables_with_overall",
        action="store_true",
        help=(
            "Write 5 separate figure-style tables for clipped networks, but each cell contains "
            "overall AUPRC(AUROC) plus clipped-network P/R/F1/E (format: 'AUPRC (AUROC) | P=.. R=.. F1=.. E=..')."
        ),
    )
    p.add_argument(
        "--split_auprc_auroc",
        action="store_true",
        help="When writing *_with_auprc_auroc tables, split AUPRC and AUROC into separate columns per dataset.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics_tsv = Path(args.metrics_tsv)
    if not metrics_tsv.exists():
        raise SystemExit(f"Input not found: {metrics_tsv}")

    table = build_table(metrics_tsv)
    out_tsv = Path(args.out_tsv)
    table.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv} ({len(table)} rows)")

    if args.write_overall_table_with_auprc_auroc:
        t = build_overall_table_with_auprc_auroc(metrics_tsv)
        outp = Path(args.out_overall_with_auprc_auroc_tsv)
        t.to_csv(outp, sep="\t", index=False)
        print(f"Wrote {outp} ({len(t)} rows)")

    if args.with_submetrics:
        table2 = build_table_with_submetrics(metrics_tsv)
        out2 = Path(args.out_submetrics_tsv)
        table2.to_csv(out2, sep="\t", index=False)
        print(f"Wrote {out2} ({len(table2)} rows)")

    if args.write_precision03_tables:
        slug = "network_precision_0_3"
        for field in ["f1", "edges", "precision", "recall"]:
            t = build_threshold_indicator_table(metrics_tsv, slug=slug, field=field)
            outp = Path(f"metrics_table_for_figure_precision0.3_{field}.tsv")
            t.to_csv(outp, sep="\t", index=False)
            print(f"Wrote {outp} ({len(t)} rows)")

    if args.write_threshold_tables:
        for slug in THRESHOLD_SLUGS:
            name = SLUG_TO_TABLE_NAME.get(slug, slug)
            t = build_threshold_table(metrics_tsv, slug=slug)
            outp = Path(f"metrics_table_for_figure_{name}.tsv")
            t.to_csv(outp, sep="\t", index=False)
            print(f"Wrote {outp} ({len(t)} rows)")

    if args.write_threshold_tables_with_overall:
        for slug in THRESHOLD_SLUGS:
            name = SLUG_TO_TABLE_NAME.get(slug, slug)
            if args.split_auprc_auroc:
                t = build_threshold_table_with_overall_split(metrics_tsv, slug=slug)
            else:
                t = build_threshold_table_with_overall(metrics_tsv, slug=slug)
            outp = Path(f"metrics_table_for_figure_{name}_with_auprc_auroc.tsv")
            t.to_csv(outp, sep="\t", index=False)
            print(f"Wrote {outp} ({len(t)} rows)")


if __name__ == "__main__":
    main()
