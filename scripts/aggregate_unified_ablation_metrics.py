#!/usr/bin/env python3
"""
Aggregate unified-baseline evaluation outputs for ablation runs.

We scan for summary.json files produced by evaluate/unified_baseline_eval.py
and flatten them into a single combined_metrics_fullgrid.json / .tsv table.

Expected layout when run via scripts/run_ablation_metrics.sh:
  <DEST_ROOT>/<dataset>/<ablation>/summary.json

Each summary.json has the structure:
  {
    "<method>": {
      "full": {...metrics...},
      "gold": {...metrics...}
    },
    ...
  }

We emit one record per (dataset, ablation, method, split) pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def find_summary_files(root: Path) -> List[Path]:
    return list(root.rglob("summary.json"))


def flatten_summary(path: Path) -> List[Dict]:
    # path .../<dataset>/<ablation>/summary.json
    try:
        dataset = path.parent.parent.name
        ablation = path.parent.name
    except Exception:
        dataset = ""
        ablation = ""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    rows: List[Dict] = []
    for method, splits in data.items():
        if not isinstance(splits, dict):
            continue
        for split, metrics in splits.items():
            if not isinstance(metrics, dict):
                continue
            row = {
                "dataset": dataset,
                "ablation": ablation,
                "method": method,
                "split": split,
                "metrics_path": str(path),
            }
            for k, v in metrics.items():
                row[k] = v
            rows.append(row)
    return rows


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "combined_metrics_fullgrid.json").write_text(
        json.dumps(rows, indent=2)
    )
    # Simple TSV
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_dir / "combined_metrics_fullgrid.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate unified ablation metrics.")
    ap.add_argument(
        "--root",
        default="output/evaluate/ablations_fullgrid",
        help="Root directory that contains <dataset>/<ablation>/summary.json.",
    )
    args = ap.parse_args()
    root = Path(args.root)
    summaries = find_summary_files(root)
    rows: List[Dict] = []
    for path in summaries:
        rows.extend(flatten_summary(path))
    write_outputs(rows, root)
    print(f"[aggregate] summaries={len(summaries)} rows={len(rows)} root={root}")


if __name__ == "__main__":
    main()
