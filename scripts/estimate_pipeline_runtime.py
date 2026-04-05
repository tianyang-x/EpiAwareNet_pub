#!/usr/bin/env python3
"""Estimate stage1 + stage2 runtimes for ablation runs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def _read_runtime_seconds(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        with path.open() as handle:
            data = json.load(handle)
        return data.get("elapsed_sec")
    except (json.JSONDecodeError, OSError):
        return None


def _stage1_dir(run_dir: Path, key: str) -> Path:
    shared = run_dir / "shared"
    if key == "ablation_multiomics_rna_only__stage2":
        return shared / "stage1_rna_only"
    if key == "ablation_weak_priors__random":
        return shared / "stage1_candidates_random"
    if key.startswith("ablation_stage1_topk_routing__"):
        variant = key.split("__")[1]
        return shared / f"stage1_{variant}"
    return shared / "stage1_multiomic"


def _stage2_base(run_dir: Path, key: str) -> Path:
    if key == "shared__stage2_baseline_pu":
        return run_dir / "shared" / "stage2_baseline_pu"
    if key == "ablation_multiomics_rna_only__stage2":
        return run_dir / "ablation_multiomics_rna_only"
    if key.startswith("ablation_pu_learning__"):
        variant = key.split("__")[1]
        return run_dir / "ablation_pu_learning" / variant
    if key.startswith("ablation_stage1_topk_routing__"):
        variant = key.split("__")[1]
        return run_dir / "ablation_stage1_topk_routing" / variant
    if key.startswith("ablation_weak_priors__"):
        variant = key.split("__")[1]
        return run_dir / "ablation_weak_priors" / variant
    raise KeyError(f"Unhandled experiment key '{key}'")


def _stage2_runtime_path(base: Path) -> Optional[Path]:
    candidates = [base / "stage2" / "runtime.json", base / "runtime.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _seconds_to_hours(seconds: Optional[float]) -> Optional[float]:
    return None if seconds is None else seconds / 3600.0


def estimate_pipeline_runtimes(results_json: Path, repo_root: Path) -> Dict[str, Dict[str, dict]]:
    with results_json.open() as handle:
        combined = json.load(handle)
    report: Dict[str, Dict[str, dict]] = {}
    stage1_cache: Dict[Path, Optional[float]] = {}

    for dataset, experiments in combined.items():
        dataset_report: Dict[str, dict] = {}
        for exp_key, exp_data in experiments.items():
            run_id = exp_data.get("run")
            run_dir = repo_root / "output" / run_id
            entry = {"run": run_id}

            stage1_dir = _stage1_dir(run_dir, exp_key)
            if not stage1_dir.is_dir():
                fallback_dir = run_dir / "shared" / "stage1_multiomic"
                if fallback_dir.is_dir() and fallback_dir != stage1_dir:
                    print(
                        f"[warn] stage1 dir {stage1_dir} missing, falling back to {fallback_dir}",
                        file=sys.stderr,
                    )
                    stage1_dir = fallback_dir
            stage1_runtime_path = stage1_dir / "runtime.json"
            if stage1_runtime_path not in stage1_cache:
                stage1_cache[stage1_runtime_path] = _read_runtime_seconds(stage1_runtime_path)
                if stage1_cache[stage1_runtime_path] is None:
                    print(f"[warn] missing stage1 runtime for {stage1_runtime_path}", file=sys.stderr)
            stage1_seconds = stage1_cache.get(stage1_runtime_path)

            try:
                stage2_base = _stage2_base(run_dir, exp_key)
            except KeyError as exc:
                print(f"[warn] {exc}", file=sys.stderr)
                stage2_base = run_dir
            stage2_runtime_path = _stage2_runtime_path(stage2_base)
            if stage2_runtime_path is None:
                print(f"[warn] missing stage2 runtime under {stage2_base}", file=sys.stderr)
            stage2_seconds = _read_runtime_seconds(stage2_runtime_path) if stage2_runtime_path else None

            pipeline_seconds = None
            if stage1_seconds is not None or stage2_seconds is not None:
                pipeline_seconds = (stage1_seconds or 0.0) + (stage2_seconds or 0.0)

            entry.update(
                stage1_runtime_path=str(stage1_runtime_path),
                stage1_runtime_sec=stage1_seconds,
                stage1_runtime_hours=_seconds_to_hours(stage1_seconds),
                stage2_runtime_path=str(stage2_runtime_path) if stage2_runtime_path else None,
                stage2_runtime_sec=stage2_seconds,
                stage2_runtime_hours=_seconds_to_hours(stage2_seconds),
                pipeline_runtime_sec=pipeline_seconds,
                pipeline_runtime_hours=_seconds_to_hours(pipeline_seconds),
            )
            dataset_report[exp_key] = entry
        report[dataset] = dataset_report
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate pipeline runtimes from runtime.json files.")
    parser.add_argument(
        "--results-json",
        default="output/evaluate/ablations_fullgrid_3/combined_all_results.json",
        help="Path to combined_all_results.json",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the aggregated runtime report as JSON (defaults to stdout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]

    results_path = Path(args.results_json)
    if not results_path.is_absolute():
        results_path = (repo_root / results_path).resolve()
    if not results_path.is_file():
        print(f"[error] results JSON '{results_path}' not found", file=sys.stderr)
        sys.exit(1)

    report = estimate_pipeline_runtimes(results_path, repo_root)
    output_payload = json.dumps(report, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (repo_root / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_payload + "\n")
    else:
        print(output_payload)


if __name__ == "__main__":
    main()
