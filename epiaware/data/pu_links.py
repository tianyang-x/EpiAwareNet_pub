import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .multiomic_dataset import load_feature_names


def _ensure_prefix(value: str, prefix: str) -> str:
    if not prefix:
        return value
    if value.startswith(prefix):
        return value
    return f"{prefix}{value}"


def load_tf_list(path: str, fallback: Sequence[str], prefix: str = "") -> List[str]:
    if path is None:
        base = list(fallback)
    else:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(path)
        if file_path.suffix == ".json":
            data = json.loads(file_path.read_text())
            if isinstance(data, dict) and "tfs" in data:
                data = data["tfs"]
            if not isinstance(data, list):
                raise ValueError("TF list JSON must be a list or contain 'tfs'")
            base = [str(item) for item in data]
        else:
            with open(file_path, "r", encoding="utf-8") as handle:
                base = []
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    tokens = stripped.split()
                    base.append(tokens[0])
    return [_ensure_prefix(value, prefix) for value in base]


def load_positive_links(
    path: str,
    tf_prefix: str = "",
    target_prefix: str = "",
) -> List[Tuple[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    result: List[Tuple[str, str]] = []
    if file_path.suffix == ".json":
        payload = json.loads(file_path.read_text())
        if isinstance(payload, dict) and "links" in payload:
            payload = payload["links"]
        if not isinstance(payload, list):
            raise ValueError("Positive link JSON must be a list or contain 'links'")
        for item in payload:
            if isinstance(item, dict):
                tf = item.get("tf") or item.get("regulator")
                target = item.get("target") or item.get("gene")
            else:
                raise ValueError("JSON list items must be dicts with tf/target keys")
            if tf is None or target is None:
                continue
            result.append(
                (
                    _ensure_prefix(str(tf), tf_prefix),
                    _ensure_prefix(str(target), target_prefix),
                )
            )
        return result
    # Assume delimited text with headless columns tf,target
    with open(file_path, "r", encoding="utf-8") as handle:
        sample = handle.readline()
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames and {h.lower() for h in reader.fieldnames}.intersection(
            {"tf", "target", "gene", "regulator", "source", "sink"}
        ):
            tf_col = None
            target_col = None
            for column in reader.fieldnames:
                lower = column.lower()
                if lower in {"tf", "transcription_factor", "regulator", "source"}:
                    tf_col = column
                elif lower in {"target", "gene", "sink"}:
                    target_col = column
            if tf_col is None or target_col is None:
                raise ValueError(
                    f"Positive link file {path} must contain columns 'tf' and 'target'"
                )
            for row in reader:
                tf = row[tf_col]
                target = row[target_col]
                if not tf or not target:
                    continue
                result.append(
                    (
                        _ensure_prefix(tf, tf_prefix),
                        _ensure_prefix(target, target_prefix),
                    )
                )
        else:
            handle.seek(0)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split(delimiter)
                if len(tokens) < 2:
                    continue
                tf, target = tokens[0], tokens[1]
                result.append(
                    (
                        _ensure_prefix(tf, tf_prefix),
                        _ensure_prefix(target, target_prefix),
                    )
                )
    return result


class PositiveUnlabeledSampler:
    """
    Samples positive and unlabeled TF-gene pairs for nnPU training.
    """

    def __init__(
        self,
        tf_list: Sequence[str],
        gene_names: Sequence[str],
        positive_links: Sequence[Tuple[str, str]],
        seed: int = 42,
    ) -> None:
        self.tf_to_idx = {name: idx for idx, name in enumerate(tf_list)}
        self.gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
        tf_gene_indices: List[int] = []
        for tf in tf_list:
            if tf not in self.gene_to_idx:
                raise KeyError(f"Transcription factor {tf} not found among genes.")
            tf_gene_indices.append(self.gene_to_idx[tf])
        self.tf_indices = np.array(tf_gene_indices, dtype=np.int64)
        self.num_genes = len(gene_names)
        mapped_positive = []
        for tf, tgt in positive_links:
            if tf not in self.tf_to_idx or tgt not in self.gene_to_idx:
                continue
            mapped_positive.append((self.gene_to_idx[tf], self.gene_to_idx[tgt]))
        if not mapped_positive:
            raise ValueError("No positive links matched the provided gene list.")
        self.positive_pairs = np.array(mapped_positive, dtype=np.int64)
        self.positive_set = {
            (int(tf_idx), int(tgt_idx)) for tf_idx, tgt_idx in mapped_positive
        }
        self.rng = np.random.default_rng(seed)

    def sample_positive(self, batch_size: int) -> torch.LongTensor:
        choice = self.rng.choice(len(self.positive_pairs), size=batch_size, replace=True)
        return torch.from_numpy(self.positive_pairs[choice])

    def sample_unlabeled(self, batch_size: int) -> torch.LongTensor:
        samples: List[Tuple[int, int]] = []
        attempts = 0
        while len(samples) < batch_size:
            remaining = batch_size - len(samples)
            tf_choices = self.rng.choice(self.tf_indices, size=remaining * 2, replace=True)
            tgt_choices = self.rng.choice(self.num_genes, size=remaining * 2, replace=True)
            for tf_idx, tgt_idx in zip(tf_choices, tgt_choices):
                if (int(tf_idx), int(tgt_idx)) not in self.positive_set:
                    samples.append((int(tf_idx), int(tgt_idx)))
                    if len(samples) == batch_size:
                        break
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError("Unable to sample sufficient unlabeled pairs.")
        return torch.tensor(samples, dtype=torch.long)


__all__ = [
    "PositiveUnlabeledSampler",
    "load_positive_links",
    "load_tf_list",
    "load_feature_names",
]
