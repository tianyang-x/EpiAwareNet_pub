import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_np_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        data = np.load(path, mmap_mode="r")
        if "arr_0" in data:
            return data["arr_0"]
        # Prefer explicit keys when available
        for key in ("rna", "atac", "matrix", "data"):
            if key in data:
                return data[key]
        raise ValueError(f"Unable to locate array payload in {path}")
    if path.suffix in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(tensor, torch.Tensor):
            return tensor.numpy()
        raise ValueError(
            f"Expected a tensor inside {path}, found {type(tensor).__name__}"
        )
    if path.suffix == ".h5ad":
        try:
            import anndata as ad
        except ImportError as err:  # pragma: no cover - optional dependency
            raise ImportError(
                "Reading .h5ad files requires the anndata package. "
                "Install with `pip install anndata` or provide a .npy/.npz file."
            ) from err
        adata = ad.read_h5ad(path)
        matrix = adata.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix)
    raise ValueError(f"Unsupported matrix format: {path}")


def load_feature_names(path: Optional[str], fallback_n: int, prefix: str) -> List[str]:
    if path is None:
        return [f"{prefix}_{i}" for i in range(fallback_n)]
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix in {".txt", ".tsv"}:
        with open(file_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    if file_path.suffix == ".json":
        data = json.loads(file_path.read_text())
        if isinstance(data, dict) and "names" in data:
            data = data["names"]
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of names in {file_path}")
        return [str(item) for item in data]
    raise ValueError(f"Unsupported feature-name format: {file_path}")


class CandidateSet:
    """
    Container that stores the candidate peak indices for every gene token.

    The input file can be:
      * JSON mapping gene name -> list of peak names or indices
      * JSON with {"candidates": [{"gene": "geneA", "peaks": [...]}, ...]}
      * TSV with columns gene<TAB>peak_idx, optionally multiple peak indices per row
    """

    def __init__(
        self,
        candidate_file: str,
        gene_names: Sequence[str],
        peak_names: Sequence[str],
        max_candidates: Optional[int] = None,
    ) -> None:
        self.gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
        self.peak_to_idx = {name: idx for idx, name in enumerate(peak_names)}
        self.num_genes = len(gene_names)
        self.num_peaks = len(peak_names)
        self.raw_map = self._load_candidate_map(candidate_file)
        self.max_candidates = (
            max_candidates if max_candidates is not None else self._infer_max_candidates()
        )
        self.index, self.mask = self._build_index_tensors()

    def _load_candidate_map(self, candidate_file: str) -> Dict[int, List[int]]:
        path = Path(candidate_file)
        if not path.exists():
            raise FileNotFoundError(candidate_file)
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and "candidates" in payload:
                payload = payload["candidates"]
            if isinstance(payload, list):
                mapping: Dict[int, List[int]] = {idx: [] for idx in range(self.num_genes)}
                for entry in payload:
                    gene_name = entry.get("gene") or entry.get("gene_id")
                    peaks = entry.get("peaks") or entry.get("peak_ids") or []
                    gene_idx = self._gene_index(gene_name)
                    mapping[gene_idx] = self._coerce_peak_indices(peaks)
                return mapping
            if isinstance(payload, dict):
                mapping = {idx: [] for idx in range(self.num_genes)}
                for gene, peaks in payload.items():
                    idx = self._gene_index(gene)
                    mapping[idx] = self._coerce_peak_indices(peaks)
                return mapping
            raise ValueError(f"Unsupported JSON format in {candidate_file}")
        # fallback: assume TSV
        mapping = {idx: [] for idx in range(self.num_genes)}
        with open(candidate_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split("\t")
                gene_token = tokens[0]
                if gene_token.lower() in {"gene", "gene_id", "gene_name"}:
                    continue
                peak_tokens = (
                    tokens[1].split(",") if len(tokens) > 1 else []
                )
                try:
                    idx = self._gene_index(gene_token)
                except KeyError:
                    # allow TSV headers or stray identifiers that are not part of the gene set
                    continue
                mapping[idx].extend(self._coerce_peak_indices(peak_tokens))
        return mapping

    def _gene_index(self, gene: Optional[str]) -> int:
        if gene is None:
            raise ValueError("Gene identifier missing in candidate file")
        if gene.isdigit():
            idx = int(gene)
            if idx >= self.num_genes:
                raise IndexError(f"Gene index {idx} exceeds gene set size {self.num_genes}")
            return idx
        if gene not in self.gene_to_idx:
            raise KeyError(f"Unknown gene {gene} referenced in candidate file")
        return self.gene_to_idx[gene]

    def _coerce_peak_indices(self, peaks: Sequence) -> List[int]:
        indices: List[int] = []
        for peak in peaks:
            if peak is None or peak == "":
                continue
            if isinstance(peak, (int, np.integer)):
                idx = int(peak)
            elif isinstance(peak, str) and peak.isdigit():
                idx = int(peak)
            else:
                peak_name = str(peak)
                if peak_name not in self.peak_to_idx:
                    raise KeyError(f"Unknown peak {peak_name} in candidate file")
                idx = self.peak_to_idx[peak_name]
            if idx >= self.num_peaks:
                raise IndexError(
                    f"Peak index {idx} exceeds number of peaks {self.num_peaks}"
                )
            indices.append(idx)
        return sorted(set(indices))

    def _infer_max_candidates(self) -> int:
        max_len = max(len(peaks) for peaks in self.raw_map.values())
        return max(1, max_len)

    def _build_index_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        index = torch.full(
            (self.num_genes, self.max_candidates), fill_value=-1, dtype=torch.long
        )
        mask = torch.zeros(
            (self.num_genes, self.max_candidates), dtype=torch.bool
        )
        for gene_idx, peaks in self.raw_map.items():
            capped = peaks[: self.max_candidates]
            if not capped:
                continue
            cand_len = len(capped)
            index[gene_idx, :cand_len] = torch.tensor(capped, dtype=torch.long)
            mask[gene_idx, :cand_len] = True
        return index, mask

    def to(self, device: torch.device) -> "CandidateSet":
        new_set = CandidateSet.__new__(CandidateSet)
        new_set.gene_to_idx = self.gene_to_idx
        new_set.peak_to_idx = self.peak_to_idx
        new_set.num_genes = self.num_genes
        new_set.num_peaks = self.num_peaks
        new_set.raw_map = self.raw_map
        new_set.max_candidates = self.max_candidates
        new_set.index = self.index.to(device)
        new_set.mask = self.mask.to(device)
        return new_set
    
    @staticmethod
    def generate_full_candidate_set(gene_names, peak_names, max_candidates=None) -> "CandidateSet":
        raw_map = {
            idx: list(range(len(peak_names))) for idx in range(len(gene_names))
        }
        candidate_set = CandidateSet.__new__(CandidateSet)
        candidate_set.gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
        candidate_set.peak_to_idx = {name: idx for idx, name in enumerate(peak_names)}
        candidate_set.num_genes = len(gene_names)
        candidate_set.num_peaks = len(peak_names)
        candidate_set.raw_map = raw_map
        candidate_set.max_candidates = (
            max_candidates if max_candidates is not None else len(peak_names)
        )
        candidate_set.index, candidate_set.mask = candidate_set._build_index_tensors()
        return candidate_set


class MultiOmicDataset(Dataset):
    """
    Lightweight dataset that keeps paired RNA/ATAC matrices in shared memory.
    """

    def __init__(
        self,
        rna_path: str,
        atac_path: str,
        gene_names_path: Optional[str] = None,
        peak_names_path: Optional[str] = None,
        cache_in_memory: bool = True,
        log_normalize: bool = True,
    ) -> None:
        rna_matrix = _load_np_matrix(Path(rna_path))
        atac_matrix = _load_np_matrix(Path(atac_path))
        if rna_matrix.shape[0] != atac_matrix.shape[0]:
            raise ValueError(
                f"RNA and ATAC matrices must have the same number of cells "
                f"({rna_matrix.shape[0]} vs {atac_matrix.shape[0]})"
            )
        self.num_cells, self.num_genes = rna_matrix.shape
        self.num_peaks = atac_matrix.shape[1]
        self.gene_names = load_feature_names(gene_names_path, self.num_genes, "gene")
        self.peak_names = load_feature_names(peak_names_path, self.num_peaks, "peak")
        if len(self.gene_names) != self.num_genes:
            raise ValueError(
                "Gene names list length does not match RNA matrix width "
                f"({len(self.gene_names)} vs {self.num_genes})"
            )
        if len(self.peak_names) != self.num_peaks:
            raise ValueError(
                "Peak names list length does not match ATAC matrix width "
                f"({len(self.peak_names)} vs {self.num_peaks})"
            )
        if log_normalize:
            rna_matrix = np.log1p(rna_matrix)
            atac_matrix = np.log1p(atac_matrix)
        rna_tensor = torch.from_numpy(np.asarray(rna_matrix, dtype=np.float32))
        atac_tensor = torch.from_numpy(np.asarray(atac_matrix, dtype=np.float32))
        if cache_in_memory:
            self.rna = rna_tensor.clone()
            self.atac = atac_tensor.clone()
        else:
            self.rna = rna_tensor
            self.atac = atac_tensor

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "rna": self.rna[idx],
            "atac": self.atac[idx],
        }


class RNADataset(Dataset):
    """Dataset that serves only RNA (no ATAC).

    This is intended for RNA-only / gene-to-gene self-attention ablations.
    It mirrors :class:`MultiOmicDataset`'s RNA loading and returns batches with
    a single key: ``{"rna": ...}``.
    """

    def __init__(
        self,
        rna_path: str,
        gene_names_path: Optional[str] = None,
        cache_in_memory: bool = True,
        log_normalize: bool = True,
    ) -> None:
        rna_matrix = _load_np_matrix(Path(rna_path))
        self.num_cells, self.num_genes = rna_matrix.shape
        self.gene_names = load_feature_names(gene_names_path, self.num_genes, "gene")
        if len(self.gene_names) != self.num_genes:
            raise ValueError(
                "Gene names list length does not match RNA matrix width "
                f"({len(self.gene_names)} vs {self.num_genes})"
            )
        if log_normalize:
            rna_matrix = np.log1p(rna_matrix)
        rna_tensor = torch.from_numpy(np.asarray(rna_matrix, dtype=np.float32))
        self.rna = rna_tensor.clone() if cache_in_memory else rna_tensor

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "rna": self.rna[idx],
        }
