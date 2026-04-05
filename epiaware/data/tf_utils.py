"""Utilities for working with transcription factor identifiers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence


def _add_alias(mapping: Dict[str, int], key: str, index: int) -> None:
    if key not in mapping:
        mapping[key] = index


def tf_gene_lookup(
    tf_names: Sequence[str],
    gene_names: Sequence[str],
    extra_prefixes: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[int]]:
    """Map TF identifiers to the matching gene index if present.

    The lookup tries exact matches first, then lower-cased, then stripped-prefix
    variants (handling tokens separated by ':'), and finally tries user-provided
    prefixes. A missing entry maps to ``None`` so callers can detect TFs that do
    not exist in the gene list.
    """

    gene_index = {name: idx for idx, name in enumerate(gene_names)}
    alias_to_index: Dict[str, int] = {}
    for name, idx in gene_index.items():
        _add_alias(alias_to_index, name, idx)
        _add_alias(alias_to_index, name.lower(), idx)
        if ":" in name:
            suffix = name.split(":", 1)[1]
            _add_alias(alias_to_index, suffix, idx)
            _add_alias(alias_to_index, suffix.lower(), idx)
    if extra_prefixes:
        for prefix in extra_prefixes:
            for name, idx in gene_index.items():
                if name.startswith(prefix):
                    trimmed = name[len(prefix) :]
                    if trimmed:
                        _add_alias(alias_to_index, trimmed, idx)
                        _add_alias(alias_to_index, trimmed.lower(), idx)

    result: Dict[str, Optional[int]] = {}
    for tf in tf_names:
        idx = alias_to_index.get(tf)
        if idx is None:
            idx = alias_to_index.get(tf.lower())
        if idx is None and ":" in tf:
            suffix = tf.split(":", 1)[1]
            idx = alias_to_index.get(suffix) or alias_to_index.get(suffix.lower())
        if idx is None and extra_prefixes:
            for prefix in extra_prefixes:
                if tf.startswith(prefix):
                    stripped = tf[len(prefix) :]
                    idx = alias_to_index.get(stripped) or alias_to_index.get(stripped.lower())
                    if idx is not None:
                        break
        result[tf] = idx
    return result
