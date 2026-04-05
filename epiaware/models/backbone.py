from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .attention import GenePeakCrossAttention, SparseGeneSelfAttention


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple op
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GenePeakBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_neighbors: int,
        topk_peaks: int,
        ffn_hidden: int,
        dropout: float,
        use_faiss_knn: bool,
        cross_chunk_size: int,
        self_chunk_size: int,
    ) -> None:
        super().__init__()
        self.self_attn = SparseGeneSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_neighbors=k_neighbors,
            dropout=dropout,
            use_faiss=use_faiss_knn,
            attn_chunk_size=self_chunk_size,
        )
        self.cross_attn = GenePeakCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            topk=topk_peaks,
            dropout=dropout,
            gene_chunk_size=cross_chunk_size,
        )
        self.gene_norm1 = nn.LayerNorm(embed_dim)
        self.gene_norm2 = nn.LayerNorm(embed_dim)
        self.gene_norm3 = nn.LayerNorm(embed_dim)
        self.peak_norm = nn.LayerNorm(embed_dim)
        self.gene_ffn = FeedForward(embed_dim, ffn_hidden, dropout)
        self.peak_ffn = FeedForward(embed_dim, ffn_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gene_states: torch.Tensor,
        peak_states: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        neighbor_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gsa = self.self_attn(gene_states, neighbor_idx=neighbor_idx)
        gene_states = self.gene_norm1(gene_states + self.dropout(gsa))
        cross = self.cross_attn(gene_states, peak_states, candidate_index, candidate_mask)
        gene_states = self.gene_norm2(gene_states + self.dropout(cross))
        gene_states = self.gene_norm3(gene_states + self.dropout(self.gene_ffn(gene_states)))
        peak_states = self.peak_norm(peak_states + self.dropout(self.peak_ffn(peak_states)))
        return gene_states, peak_states


class GeneOnlyBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_neighbors: int,
        ffn_hidden: int,
        dropout: float,
        use_faiss_knn: bool,
        self_chunk_size: int,
    ) -> None:
        super().__init__()
        self.self_attn = SparseGeneSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_neighbors=k_neighbors,
            dropout=dropout,
            use_faiss=use_faiss_knn,
            attn_chunk_size=self_chunk_size,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gene_states: torch.Tensor, neighbor_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        gsa = self.self_attn(gene_states, neighbor_idx=neighbor_idx)
        gene_states = self.norm1(gene_states + self.dropout(gsa))
        gene_states = self.norm2(gene_states + self.dropout(self.ffn(gene_states)))
        return gene_states


class MaskedNegativeBinomialHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        depth: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = embed_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    Swish(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.mean_proj = nn.Linear(hidden_dim, 1)
        self.disp_proj = nn.Linear(hidden_dim, 1)

    def forward(self, gene_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.mlp(gene_states)
        mean = torch.nn.functional.softplus(self.mean_proj(features).squeeze(-1)) + 1e-4
        dispersion = torch.nn.functional.softplus(
            self.disp_proj(features).squeeze(-1)
        ) + 1e-4
        return mean, dispersion


def negative_binomial_loss(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    log_prob = (
        torch.lgamma(target + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(target + 1.0)
        + dispersion * (torch.log(dispersion) - torch.log(dispersion + mean))
        + target * (torch.log(mean) - torch.log(dispersion + mean))
    )
    masked = -log_prob * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


class EpiAwareTransformer(nn.Module):
    """
    Backbone that fuses gene expression and chromatin accessibility signals.
    """

    def __init__(
        self,
        num_genes: int,
        num_peaks: int,
        candidate_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        k_neighbors: int = 12,
        topk_peaks: int = 8,
        ffn_hidden: int = 512,
        dropout: float = 0.1,
        use_faiss_knn: bool = False,
        cross_chunk_size: int = 1024,
        self_chunk_size: int = 1024,
        reuse_knn_indices: bool = False,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.num_peaks = num_peaks
        self.embed_dim = embed_dim
        self.gene_token_embed = nn.Embedding(num_genes, embed_dim)
        self.peak_token_embed = nn.Embedding(num_peaks, embed_dim)
        self.gene_value_proj = nn.Linear(1, embed_dim)
        self.peak_value_proj = nn.Linear(1, embed_dim)
        self.gene_type = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.peak_type = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "gene_index_buffer", torch.arange(num_genes, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "peak_index_buffer", torch.arange(num_peaks, dtype=torch.long), persistent=False
        )
        self.register_buffer("candidate_index", candidate_index, persistent=False)
        self.register_buffer("candidate_mask", candidate_mask, persistent=False)
        self.reuse_knn_indices = reuse_knn_indices
        self.blocks = nn.ModuleList(
            [
                GenePeakBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    k_neighbors=k_neighbors,
                    topk_peaks=topk_peaks,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                    use_faiss_knn=use_faiss_knn,
                    cross_chunk_size=cross_chunk_size,
                    self_chunk_size=self_chunk_size,
                )
                for _ in range(depth)
            ]
        )

    def encode_inputs(self, rna: torch.Tensor, atac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = rna.size(0)
        gene_values = self.gene_value_proj(rna.unsqueeze(-1))
        peak_values = self.peak_value_proj(atac.unsqueeze(-1))
        gene_ids = self.gene_token_embed(self.gene_index_buffer).unsqueeze(0).expand(batch, -1, -1)
        peak_ids = self.peak_token_embed(self.peak_index_buffer).unsqueeze(0).expand(batch, -1, -1)
        gene_states = gene_values + gene_ids + self.gene_type
        peak_states = peak_values + peak_ids + self.peak_type
        gene_states = self.dropout(gene_states)
        peak_states = self.dropout(peak_states)
        return gene_states, peak_states

    def forward(self, rna: torch.Tensor, atac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gene_states, peak_states = self.encode_inputs(rna, atac)
        shared_neighbor_idx: Optional[torch.Tensor] = None
        if self.reuse_knn_indices and (not self.training) and self.blocks:
            shared_neighbor_idx = self.blocks[0].self_attn._knn_indices(gene_states)
        for block in self.blocks:
            gene_states, peak_states = block(
                gene_states,
                peak_states,
                self.candidate_index,
                self.candidate_mask,
                neighbor_idx=shared_neighbor_idx,
            )
        return gene_states, peak_states


class GeneOnlyTransformer(nn.Module):
    """Backbone that uses only gene-gene self-attention (no ATAC / cross-attention).

    Intended for RNA-only ablations.
    """

    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        k_neighbors: int = 12,
        ffn_hidden: int = 512,
        dropout: float = 0.1,
        use_faiss_knn: bool = False,
        self_chunk_size: int = 1024,
        reuse_knn_indices: bool = False,
    ) -> None:
        super().__init__()
        self.num_genes = num_genes
        self.embed_dim = embed_dim
        self.gene_token_embed = nn.Embedding(num_genes, embed_dim)
        self.gene_value_proj = nn.Linear(1, embed_dim)
        self.gene_type = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "gene_index_buffer", torch.arange(num_genes, dtype=torch.long), persistent=False
        )
        self.reuse_knn_indices = reuse_knn_indices
        self.blocks = nn.ModuleList(
            [
                GeneOnlyBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    k_neighbors=k_neighbors,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                    use_faiss_knn=use_faiss_knn,
                    self_chunk_size=self_chunk_size,
                )
                for _ in range(depth)
            ]
        )

    def encode_inputs(self, rna: torch.Tensor) -> torch.Tensor:
        batch = rna.size(0)
        gene_values = self.gene_value_proj(rna.unsqueeze(-1))
        gene_ids = self.gene_token_embed(self.gene_index_buffer).unsqueeze(0).expand(batch, -1, -1)
        gene_states = gene_values + gene_ids + self.gene_type
        return self.dropout(gene_states)

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        gene_states = self.encode_inputs(rna)
        shared_neighbor_idx: Optional[torch.Tensor] = None
        if self.reuse_knn_indices and (not self.training) and self.blocks:
            shared_neighbor_idx = self.blocks[0].self_attn._knn_indices(gene_states)
        for block in self.blocks:
            gene_states = block(gene_states, neighbor_idx=shared_neighbor_idx)
        return gene_states


class EpiAwarePretrainingModel(nn.Module):
    def __init__(
        self,
        backbone: EpiAwareTransformer,
        recon_head: MaskedNegativeBinomialHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.recon_head = recon_head

    def forward(self, rna: torch.Tensor, atac: torch.Tensor) -> Dict[str, torch.Tensor]:
        gene_states, peak_states = self.backbone(rna, atac)
        mean, dispersion = self.recon_head(gene_states)
        return {
            "gene_states": gene_states,
            "peak_states": peak_states,
            "mean": mean,
            "dispersion": dispersion,
        }

    def compute_loss(
        self,
        rna: torch.Tensor,
        atac: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(rna, atac)
        loss = negative_binomial_loss(
            target=rna,
            mean=outputs["mean"],
            dispersion=outputs["dispersion"],
            mask=mask,
        )
        outputs["loss"] = loss
        return outputs


class GeneOnlyPretrainingModel(nn.Module):
    def __init__(
        self,
        backbone: GeneOnlyTransformer,
        recon_head: MaskedNegativeBinomialHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.recon_head = recon_head

    def forward(self, rna: torch.Tensor) -> Dict[str, torch.Tensor]:
        gene_states = self.backbone(rna)
        mean, dispersion = self.recon_head(gene_states)
        return {
            "gene_states": gene_states,
            "mean": mean,
            "dispersion": dispersion,
        }

    def compute_loss(
        self,
        rna: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(rna)
        loss = negative_binomial_loss(
            target=rna,
            mean=outputs["mean"],
            dispersion=outputs["dispersion"],
            mask=mask,
        )
        outputs["loss"] = loss
        return outputs
