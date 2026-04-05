import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SparseGeneSelfAttention(nn.Module):
    """
    Gene-gene sparse self-attention with adaptive kNN masking.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_neighbors: int,
        dropout: float = 0.0,
        use_faiss: bool = False,
        knn_chunk_size: int = 512,
        attn_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_neighbors = k_neighbors
        self.use_faiss = use_faiss
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.knn_chunk_size = max(1, knn_chunk_size)
        self.attn_chunk_size = max(1, attn_chunk_size)
        if use_faiss:
            try:
                import faiss  # noqa: F401
            except ImportError as err:  # pragma: no cover - optional dependency
                raise ImportError(
                    "SparseGeneSelfAttention requires faiss when use_faiss=True"
                ) from err

    def _knn_indices(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, genes, dim)
        Returns:
            neighbor indices tensor of shape (batch, genes, k_neighbors)
        """
        batch, genes, dim = hidden_states.shape
        k = min(self.k_neighbors + 1, genes)
        if not self.use_faiss:
            normalized = F.normalize(hidden_states, p=2, dim=-1).detach()
            device = hidden_states.device
            chunk_size = self.knn_chunk_size
            neighbors_per_batch = []
            for b in range(batch):
                vecs = normalized[b]
                if device.type == "cuda":
                    vecs = vecs.float()
                vecs_t = vecs.transpose(0, 1).contiguous()
                indices_chunks = []
                for start in range(0, genes, chunk_size):
                    end = min(genes, start + chunk_size)
                    chunk = vecs[start:end]
                    sims = torch.matmul(chunk, vecs_t)
                    topk = sims.topk(k=k, dim=-1).indices
                    topk = topk[:, 1:] if k > 1 else topk
                    indices_chunks.append(topk)
                neighbors_per_batch.append(torch.cat(indices_chunks, dim=0))
            neighbors = torch.stack(neighbors_per_batch, dim=0).to(device)
        else:
            import faiss  # type: ignore

            hidden_np = hidden_states.detach().cpu().numpy().astype("float32")
            all_indices = []
            for b in range(batch):
                index = faiss.IndexFlatIP(dim)
                vectors = hidden_np[b]
                faiss.normalize_L2(vectors)
                index.add(vectors)
                distances, neigh = index.search(vectors, k)
                _ = distances  # unused
                neigh = neigh[:, 1:] if k > 1 else neigh
                all_indices.append(torch.from_numpy(neigh))
            neighbors = torch.stack(all_indices, dim=0).to(hidden_states.device)
        return neighbors

    def forward(
        self,
        hidden_states: torch.Tensor,
        neighbor_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, genes, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(batch, genes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, heads, genes, head_dim)
        k_all = (
            self.k_proj(hidden_states)
            .view(batch, genes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_all = (
            self.v_proj(hidden_states)
            .view(batch, genes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if neighbor_idx is None:
            neighbor_idx = self._knn_indices(hidden_states)
        diag = (
            torch.arange(genes, device=hidden_states.device)
            .view(1, genes, 1)
            .expand(batch, -1, 1)
        )
        neighbor_idx = torch.cat([neighbor_idx, diag], dim=-1)
        if neighbor_idx.size(-1) < self.k_neighbors:
            pad = self.k_neighbors - neighbor_idx.size(-1)
            if pad > 0:
                pad_idx = diag.expand(-1, -1, pad)
                neighbor_idx = torch.cat([neighbor_idx, pad_idx], dim=-1)
        neighbor_idx = neighbor_idx[..., : self.k_neighbors].clamp_(min=0, max=genes - 1)
        neighbor_idx = neighbor_idx.long()

        device = hidden_states.device
        chunk_size = getattr(self, "attn_chunk_size", 1024)
        num_neighbors = neighbor_idx.size(-1)
        batch_indexer = torch.arange(batch, device=device).view(batch, 1, 1, 1)
        head_indexer = torch.arange(self.num_heads, device=device).view(1, self.num_heads, 1, 1)
        contexts = []
        for start in range(0, genes, chunk_size):
            end = min(genes, start + chunk_size)
            chunk_len = end - start
            q_chunk = q[:, :, start:end, :]  # (batch, heads, chunk, head_dim)
            idx_chunk = neighbor_idx[:, start:end, :]  # (batch, chunk, k)
            idx_chunk = idx_chunk.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            batch_idx = batch_indexer.expand(batch, self.num_heads, chunk_len, num_neighbors)
            head_idx = head_indexer.expand(batch, self.num_heads, chunk_len, num_neighbors)
            k_chunk = k_all[batch_idx, head_idx, idx_chunk]
            v_chunk = v_all[batch_idx, head_idx, idx_chunk]
            attn_logits = torch.einsum(
                "bhqd,bhqkd->bhqk", q_chunk, k_chunk
            ) / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(attn_logits, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_chunk = torch.einsum("bhqk,bhqkd->bhqd", attn_weights, v_chunk)
            contexts.append(context_chunk)
        context = torch.cat(contexts, dim=2)
        context = context.transpose(1, 2).contiguous().view(batch, genes, self.embed_dim)
        return self.out_proj(context)


class GenePeakCrossAttention(nn.Module):
    """
    Masked cross-attention between genes and their candidate peaks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        topk: int,
        dropout: float = 0.0,
        gene_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.topk = topk
        self.gene_chunk_size = max(1, gene_chunk_size)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gene_states: torch.Tensor,
        peak_states: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gene_states: (batch, genes, dim)
            peak_states: (batch, peaks, dim)
            candidate_index: (genes, max_candidates)
            candidate_mask: (genes, max_candidates)
        """
        batch, genes, _ = gene_states.shape
        max_candidates = candidate_index.size(-1)
        index = candidate_index.unsqueeze(0).expand(batch, -1, -1)
        mask = candidate_mask.unsqueeze(0).expand(batch, -1, -1)

        q = (
            self.q_proj(gene_states)
            .view(batch, genes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, heads, genes, head_dim)
        k_all = (
            self.k_proj(peak_states)
            .view(batch, peak_states.size(1), self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, heads, peaks, head_dim)
        v_all = (
            self.v_proj(peak_states)
            .view(batch, peak_states.size(1), self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        chunk_size = self.gene_chunk_size
        num_neighbors = max_candidates
        device = gene_states.device
        batch_indexer = torch.arange(batch, device=device).view(batch, 1, 1, 1)
        head_indexer = torch.arange(self.num_heads, device=device).view(1, self.num_heads, 1, 1)
        contexts = []
        # If topk <= 0, treat it as "no routing" and attend over all candidates.
        # Otherwise use Top-K routing.
        use_topk = self.topk is not None and int(self.topk) > 0 and int(self.topk) < int(max_candidates)
        top_k = min(int(self.topk), int(max_candidates)) if use_topk else int(max_candidates)

        for start in range(0, genes, chunk_size):
            end = min(genes, start + chunk_size)
            chunk_len = end - start
            q_chunk = q[:, :, start:end, :]  # (batch, heads, chunk, head_dim)
            idx_chunk = index[:, start:end, :].clamp(
                min=0, max=peak_states.size(1) - 1
            )
            mask_chunk = mask[:, start:end, :]  # (batch, chunk, candidates)
            expanded_idx = idx_chunk.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            batch_idx = batch_indexer.expand(batch, self.num_heads, chunk_len, num_neighbors)
            head_idx = head_indexer.expand(batch, self.num_heads, chunk_len, num_neighbors)
            k_chunk = k_all[batch_idx, head_idx, expanded_idx]
            v_chunk = v_all[batch_idx, head_idx, expanded_idx]
            mask_chunk = mask_chunk.unsqueeze(1)  # (batch, 1, chunk, candidates)

            attn_logits = torch.einsum(
                "bhqd,bhqkd->bhqk", q_chunk, k_chunk
            ) / math.sqrt(self.head_dim)
            has_candidate = mask_chunk.any(dim=-1, keepdim=True)

            if use_topk:
                attn_logits = attn_logits.masked_fill(~mask_chunk, float("-inf"))
                top_scores, top_idx = attn_logits.topk(k=top_k, dim=-1)

                safe_scores = torch.where(
                    has_candidate.expand_as(top_scores),
                    top_scores,
                    torch.zeros_like(top_scores),
                )
                attn_weights = torch.softmax(safe_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                attn_weights = torch.where(
                    has_candidate.expand_as(attn_weights),
                    attn_weights,
                    torch.zeros_like(attn_weights),
                )
                safe_top_idx = torch.where(
                    has_candidate.expand_as(top_idx),
                    top_idx,
                    torch.zeros_like(top_idx),
                )
                gather_idx = safe_top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
                selected_v = torch.gather(v_chunk, dim=3, index=gather_idx)
                context_chunk = torch.einsum("bhqk,bhqkd->bhqd", attn_weights, selected_v)
            else:
                # Full (masked) attention over all candidate peaks.
                # Use a finite, dtype-aware mask value to avoid NaNs when a query has zero candidates.
                mask_fill_value = torch.finfo(attn_logits.dtype).min
                attn_logits = attn_logits.masked_fill(~mask_chunk, mask_fill_value)
                safe_logits = torch.where(
                    has_candidate.expand_as(attn_logits),
                    attn_logits,
                    torch.zeros_like(attn_logits),
                )
                attn_weights = torch.softmax(safe_logits, dim=-1)
                attn_weights = self.dropout(attn_weights)
                attn_weights = torch.where(
                    has_candidate.expand_as(attn_weights),
                    attn_weights,
                    torch.zeros_like(attn_weights),
                )
                context_chunk = torch.einsum("bhqk,bhqkd->bhqd", attn_weights, v_chunk)
            contexts.append(context_chunk)

        context = torch.cat(contexts, dim=2)
        context = context.transpose(1, 2).contiguous().view(batch, genes, self.embed_dim)
        return self.out_proj(context)
