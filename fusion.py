"""
Reciprocal Rank Fusion and hybrid scoring.
"""

from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

from config import RRF_K, RRF_WEIGHT, TOPK_RERANK, DEFAULT_TEXT_RERANK_WEIGHT
from fingerprints import tanimoto_similarity


def reciprocal_rank_fusion(
    rank_a: List[int],
    rank_b: List[int],
    k: int = RRF_K,
) -> Dict[int, float]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.
    
    RRF score = sum of 1/(k + rank) for each ranking.
    
    Args:
        rank_a: First ranked list of indices
        rank_b: Second ranked list of indices
        k: RRF constant (default 60)
        
    Returns:
        Dictionary mapping index to fused score
    """
    scores = {}
    
    for rank, idx in enumerate(rank_a, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    
    for rank, idx in enumerate(rank_b, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    
    return scores


def hybrid_score_and_select(
    dense_values: list,
    dense_indices: list,
    query_fp,
    pool_fps: list,
    pool_captions: list,
    alpha: float = 0.7,
    topk_final: int = 1,
    rrf_k: int = RRF_K,
    rrf_weight: float = RRF_WEIGHT,
    # Text re-ranking parameters
    query_graph_emb: Optional[torch.Tensor] = None,
    pool_text_emb: Optional[torch.Tensor] = None,
    text_rerank_weight: float = DEFAULT_TEXT_RERANK_WEIGHT,
    topk_rerank: int = TOPK_RERANK,
) -> str:
    """
    Compute hybrid scores and select the best caption.
    
    Combines:
    - Dense cosine similarity (weight: alpha)
    - Fingerprint Tanimoto similarity (weight: 1-alpha)
    - RRF fusion score (weight: rrf_weight)
    - [Optional] Text re-ranking using graph-text cross-modal similarity
    
    Args:
        dense_values: Dense similarity scores
        dense_indices: Candidate indices from dense retrieval
        query_fp: Query molecule fingerprint
        pool_fps: Pool fingerprints
        pool_captions: Pool captions
        alpha: Weight for dense vs fingerprint
        topk_final: Number of top candidates to consider
        rrf_k: RRF constant
        rrf_weight: Weight for RRF score
        query_graph_emb: Query graph embedding for text re-ranking
        pool_text_emb: Pool text embeddings for text re-ranking
        text_rerank_weight: Weight for text similarity in re-ranking
        topk_rerank: Number of candidates to re-rank
        
    Returns:
        Best caption string
    """
    candidate_indices = list(map(int, dense_indices))
    cosine_scores = list(map(float, dense_values))
    
    # Compute fingerprint scores for candidates
    fp_scores = [tanimoto_similarity(query_fp, pool_fps[j]) for j in candidate_indices]
    
    # Create rankings
    dense_ranking = [j for _, j in sorted(zip(cosine_scores, candidate_indices), reverse=True)]
    fp_ranking = [j for _, j in sorted(zip(fp_scores, candidate_indices), reverse=True)]
    
    # RRF fusion
    rrf_scores = reciprocal_rank_fusion(dense_ranking, fp_ranking, k=rrf_k)
    
    # Compute final scores
    scored_candidates = []
    for cos, fp, idx in zip(cosine_scores, fp_scores, candidate_indices):
        final_score = alpha * cos + (1.0 - alpha) * fp + rrf_weight * rrf_scores.get(idx, 0.0)
        scored_candidates.append((final_score, idx))
    
    # Sort by score descending
    scored_candidates.sort(reverse=True, key=lambda x: x[0])
    
    # If text re-ranking is enabled (embeddings provided)
    if query_graph_emb is not None and pool_text_emb is not None:
        # Get top candidates for re-ranking
        top_candidates = scored_candidates[:topk_rerank]
        
        # Text re-ranking
        best_idx = text_rerank_candidates(
            query_graph_emb=query_graph_emb,
            candidate_indices=[idx for _, idx in top_candidates],
            rrf_scores={idx: score for score, idx in top_candidates},
            pool_text_emb=pool_text_emb,
            text_weight=text_rerank_weight,
        )
        return pool_captions[best_idx]
    
    # Without text re-ranking, use original selection
    top_candidates = scored_candidates[:max(1, topk_final)]
    best_idx = top_candidates[0][1]
    
    return pool_captions[best_idx]


def text_rerank_candidates(
    query_graph_emb: torch.Tensor,
    candidate_indices: List[int],
    rrf_scores: Dict[int, float],
    pool_text_emb: torch.Tensor,
    text_weight: float = 0.3,
) -> int:
    """
    Re-rank candidates using text-graph cross-modal similarity.
    
    The graph encoder and text encoder were trained together with contrastive loss,
    so query_graph_emb @ text_emb should give high similarity for matching pairs.
    
    Args:
        query_graph_emb: [D] query graph embedding (normalized)
        candidate_indices: List of candidate indices to re-rank
        rrf_scores: Dict mapping index to RRF-based score
        pool_text_emb: [N, D] pool text embeddings (normalized)
        text_weight: Weight for text similarity (vs RRF score)
        
    Returns:
        Index of best candidate after re-ranking
    """
    # Ensure tensor on CPU for compatibility
    if isinstance(query_graph_emb, torch.Tensor):
        query_emb = query_graph_emb.cpu().float()
    else:
        query_emb = torch.tensor(query_graph_emb, dtype=torch.float32)
    
    # Normalize query if needed
    query_emb = query_emb / (query_emb.norm() + 1e-9)
    
    # Get text embeddings for candidates
    candidate_text_embs = pool_text_emb[candidate_indices].float()  # [K, D]
    
    # Normalize text embeddings
    candidate_text_embs = candidate_text_embs / (candidate_text_embs.norm(dim=-1, keepdim=True) + 1e-9)
    
    # Compute cross-modal similarity: graph query vs text candidates
    text_sims = (query_emb @ candidate_text_embs.T).numpy()  # [K]
    
    # Normalize RRF scores to [0, 1] range for fair combination
    rrf_vals = np.array([rrf_scores[idx] for idx in candidate_indices])
    if rrf_vals.max() > rrf_vals.min():
        rrf_normalized = (rrf_vals - rrf_vals.min()) / (rrf_vals.max() - rrf_vals.min())
    else:
        rrf_normalized = np.ones_like(rrf_vals)
    
    # Normalize text similarities to [0, 1]
    if text_sims.max() > text_sims.min():
        text_normalized = (text_sims - text_sims.min()) / (text_sims.max() - text_sims.min())
    else:
        text_normalized = np.ones_like(text_sims)
    
    # Combine scores
    final_scores = (1 - text_weight) * rrf_normalized + text_weight * text_normalized
    
    # Return best candidate index
    best_pos = np.argmax(final_scores)
    return candidate_indices[best_pos]
