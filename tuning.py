"""
Hyperparameter tuning on validation set.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from config import (
    DEVICE, 
    TOPK_DENSE_LIST, TOPK_FINAL_LIST, ALPHA_LIST,
    RRF_K, TOPK_RERANK, TEXT_RERANK_WEIGHT_LIST,
)
from retrieval import batch_topk_dense
from fusion import hybrid_score_and_select


def cosine_proxy_score(
    predicted_captions: List[str],
    true_captions: List[str],
    pred_embeddings: Optional[torch.Tensor] = None,
    true_embeddings: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute proxy score for evaluation.
    
    If embeddings are provided, uses cosine similarity.
    Otherwise falls back to exact match rate.
    
    Args:
        predicted_captions: List of predicted caption strings
        true_captions: List of ground truth captions
        pred_embeddings: Optional embeddings for predictions
        true_embeddings: Optional embeddings for ground truth
        
    Returns:
        Proxy score (higher is better)
    """
    if pred_embeddings is not None and true_embeddings is not None:
        pred_emb = torch.as_tensor(pred_embeddings, dtype=torch.float32)
        true_emb = torch.as_tensor(true_embeddings, dtype=torch.float32)
        
        # Normalize
        pred_emb = pred_emb / (pred_emb.norm(dim=-1, keepdim=True) + 1e-9)
        true_emb = true_emb / (true_emb.norm(dim=-1, keepdim=True) + 1e-9)
        
        # Cosine similarity
        return float((pred_emb * true_emb).sum(-1).mean().item())
    
    # Fallback: exact match rate
    return float(np.mean([p.strip() == t.strip() for p, t in zip(predicted_captions, true_captions)]))


def tune_hyperparameters(
    val_emb: torch.Tensor,
    val_fps: list,
    val_captions: List[str],
    pool_emb: torch.Tensor,
    pool_fps: list,
    pool_captions: List[str],
    train_txt_emb: Optional[torch.Tensor] = None,
    val_txt_emb: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Grid search for optimal hyperparameters on validation set.
    
    Now includes text re-ranking weight in the search.
    
    Args:
        val_emb: Validation graph embeddings
        val_fps: Validation fingerprints
        val_captions: Validation ground truth captions
        pool_emb: Pool graph embeddings (train-only for tuning)
        pool_fps: Pool fingerprints
        pool_captions: Pool captions
        train_txt_emb: Optional training text embeddings for proxy and re-ranking
        val_txt_emb: Optional validation text embeddings for proxy
        
    Returns:
        Dictionary with best configuration
    """
    best_config = None
    best_score = -float("inf")
    
    # Build caption to embedding lookup
    cap_to_idx = {cap: i for i, cap in enumerate(pool_captions)}
    use_proxy = train_txt_emb is not None and val_txt_emb is not None
    
    print("\nTuning hyperparameters on validation set (with text re-ranking)...")
    
    for topk_dense in TOPK_DENSE_LIST:
        # Pre-compute dense retrieval for this topk
        all_vals, all_idx = batch_topk_dense(
            val_emb, pool_emb, topk_dense,
            desc=f"VAL dense topk={topk_dense}",
        )
        
        for topk_final in TOPK_FINAL_LIST:
            for alpha in ALPHA_LIST:
                for text_rerank_weight in TEXT_RERANK_WEIGHT_LIST:
                    # Generate predictions with text re-ranking
                    predictions = []
                    for i in range(len(val_emb)):
                        pred = hybrid_score_and_select(
                            dense_values=all_vals[i],
                            dense_indices=all_idx[i],
                            query_fp=val_fps[i],
                            pool_fps=pool_fps,
                            pool_captions=pool_captions,
                            alpha=alpha,
                            topk_final=topk_final,
                            rrf_k=RRF_K,
                            # Text re-ranking parameters
                            query_graph_emb=val_emb[i] if train_txt_emb is not None else None,
                            pool_text_emb=train_txt_emb,
                            text_rerank_weight=text_rerank_weight,
                            topk_rerank=TOPK_RERANK,
                        )
                        predictions.append(pred)
                    
                    # Compute proxy score
                    if use_proxy:
                        # Look up embeddings for predictions
                        pred_emb = []
                        for p in predictions:
                            idx = cap_to_idx.get(p)
                            if idx is not None:
                                pred_emb.append(train_txt_emb[idx].cpu())
                            else:
                                pred_emb.append(torch.zeros(train_txt_emb.size(1)))
                        pred_emb = torch.stack(pred_emb, dim=0)
                        
                        score = cosine_proxy_score(
                            predictions, val_captions,
                            pred_emb, val_txt_emb,
                        )
                    else:
                        score = cosine_proxy_score(predictions, val_captions)
                    
                    if score > best_score:
                        best_score = score
                        best_config = {
                            "topk_dense": topk_dense,
                            "topk_final": topk_final,
                            "alpha": alpha,
                            "text_rerank_weight": text_rerank_weight,
                        }
                        print(f"  NEW BEST: score={score:.5f} config={best_config}")
    
    print(f"\nBest configuration: {best_config}, score={best_score:.5f}")
    return best_config
