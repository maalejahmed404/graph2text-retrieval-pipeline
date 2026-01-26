"""
Dense retrieval functions using pre-computed embeddings.
"""

from typing import Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from config import DEVICE, RETRIEVAL_BATCH_SIZE


@torch.no_grad()
def topk_dense(
    query_emb: torch.Tensor,
    pool_emb: torch.Tensor,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k most similar items using dense embeddings.
    
    Args:
        query_emb: [B, D] query embeddings on device, normalized
        pool_emb: [N, D] pool embeddings on device, normalized
        topk: Number of results to return
        
    Returns:
        Tuple of (similarities, indices) as numpy arrays
    """
    sims = query_emb @ pool_emb.t()
    vals, idx = torch.topk(sims, k=topk, dim=-1)
    return vals.detach().cpu().numpy(), idx.detach().cpu().numpy()


def batch_topk_dense(
    query_emb: torch.Tensor,
    pool_emb: torch.Tensor,
    topk: int,
    batch_size: int = RETRIEVAL_BATCH_SIZE,
    device: torch.device = DEVICE,
    desc: str = "Dense retrieval",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch process dense retrieval for large query sets.
    
    Args:
        query_emb: [N, D] query embeddings on CPU
        pool_emb: [M, D] pool embeddings on CPU
        topk: Number of results per query
        batch_size: Processing batch size
        device: Device to use
        desc: Progress bar description
        
    Returns:
        Tuple of (all_similarities, all_indices) as numpy arrays
    """
    pool_emb_gpu = pool_emb.to(device)
    
    all_vals = []
    all_idx = []
    
    for i in tqdm(range(0, len(query_emb), batch_size), desc=desc):
        batch = query_emb[i:i + batch_size].to(device)
        vals, idx = topk_dense(batch, pool_emb_gpu, topk)
        all_vals.append(vals)
        all_idx.append(idx)
    
    return np.concatenate(all_vals, axis=0), np.concatenate(all_idx, axis=0)
