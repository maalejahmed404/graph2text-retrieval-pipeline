"""
Molecular fingerprint operations and Tanimoto similarity.
"""

from typing import Union
import numpy as np


def tanimoto_similarity(a: Union[int, set, np.ndarray], b: Union[int, set, np.ndarray]) -> float:
    """
    Compute Tanimoto similarity between two fingerprints.
    
    Supports multiple fingerprint representations:
    - Python int bitset (fastest): uses bit_count
    - Python set of bit positions: uses set operations
    - NumPy uint64 array: fallback with bitwise operations
    
    Args:
        a: First fingerprint
        b: Second fingerprint
        
    Returns:
        Tanimoto similarity coefficient (0.0 to 1.0)
    """
    # Integer bitset (fastest)
    if isinstance(a, int) and isinstance(b, int):
        intersection = (a & b).bit_count()
        union = (a | b).bit_count()
        return 1.0 if union == 0 else intersection / union
    
    # Set of bit positions
    if isinstance(a, set) and isinstance(b, set):
        if (not a) and (not b):
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / max(1, union)
    
    # NumPy array fallback
    try:
        aa = np.asarray(a, dtype=np.uint64)
        bb = np.asarray(b, dtype=np.uint64)
        intersection = int(np.bitwise_and(aa, bb).view(np.uint64).sum())
        union = int(np.bitwise_or(aa, bb).view(np.uint64).sum())
        return 1.0 if union == 0 else intersection / union
    except Exception:
        return 0.0


def compute_fingerprint_scores(
    query_fp,
    candidate_indices: list,
    pool_fps: list,
) -> list:
    """
    Compute Tanimoto similarities for a set of candidates.
    
    Args:
        query_fp: Query molecule fingerprint
        candidate_indices: Indices into pool
        pool_fps: List of pool fingerprints
        
    Returns:
        List of Tanimoto scores corresponding to candidate indices
    """
    return [tanimoto_similarity(query_fp, pool_fps[j]) for j in candidate_indices]
