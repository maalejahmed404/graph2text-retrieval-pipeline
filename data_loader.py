"""
Data loading utilities for the hybrid retrieval approach.
"""

import os
import pickle
from typing import List, Optional, Tuple

import torch


def find_first_existing(paths: List[str]) -> Optional[str]:
    """Find the first existing path from a list of candidates."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_pickle(path: str):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_graph_id(graph) -> str:
    """Extract ID from a graph object."""
    return str(getattr(graph, "id", ""))


def get_caption(graph) -> str:
    """Extract caption/description from a graph object."""
    return str(getattr(graph, "description", ""))


def load_graphs(train_paths: List[str], val_paths: List[str], test_paths: List[str]):
    """
    Load train, validation, and test graphs.
    
    Args:
        train_paths: Candidate paths for training data
        val_paths: Candidate paths for validation data
        test_paths: Candidate paths for test data
        
    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    train_path = find_first_existing(train_paths)
    val_path = find_first_existing(val_paths)
    test_path = find_first_existing(test_paths)
    
    assert train_path, f"Could not find train data in {train_paths}"
    assert val_path, f"Could not find val data in {val_paths}"
    assert test_path, f"Could not find test data in {test_paths}"
    
    train_graphs = load_pickle(train_path)
    val_graphs = load_pickle(val_path)
    test_graphs = load_pickle(test_path)
    
    print(f"Loaded: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    
    return train_graphs, val_graphs, test_graphs


def load_embeddings(cache_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load pre-computed graph embeddings.
    
    Args:
        cache_dir: Directory containing embedding files
        
    Returns:
        Tuple of (pool_emb, val_emb, test_emb)
    """
    pool_path = os.path.join(cache_dir, "pool_graph_emb.pt")
    val_path = os.path.join(cache_dir, "val_graph_emb.pt")
    test_path = os.path.join(cache_dir, "test_graph_emb.pt")
    
    assert os.path.exists(pool_path), f"Missing {pool_path}"
    assert os.path.exists(val_path), f"Missing {val_path}"
    assert os.path.exists(test_path), f"Missing {test_path}"
    
    pool_emb = torch.load(pool_path, map_location="cpu").float()
    val_emb = torch.load(val_path, map_location="cpu").float()
    test_emb = torch.load(test_path, map_location="cpu").float()
    
    # Normalize
    pool_emb = pool_emb / (pool_emb.norm(dim=-1, keepdim=True) + 1e-9)
    val_emb = val_emb / (val_emb.norm(dim=-1, keepdim=True) + 1e-9)
    test_emb = test_emb / (test_emb.norm(dim=-1, keepdim=True) + 1e-9)
    
    print(f"Loaded embeddings: pool={pool_emb.shape}, val={val_emb.shape}, test={test_emb.shape}")
    
    return pool_emb, val_emb, test_emb


def load_fingerprints(cache_dir: str) -> Tuple[list, list, list]:
    """
    Load pre-computed molecular fingerprints.
    
    Args:
        cache_dir: Directory containing fps.pkl
        
    Returns:
        Tuple of (pool_fps, val_fps, test_fps)
    """
    fps_path = os.path.join(cache_dir, "fps.pkl")
    assert os.path.exists(fps_path), f"Missing {fps_path}"
    
    with open(fps_path, "rb") as f:
        pool_fps, val_fps, test_fps = pickle.load(f)
    
    print(f"Loaded fingerprints: pool={len(pool_fps)}, val={len(val_fps)}, test={len(test_fps)}")
    
    return pool_fps, val_fps, test_fps


def load_text_embeddings(cache_dir: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load optional text embeddings for proxy scoring.
    
    Args:
        cache_dir: Directory containing text embedding files
        
    Returns:
        Tuple of (train_txt_emb, val_txt_emb) or (None, None) if not found
    """
    train_path = os.path.join(cache_dir, "train_txt_emb.pt")
    val_path = os.path.join(cache_dir, "val_txt_emb.pt")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Text embeddings not found, using weak proxy")
        return None, None
    
    train_emb = torch.load(train_path, map_location="cpu").float()
    val_emb = torch.load(val_path, map_location="cpu").float()
    
    # Normalize
    train_emb = train_emb / (train_emb.norm(dim=-1, keepdim=True) + 1e-9)
    val_emb = val_emb / (val_emb.norm(dim=-1, keepdim=True) + 1e-9)
    
    print("Loaded text embeddings for proxy scoring")
    
    return train_emb, val_emb
