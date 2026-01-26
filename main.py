"""
Main entry point for the hybrid retrieval molecular captioning pipeline.

Usage:
    python main.py                           # Run with defaults
    python main.py --cache-dir cache_hybrid  # Custom cache directory
    python main.py --output submission.csv   # Custom output path
"""

import argparse

import pandas as pd
from tqdm.auto import tqdm

from config import (
    DATA_PATHS, VAL_PATHS, TEST_PATHS,
    CACHE_DIR, DEVICE,
    POOL_INCLUDE_VAL_FOR_TEST, VAL_TUNE_POOL_TRAIN_ONLY,
    RRF_K, TOPK_RERANK, DEFAULT_TEXT_RERANK_WEIGHT,
)
from data_loader import (
    load_graphs, get_caption, get_graph_id,
    load_embeddings, load_fingerprints, load_text_embeddings,
)
from retrieval import batch_topk_dense
from fusion import hybrid_score_and_select
from tuning import tune_hyperparameters


def main(cache_dir: str = CACHE_DIR, output_path: str = "submission33.csv"):
    """
    Run the complete hybrid retrieval pipeline.
    
    Args:
        cache_dir: Directory containing cached embeddings and fingerprints
        output_path: Path for output submission CSV
    """
    print("=" * 60)
    print("Hybrid Retrieval Molecular Captioning Pipeline")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    train_graphs, val_graphs, test_graphs = load_graphs(DATA_PATHS, VAL_PATHS, TEST_PATHS)
    
    train_captions = [get_caption(g) for g in train_graphs]
    val_captions = [get_caption(g) for g in val_graphs]
    test_ids = [get_graph_id(g) for g in test_graphs]
    
    # -------------------------------------------------------------------------
    # Load cached embeddings and fingerprints
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading embeddings and fingerprints...")
    pool_emb, val_emb, test_emb = load_embeddings(cache_dir)
    pool_fps, val_fps, test_fps = load_fingerprints(cache_dir)
    train_txt_emb, val_txt_emb = load_text_embeddings(cache_dir)
    
    # Pool (train + val) and counts
    train_count = len(train_graphs)
    val_count = len(val_graphs)
    pool_captions = train_captions + val_captions
    
    # -------------------------------------------------------------------------
    # Hyperparameter tuning
    # -------------------------------------------------------------------------
    print("\n[3/5] Tuning hyperparameters on validation set...")
    
    # For tuning, use train-only pool to avoid leakage
    if VAL_TUNE_POOL_TRAIN_ONLY:
        tune_pool_emb = pool_emb[:train_count]
        tune_pool_fps = pool_fps[:train_count]
        tune_pool_captions = train_captions
    else:
        tune_pool_emb = pool_emb
        tune_pool_fps = pool_fps
        tune_pool_captions = pool_captions
    
    best_config = tune_hyperparameters(
        val_emb=val_emb,
        val_fps=val_fps,
        val_captions=val_captions,
        pool_emb=tune_pool_emb,
        pool_fps=tune_pool_fps,
        pool_captions=tune_pool_captions,
        train_txt_emb=train_txt_emb,
        val_txt_emb=val_txt_emb,
    )
    
    # -------------------------------------------------------------------------
    # Test inference
    # -------------------------------------------------------------------------
    print("\n[4/5] Running inference on test set...")
    
    # For test, optionally include validation in pool
    if POOL_INCLUDE_VAL_FOR_TEST:
        final_pool_emb = pool_emb
        final_pool_fps = pool_fps
        final_pool_captions = pool_captions
        # Combine text embeddings to match pool size
        import torch
        final_pool_txt_emb = torch.cat([train_txt_emb, val_txt_emb], dim=0)
    else:
        final_pool_emb = pool_emb[:train_count]
        final_pool_fps = pool_fps[:train_count]
        final_pool_captions = train_captions
        final_pool_txt_emb = train_txt_emb
    
    # Dense retrieval
    topk_dense = best_config["topk_dense"]
    all_vals, all_idx = batch_topk_dense(
        test_emb, final_pool_emb, topk_dense,
        desc=f"TEST dense topk={topk_dense}",
    )
    
    # Get text rerank weight from tuning (or default)
    text_rerank_weight = best_config.get("text_rerank_weight", DEFAULT_TEXT_RERANK_WEIGHT)
    
    # Hybrid scoring with text re-ranking
    predictions = []
    for i in tqdm(range(len(test_emb)), desc="Hybrid + Text Re-ranking"):
        pred = hybrid_score_and_select(
            dense_values=all_vals[i],
            dense_indices=all_idx[i],
            query_fp=test_fps[i],
            pool_fps=final_pool_fps,
            pool_captions=final_pool_captions,
            alpha=best_config["alpha"],
            topk_final=best_config["topk_final"],
            rrf_k=RRF_K,
            # Text re-ranking parameters
            query_graph_emb=test_emb[i],
            pool_text_emb=final_pool_txt_emb,  # Pool text embeddings (matches pool size)
            text_rerank_weight=text_rerank_weight,
            topk_rerank=TOPK_RERANK,
        )
        predictions.append(pred)
    
    # -------------------------------------------------------------------------
    # Save submission
    # -------------------------------------------------------------------------
    print("\n[5/5] Saving submission...")
    
    submission = pd.DataFrame({
        "ID": test_ids,
        "description": predictions,
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\nSaved submission to {output_path}")
    print(submission.head())
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Retrieval Molecular Captioning Pipeline"
    )
    parser.add_argument(
        "--cache-dir",
        default=CACHE_DIR,
        help="Directory containing cached embeddings and fingerprints",
    )
    parser.add_argument(
        "--output",
        default="submission33.csv",
        help="Output path for submission file",
    )
    
    args = parser.parse_args()
    main(cache_dir=args.cache_dir, output_path=args.output)
