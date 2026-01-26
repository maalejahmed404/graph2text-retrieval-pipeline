"""
Configuration settings for the hybrid retrieval molecular captioning approach.
"""
# Best configuration: {'topk_dense': 256, 'topk_final': 1, 'alpha': 0.8, 'text_rerank_weight': 0.3}, score=0.95777
import torch

# =============================================================================
# Paths
# =============================================================================
# Directory containing train_graphs.pkl, validation_graphs.pkl, test_graphs.pkl
DATA_PATHS = [
    "data/train_graphs.pkl",
    "data_origin/train_graphs.pkl",
    "c:/Users/Lenovo/Desktop/graph2text retrieval/data_origin/train_graphs.pkl",
]

VAL_PATHS = [
    "data/validation_graphs.pkl",
    "data_origin/validation_graphs.pkl",
    "c:/Users/Lenovo/Desktop/graph2text retrieval/data_origin/validation_graphs.pkl",
]

TEST_PATHS = [
    "data/test_graphs.pkl",
    "data_origin/test_graphs.pkl",
    "c:/Users/Lenovo/Desktop/graph2text retrieval/data_origin/test_graphs.pkl",
]

# Cache directory for pre-computed embeddings
CACHE_DIR = "c:/Users/Lenovo/Desktop/graph2text retrieval/cache_hybrid"

# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Pool Configuration
# =============================================================================
# Whether to include validation set in the retrieval pool for test inference
POOL_INCLUDE_VAL_FOR_TEST = True

# Whether to use train-only pool for validation (avoids leakage)
VAL_TUNE_POOL_TRAIN_ONLY = True

# =============================================================================
# Hyperparameter Search Ranges
# =============================================================================
# Number of candidates after dense retrieval
TOPK_DENSE_LIST = [128, 256, 512]

# Number of final candidates to consider
TOPK_FINAL_LIST = [1, 3, 5]

# Weight for dense vs fingerprint scores
ALPHA_LIST = [0.40, 0.50, 0.60, 0.70, 0.80]

# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================
RRF_K = 60  # Constant for RRF scoring
RRF_WEIGHT = 0.15  # Weight for RRF score in final combination

# =============================================================================
# Text Re-ranking (Second Stage)
# =============================================================================
# Number of candidates to pass to text re-ranking after RRF fusion
TOPK_RERANK = 10

# Weight for text similarity in final scoring (vs RRF score)
TEXT_RERANK_WEIGHT_LIST = [0.2, 0.3, 0.4, 0.5]
DEFAULT_TEXT_RERANK_WEIGHT = 0.3

# =============================================================================
# Batch Sizes
# =============================================================================
RETRIEVAL_BATCH_SIZE = 256
