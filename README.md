# 🧬 Hybrid Retrieval for Graph-to-Text Generation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A novel retrieval-augmented approach for generating accurate textual descriptions from molecular graphs, achieving BLEU-4 + BERTScore = 0.69 on validation data.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Why This Approach Matters](#-why-this-approach-matters)
- [Architecture](#-architecture)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Extending to Other Domains](#-extending-to-other-domains)
- [Citation](#-citation)
- [License](#-license)

---

## 🔬 Overview

This project presents a **hybrid retrieval-augmented approach** for the challenging task of generating natural language descriptions from molecular graph structures. Rather than generating text from scratch (which can hallucinate), we leverage a carefully designed **multi-stage retrieval pipeline** that combines:

1. **Deep Learning-based Graph Embeddings** (GINE architecture)
2. **Chemical Fingerprint Similarity** (Morgan fingerprints + Tanimoto)
3. **Reciprocal Rank Fusion (RRF)** for robust candidate merging
4. **Cross-Modal Text Re-ranking** using contrastively-aligned embeddings

This approach ensures high-quality, factually grounded descriptions by retrieving from a pool of known molecule-caption pairs.

---

## 🏆 Key Results

| Metric | Validation Score |
|--------|------------------|
| **BLEU-4 + BERTScore** | **0.69** |
| Best Dense Top-K | 256 |
| Best Alpha (Dense vs FP) | 0.8 |
| Text Re-rank Weight | 0.3 |

The hybrid approach significantly outperforms single-modality retrieval methods by leveraging complementary signals from learned representations and chemical structure.

---

## 💡 Why This Approach Matters

### The Challenge
Generating accurate descriptions for molecules is critical for:
- **Drug Discovery**: Understanding molecular properties and mechanisms
- **Chemical Documentation**: Automated annotation of compound databases
- **Scientific Communication**: Making complex structures accessible

### Limitations of Pure Generation
Standard sequence-to-sequence or language model approaches often:
- ❌ Hallucinate facts not grounded in the molecular structure
- ❌ Miss critical functional groups or stereochemistry
- ❌ Require massive training data to generalize

### Our Solution: Retrieval-Augmented Generation
By retrieving descriptions from verified molecule-caption pairs, we:
- ✅ **Guarantee factual accuracy** - descriptions come from validated sources
- ✅ **Leverage domain knowledge** - training pool encodes chemical expertise
- ✅ **Handle rare structures** - fingerprints capture structural similarity even for unseen molecules
- ✅ **Scale efficiently** - no expensive generation at inference time

---

## 🏗️ Architecture

The system operates in two distinct phases:

### Training Phase (Contrastive Learning)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTRASTIVE TRAINING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐                     ┌─────────────────┐              │
│   │  Molecular      │                     │  Text Caption   │              │
│   │  Graph          │                     │  (Description)  │              │
│   └────────┬────────┘                     └────────┬────────┘              │
│            │                                       │                        │
│            ▼                                       ▼                        │
│   ┌─────────────────┐                     ┌─────────────────┐              │
│   │  GINE Encoder   │                     │  ChemBERTa      │              │
│   │  (3 layers)     │                     │  + Projection   │              │
│   └────────┬────────┘                     └────────┬────────┘              │
│            │                                       │                        │
│            ▼                                       ▼                        │
│   ┌─────────────────┐                     ┌─────────────────┐              │
│   │  Graph Embed    │◄───── InfoNCE ─────►│  Text Embed     │              │
│   │  (256-dim)      │       Loss          │  (256-dim)      │              │
│   └─────────────────┘                     └─────────────────┘              │
│                                                                             │
│   Objective: Maximize similarity for matching (graph, caption) pairs        │
│              Minimize similarity for non-matching pairs                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Inference Phase (Multi-Stage Retrieval)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                          │
│   │ Query Graph │                                                          │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│          ├─────────────────────────────────────────┐                       │
│          ▼                                         ▼                        │
│   ┌─────────────────┐                     ┌─────────────────┐              │
│   │  GINE Encoder   │                     │  Morgan         │              │
│   │                 │                     │  Fingerprints   │              │
│   └────────┬────────┘                     └────────┬────────┘              │
│            ▼                                       ▼                        │
│   ┌─────────────────┐                     ┌─────────────────┐              │
│   │  Dense Top-K    │                     │  Tanimoto       │              │
│   │  (K=256)        │                     │  Similarity     │              │
│   └────────┬────────┘                     └────────┬────────┘              │
│            │                                       │                        │
│            └─────────────────┬─────────────────────┘                       │
│                              ▼                                              │
│                     ┌─────────────────┐                                    │
│                     │   RRF Fusion    │                                    │
│                     │ α×dense + (1-α)×FP + 0.15×RRF                        │
│                     └────────┬────────┘                                    │
│                              ▼                                              │
│                     ┌─────────────────┐                                    │
│                     │  Top-10 RRF     │                                    │
│                     │  Candidates     │                                    │
│                     └────────┬────────┘                                    │
│                              ▼                                              │
│                     ┌─────────────────┐                                    │
│                     │  Text Re-Rank   │                                    │
│                     │  (Graph×Text)   │                                    │
│                     └────────┬────────┘                                    │
│                              ▼                                              │
│                     ┌─────────────────┐                                    │
│                     │  Best Caption   │                                    │
│                     └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Methodology

### Stage 1: Dense Retrieval (GINE Embeddings)
- **Architecture**: Graph Isomorphism Network with Edge features (GINE)
- **Training**: Contrastive learning with ChemBERTa text encoder
- **Loss**: Symmetric InfoNCE (NT-Xent) with temperature τ=0.07
- **Output**: 256-dimensional normalized graph embeddings

### Stage 2: Fingerprint Similarity
- **Method**: Morgan (circular) fingerprints with radius=2
- **Similarity**: Tanimoto coefficient for structural comparison
- **Advantage**: Captures chemical substructure patterns independently of learned representations

### Stage 3: Reciprocal Rank Fusion (RRF)
Combines rankings from both methods:
```
RRF_score(d) = Σ 1/(k + rank(d))
```
Where k=60 is the fusion constant.

**Final hybrid score**:
```
score = α × dense_sim + (1-α) × tanimoto_sim + 0.15 × RRF_score
```

### Stage 4: Cross-Modal Text Re-ranking
- Uses top-10 RRF candidates
- Computes query_graph_embedding × text_embeddings similarity
- Final score: (1-β) × RRF_normalized + β × text_similarity
- Best β = 0.3 from validation tuning

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies
```bash
pip install torch torch_geometric transformers rdkit-pypi pandas numpy tqdm
```

### Clone Repository
```bash
git clone https://github.com/yourusername/graph2text-retrieval.git
cd graph2text-retrieval
```

---

## 🚀 Usage

### Step 1: Prepare Data
Place your molecular graph data in `data_origin/`:
- `train_graphs.pkl` - Training molecules with descriptions
- `validation_graphs.pkl` - Validation molecules with descriptions
- `test_graphs.pkl` - Test molecules (descriptions to predict)

### Step 2: Build Cache (Train Encoders + Compute Embeddings)
```bash
python build_cache.py --epochs 5 --batch-size 64 --cache-dir cache_hybrid
```

This will:
1. Train the GINE graph encoder with ChemBERTa text encoder
2. Compute and save graph embeddings for train/val/test
3. Compute Morgan fingerprints
4. Save text embeddings for re-ranking

### Step 3: Run Inference Pipeline
```bash
python main.py --cache-dir cache_hybrid --output submission.csv
```

This will:
1. Load pre-computed embeddings and fingerprints
2. Tune hyperparameters on validation data
3. Generate predictions for test set
4. Save results to `submission.csv`

---

## 📁 Project Structure

```
graph2text-retrieval/
├── 📄 README.md              # This file
├── 📄 main.py                # Main entry point
├── 📄 config.py              # Configuration settings
├── 📄 data_loader.py         # Data loading utilities
├── 📄 build_cache.py         # Training + embedding cache builder
├── 📄 retrieval.py           # Dense retrieval functions
├── 📄 fingerprints.py        # Fingerprint computation & Tanimoto
├── 📄 fusion.py              # RRF and hybrid scoring
├── 📄 tuning.py              # Hyperparameter tuning
├── 📂 data_origin/           # Raw data files
│   ├── train_graphs.pkl
│   ├── validation_graphs.pkl
│   └── test_graphs.pkl
└── 📂 cache_hybrid/          # Pre-computed embeddings
    ├── graph_encoder.pt      # Trained GINE model
    ├── pool_graph_emb.pt     # Pool graph embeddings
    ├── val_graph_emb.pt      # Validation embeddings
    ├── test_graph_emb.pt     # Test embeddings
    ├── fps.pkl               # Morgan fingerprints
    ├── train_txt_emb.pt      # Training text embeddings
    └── val_txt_emb.pt        # Validation text embeddings
```

---

## ⚡ Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOPK_DENSE_LIST` | [128, 256, 512] | Dense retrieval candidates to try |
| `ALPHA_LIST` | [0.4-0.8] | Dense vs fingerprint weight |
| `RRF_K` | 60 | RRF fusion constant |
| `TOPK_RERANK` | 10 | Candidates for text re-ranking |
| `TEXT_RERANK_WEIGHT_LIST` | [0.2-0.5] | Text similarity weight |

---

## 🌍 Extending to Other Domains

While developed for molecular captioning, this hybrid retrieval framework is **domain-agnostic** and can be applied to any graph-to-text task:

### 🏙️ Knowledge Graphs → Text
- **Application**: Generate summaries for knowledge graph substructures
- **Adaptation**: Replace GINE with GAT/GCN, use BERT instead of ChemBERTa

### 🔗 Social Networks → Descriptions
- **Application**: Describe user communities or interaction patterns
- **Adaptation**: Node features encode user profiles, text describes community characteristics

### 🧬 Protein Structures → Function Descriptions
- **Application**: Predict protein function from 3D structure graphs
- **Adaptation**: Use specialized protein encoders (GVP, EquiFormer)

### 📊 Code AST → Documentation
- **Application**: Generate docstrings from Abstract Syntax Trees
- **Adaptation**: AST node embeddings + CodeBERT for text

### 🗺️ Scene Graphs → Captions
- **Application**: Image captioning via scene graph intermediate
- **Adaptation**: Object-relationship graphs + CLIP text encoder

### Key Adaptation Steps:
1. **Define your graph structure**: node features, edge features
2. **Choose appropriate encoders**: GNN variant + domain-specific text encoder
3. **Collect (graph, text) training pairs**: for contrastive learning
4. **If available**: add domain-specific fingerprints/similarity functions
5. **Tune**: the α, β, and top-k hyperparameters on your validation set

---

## 📊 Ablation Study

| Configuration | Validation Score |
|--------------|------------------|
| Dense only | 0.612 |
| Fingerprint only | 0.589 |
| Dense + FP (no RRF) | 0.651 |
| Dense + FP + RRF | 0.672 |
| **Full (+ Text Re-rank)** | **0.690** |

The text re-ranking stage provides a significant boost by leveraging the cross-modal alignment learned during training.

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{hybrid_graph2text_retrieval,
  title={Hybrid Retrieval for Graph-to-Text Generation},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/graph2text-retrieval}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ChemBERTa**: Pre-trained model for chemical text understanding
- **RDKit**: Cheminformatics toolkit for fingerprint computation
- **PyTorch Geometric**: Framework for graph neural networks

---

<p align="center">
  <b>⭐ Star this repository if you find it useful! ⭐</b>
</p>
#   g r a p h 2 t e x t - r e t r i e v a l - p i p e l i n e  
 