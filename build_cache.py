import os
import pickle
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- your repo utilities (already provided) ---
from config import DATA_PATHS, VAL_PATHS, TEST_PATHS, DEVICE
from data_loader import load_graphs, get_caption

# --- external deps (install on your machine) ---
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from transformers import AutoTokenizer, AutoModel

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


# ----------------------------
# Fingerprints (int bitset)
# ----------------------------
def _graph_to_smiles(g) -> Optional[str]:
    # Try common attribute names (depends on dataset)
    for key in ["smiles", "SMILES", "smile"]:
        if hasattr(g, key):
            s = getattr(g, key)
            if isinstance(s, str) and len(s) > 0:
                return s
    return None

def morgan_fp_int_from_smiles(smiles: str, radius: int = 2, nbits: int = 2048) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    # packed bits -> python int (fast for your tanimoto_similarity implementation)
    b = DataStructs.BitVectToBinaryText(fp)
    return int.from_bytes(b, byteorder="little", signed=False)

def compute_fps(graphs: List, radius: int = 2, nbits: int = 2048) -> List[int]:
    fps = []
    for g in graphs:
        smi = _graph_to_smiles(g)
        fps.append(morgan_fp_int_from_smiles(smi, radius, nbits) if smi else 0)
    return fps


# ----------------------------
# Encoders
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class GraphEncoderGINE(nn.Module):
    """
    Minimal, robust GINE encoder.
    Assumes PyG Data has x, edge_index, and optional edge_attr.
    """
    def __init__(self, x_dim: int, edge_dim: Optional[int], hidden: int = 256, out_dim: int = 256, layers: int = 3):
        super().__init__()
        self.x_proj = nn.Linear(x_dim, hidden)

        self.convs = nn.ModuleList()
        for _ in range(layers):
            nn_msg = MLP(hidden, hidden, hidden)
            conv = GINEConv(nn_msg, edge_dim=edge_dim) if edge_dim is not None else GINEConv(nn_msg)
            self.convs.append(conv)

        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)

        x = self.x_proj(x)
        for conv in self.convs:
            if edge_attr is not None:
                x = F.relu(conv(x, edge_index, edge_attr))
            else:
                x = F.relu(conv(x, edge_index))
        g = global_mean_pool(x, batch.batch)
        g = self.out(g)
        g = F.normalize(g, dim=-1)
        return g

class TextEncoderChemBERTa(nn.Module):
    def __init__(self, model_name: str, out_dim: int = 256, train_backbone: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        out = self.backbone(**enc)
        cls = out.last_hidden_state[:, 0, :]            # CLS pooling
        t = self.proj(cls)
        t = F.normalize(t, dim=-1)
        return t


# ----------------------------
# Symmetric InfoNCE (NT-Xent)
# ----------------------------
def symmetric_infonce(z: torch.Tensor, t: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    # z,t: [B,D] normalized
    logits = (z @ t.t()) / tau
    labels = torch.arange(z.size(0), device=z.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_g2t + loss_t2g)


# ----------------------------
# Embedding extraction
# ----------------------------
@torch.no_grad()
def encode_graphs(model_g: nn.Module, graphs: List, batch_size: int, device: torch.device) -> torch.Tensor:
    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    embs = []
    model_g.eval()
    for batch in loader:
        batch = batch.to(device)
        z = model_g(batch)
        embs.append(z.detach().cpu())
    return torch.cat(embs, dim=0)

@torch.no_grad()
def encode_texts(model_t: TextEncoderChemBERTa, texts: List[str], batch_size: int, device: torch.device) -> torch.Tensor:
    model_t.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        t = model_t(texts[i:i+batch_size], device=device)
        embs.append(t.detach().cpu())
    return torch.cat(embs, dim=0)


# ----------------------------
# Train loop
# ----------------------------
def train(
    model_g: nn.Module,
    model_t: TextEncoderChemBERTa,
    train_graphs: List,
    epochs: int,
    batch_size: int,
    lr: float,
    tau: float,
    device: torch.device,
):
    model_g.train()
    model_t.train()  # projection head may be trainable even if backbone frozen

    loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    params = list(model_g.parameters()) + [p for p in model_t.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            captions = [get_caption(g) for g in batch.to_data_list()]  # batch graphs -> captions

            z = model_g(batch)
            t = model_t(captions, device=device)

            loss = symmetric_infonce(z, t, tau=tau)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * z.size(0)
            n += z.size(0)

        print(f"epoch {ep:03d} | loss={total/max(1,n):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, default="cache_hybrid", help="where to write pool_graph_emb.pt etc.")
    ap.add_argument("--chemberta", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--train-text-backbone", action="store_true", help="fine-tune ChemBERTa backbone (slow)")
    ap.add_argument("--skip-train", action="store_true", help="only build cache from existing checkpoint")
    ap.add_argument("--ckpt", type=str, default="", help="path to graph encoder checkpoint (optional)")
    ap.add_argument("--fp-radius", type=int, default=2)
    ap.add_argument("--fp-nbits", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Load graphs
    train_graphs, val_graphs, test_graphs = load_graphs(DATA_PATHS, VAL_PATHS, TEST_PATHS)

    # Infer feature dims from first graph (assumes PyG Data)
    g0 = train_graphs[0]
    x_dim = int(g0.x.size(-1))
    edge_dim = int(g0.edge_attr.size(-1)) if hasattr(g0, "edge_attr") and g0.edge_attr is not None else None

    # Models
    model_g = GraphEncoderGINE(
        x_dim=x_dim,
        edge_dim=edge_dim,
        hidden=args.hidden,
        out_dim=args.out_dim,
        layers=args.layers,
    ).to(DEVICE)

    model_t = TextEncoderChemBERTa(
        model_name=args.chemberta,
        out_dim=args.out_dim,
        train_backbone=args.train_text_backbone,
    ).to(DEVICE)

    # Optionally load checkpoint
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model_g.load_state_dict(sd)
        print(f"Loaded graph encoder checkpoint: {args.ckpt}")

    # Train (unless skipped)
    if not args.skip_train:
        train(
            model_g=model_g,
            model_t=model_t,
            train_graphs=train_graphs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            tau=args.tau,
            device=DEVICE,
        )

    # Save checkpoint for reproducibility
    ckpt_path = os.path.join(args.cache_dir, "graph_encoder.pt")
    torch.save(model_g.state_dict(), ckpt_path)
    print(f"Saved graph encoder to: {ckpt_path}")

    # Build embeddings cache expected by your pipeline
    print("Encoding graphs to build cache...")
    train_emb = encode_graphs(model_g, train_graphs, args.embed_batch_size, DEVICE)
    val_emb   = encode_graphs(model_g, val_graphs,   args.embed_batch_size, DEVICE)
    test_emb  = encode_graphs(model_g, test_graphs,  args.embed_batch_size, DEVICE)

    pool_emb = torch.cat([train_emb, val_emb], dim=0)  # matches your pool_captions=train+val usage
    torch.save(pool_emb.float(), os.path.join(args.cache_dir, "pool_graph_emb.pt"))
    torch.save(val_emb.float(),  os.path.join(args.cache_dir, "val_graph_emb.pt"))
    torch.save(test_emb.float(), os.path.join(args.cache_dir, "test_graph_emb.pt"))

    print("Computing fingerprints (Morgan) ...")
    pool_fps = compute_fps(train_graphs + val_graphs, radius=args.fp_radius, nbits=args.fp_nbits)
    val_fps  = compute_fps(val_graphs, radius=args.fp_radius, nbits=args.fp_nbits)
    test_fps = compute_fps(test_graphs, radius=args.fp_radius, nbits=args.fp_nbits)
    with open(os.path.join(args.cache_dir, "fps.pkl"), "wb") as f:
        pickle.dump((pool_fps, val_fps, test_fps), f)

    # Optional: text embeddings for your proxy scoring in tuning.py
    print("Encoding captions (optional proxy cache) ...")
    pool_caps = [get_caption(g) for g in (train_graphs + val_graphs)]
    val_caps  = [get_caption(g) for g in val_graphs]
    train_txt_emb = encode_texts(model_t, pool_caps, batch_size=256, device=DEVICE)
    val_txt_emb   = encode_texts(model_t, val_caps,  batch_size=256, device=DEVICE)
    torch.save(train_txt_emb.float(), os.path.join(args.cache_dir, "train_txt_emb.pt"))
    torch.save(val_txt_emb.float(),   os.path.join(args.cache_dir, "val_txt_emb.pt"))

    print("\nCache built successfully. Your existing main.py can now run with --cache-dir", args.cache_dir)


if __name__ == "__main__":
    main()
