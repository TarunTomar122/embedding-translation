"""
Experiment 02b — CONNECTORS: different ways of bridging CLIP-image space
into MiniLM-text space, compared head-to-head on held-out anchors.

Connectors tested:
  A. orthogonal Procrustes  (rotation only)
  B. ridge regression       (free linear)
  C. 2-layer MLP            (nonlinear, cosine loss)
  D. MLP + InfoNCE loss     (contrastive bridge, learnable temperature)

Also: anchor-budget ablation — R@1 vs number of training pairs per connector.

Run: python connectors.py
"""
import math, random
import torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

HERE = Path(__file__).parent
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)

d = torch.load(HERE / "results" / "embeddings.pt")
S_img = F.normalize(d["clip"].float())
S_txt = F.normalize(d["mini"].float())
N = S_img.shape[0]
half = N // 2
idx = torch.randperm(N)
tr_idx, te_idx = idx[:half], idx[half:]
Xtr, Ytr = S_img[tr_idx].to(dev), S_txt[tr_idx].to(dev)
Xte, Yte = S_img[te_idx].to(dev), S_txt[te_idx].to(dev)


def eval_map(Qry, Doc):
    rec = Qry @ Doc.T
    order = rec.argsort(dim=1, descending=True)
    ranks = (order == torch.arange(rec.shape[0], device=rec.device)[:, None]).nonzero()[:, 1]
    return ((ranks < k).float().mean().item() for k in (1, 10))


def procrustes(X, Y):
    M = X.T @ Y
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return (U @ Vt)


def ridge(X, Y, lam=1.0):
    d_in = X.shape[1]
    return torch.linalg.solve(X.T @ X + lam * torch.eye(d_in, device=X.device), X.T @ Y)


class MLP(nn.Module):
    def __init__(self, din=512, dh=2048, dout=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dh), nn.GELU(), nn.Linear(dh, dout))
    def forward(self, x): return self.net(x)


def train_mlp(X, Y, loss_mode="cosine", epochs=300, bs=256, lr=1e-3):
    model = MLP().to(X.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = X.shape[0]
    temp = nn.Parameter(torch.tensor(0.07, device=X.device)).requires_grad_(loss_mode == "infonce")
    params = list(model.parameters()) + [temp]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        perm = torch.randperm(n, device=X.device)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            out = model(X[b])
            if loss_mode == "cosine":
                logits = F.normalize(out, dim=1) @ F.normalize(Y[b], dim=1).T
                loss = (1 - logits.diagonal()).mean()
            else:  # infonce
                logits = F.normalize(out, dim=1) @ F.normalize(Y[b], dim=1).T * temp.exp().clamp(max=100)
                loss = F.cross_entropy(logits, torch.arange(len(b), device=X.device))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run(Xtr, Ytr, Xte, Yte, budget=None):
    rows = []
    if budget:
        pick = torch.randperm(len(Xtr))[:budget]
        Xtr, Ytr = Xtr[pick], Ytr[pick]
    with torch.no_grad():
        Q = procrustes(Xtr, Ytr)
        r1, r10 = eval_map(F.normalize(Xte @ Q), Yte)
        rows.append(("procrustes", r1, r10))
        W = ridge(Xtr, Ytr)
        r1, r10 = eval_map(F.normalize(Xte @ W), Yte)
        rows.append(("ridge", r1, r10))
    m = train_mlp(Xtr, Ytr, "cosine")
    with torch.no_grad():
        r1, r10 = eval_map(F.normalize(m(Xte)), Yte)
    rows.append(("mlp cosine", r1, r10))
    m = train_mlp(Xtr, Ytr, "infonce")
    with torch.no_grad():
        r1, r10 = eval_map(F.normalize(m(Xte)), Yte)
    rows.append(("mlp infonce", r1, r10))
    return rows


print(f"=== full data ({len(tr_idx)} anchors) ===")
for name, r1, r10 in run(Xtr, Ytr, Xte, Yte):
    print(f"{name:<14} R@1={r1:.3f}  R@10={r10:.3f}")

print(f"\n=== anchor budget: how few pairs do you need to build the bridge? ===")
for budget in (16, 64, 256, 1024):
    print(f"--- {budget} anchors")
    for name, r1, r10 in run(Xtr, Ytr, Xte, Yte, budget=budget):
        print(f"{name:<14} R@1={r1:.3f}  R@10={r10:.3f}")
