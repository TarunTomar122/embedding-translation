"""
Connector v3 — round 3:
  A. residual bridge: out = X@Q_procrustes + MLP(X), infonce-tuned (whitened spaces)
  B. reciprocal-rank fusion (RRF) of procrustes + MLP rank lists
  C. temperature/lr mini-sweep for residual bridge

Run: python connectors_v3.py [--tag v3a]
"""
import json, time, copy
import torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from connectors_v2 import Bridge, train_bridge, whiten, dev

HERE = Path(__file__).parent
torch.manual_seed(0)
d = torch.load(HERE / "results" / "embeddings_whitened.pt")
S_img, S_txt = F.normalize(d["clip"].float()), F.normalize(d["mini"].float())
N = S_img.shape[0]
idx = torch.randperm(N)
tr, te = idx[:N//2], idx[N//2:]
Xtr, Ytr = S_img[tr].to(dev), S_txt[tr].to(dev)
Xte, Yte = S_img[te].to(dev), S_txt[te].to(dev)

@torch.no_grad()
def procrustes(X, Y):
    M = X.T @ Y
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt

@torch.no_grad()
def eval_ranks(Qry, Doc, k_max=50):
    rec = Qry @ Doc.T
    order = rec.argsort(dim=1, descending=True)
    ranks = (order == torch.arange(rec.shape[0], device=rec.device)[:, None]).nonzero()[:, 1]
    out = {f"r@{k}": round((ranks < k).float().mean().item(), 4) for k in (1, 5, 10, 50)}
    out["mrr"] = round((1 / (ranks.float() + 1)).mean().item(), 4)
    return out


class ResBridge(nn.Module):
    """out = X @ Q + MLP(X); Q fixed from procrustes."""
    def __init__(self, din, dout, Q, w=2048, depth=3):
        super().__init__()
        self.register_buffer("Q", Q)
        self.mlp = Bridge(din, dout, w=w, depth=depth)
        nn.init.zeros_(self.mlp.net[0].weight)   # start as pure procrustes
        nn.init.zeros_(self.mlp.net[0].bias)
        self.mlp.net[-1].weight.data.zero_(); self.mlp.net[-1].bias.data.zero_()
    def forward(self, x): return x @ self.Q + self.mlp(x)


def train_res(X, Y, Q, w=2048, depth=3, epochs=500, bs=512, lr=1e-4, k_val=None):
    n = X.shape[0]
    if k_val is None: k_val = max(256, n // 10)
    perm = torch.randperm(n)
    Xv, Yv = X[perm[:k_val]], Y[perm[:k_val]]
    Xt, Yt = X[perm[k_val:]], Y[perm[k_val:]]
    model = ResBridge(X.shape[1], Y.shape[1], Q, w=w, depth=depth).to(dev)
    temp = nn.Parameter(torch.tensor(0.07, device=dev))
    opt = torch.optim.AdamW(list(model.mlp.parameters()) + [temp], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state = -9, None
    for ep in range(epochs):
        model.train()
        for i in range(0, len(Xt) - 1, bs):
            b = torch.randperm(len(Xt), device=dev)[:bs]
            out = model(Xt[b])
            logits = F.normalize(out, dim=1) @ F.normalize(Yt[b], dim=1).T * temp.exp().clamp(max=100)
            loss = F.cross_entropy(logits, torch.arange(len(b), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                val = (F.normalize(model(Xv)) @ F.normalize(Yv).T).diagonal().mean().item()
            if val > best_val:
                best_val, best_state = val, copy.deepcopy(model.mlp.state_dict())
    model.mlp.load_state_dict(best_state)
    model.eval()
    return model, best_val


results = {}
Q = procrustes(Xtr, Ytr)
t0 = time.time()
results["w-procrustes"] = eval_ranks(F.normalize(Xte @ Q), F.normalize(Yte))

model, val = train_res(Xtr, Ytr, Q)
r1 = eval_ranks(F.normalize(model(Xte)), F.normalize(Yte))
r1["val_cos"] = val
results["res-bridge"] = r1

# RRF fusion: procrustes ranks + trained plain bridge ranks
modelb, _ = train_bridge(Xtr, Ytr, w=2048, depth=3, epochs=400, bs=512)
with torch.no_grad():
    Sp = F.normalize(Xte @ Q) @ F.normalize(Yte).T
    Sb = F.normalize(modelb(Xte)) @ F.normalize(Yte).T
    Sp_r = Sp.argsort(1, descending=True).argsort(1).float()
    Sb_r = Sb.argsort(1, descending=True).argsort(1).float()
    for k in (10, 60):
        fused = 1/(k+1+Sp_r) + 1/(k+1+Sb_r)
        order = fused.argsort(1, descending=True)
        ranks = (order == torch.arange(fused.shape[0], device=fused.device)[:, None]).nonzero()[:, 1]
        results[f"rrf-k{k}"] = {**{f"r@{kk}": round((ranks < kk).float().mean().item(), 4) for kk in (1, 5, 10, 50)},
            "mrr": round((1/(ranks.float()+1)).mean().item(), 4)}

for name, r in results.items():
    print(f"{name:<16} {r}")
json.dump(results, open(HERE / "results" / "connectors_v3.json", "w"), indent=1)
print("saved")
