"""
LOOP — hypothesis-driven search for the strongest unsupervised translator
CLIP-image-space <-> MiniLM-text-space (first-captions), with round counter.

Fixed protocol:
  split: seed 0, 2500 train / 2500 eval anchors
  metric: exact R@1 of the eval row among 2500 captions (chance 0.0004)
  every round logged to results/loop.jsonl
"""
import json, torch, itertools
import torch.nn.functional as F
from pathlib import Path
from connectors_v2 import whiten as wh, dev

HERE = Path(__file__).parent
LOG = HERE / "results" / "loop.jsonl"

d = torch.load(HERE / "results" / "embeddings.pt")
catT = torch.load(HERE / "results" / "captions_all5.pt")
IMG_RAW = F.normalize(d["clip"].float()).cpu()
TXT_MEAN = F.normalize(catT.mean(1)).cpu()
torch.manual_seed(0)
N = IMG_RAW.shape[0]
idx = torch.randperm(N)
TR, TE = idx[:N//2], idx[N//2:]

round_counter = {"n": 0}

def ranks_of(Score, k_max=100):
    o = Score.argsort(1, descending=True)
    r = (o == torch.arange(Score.shape[0])[:, None]).nonzero()[:, 1]
    return {f"r@{k}": round((r < k).float().mean().item(), 4) for k in (1, 5, 10, 50)} | \
           {"mrr": round((1 / (r.float() + 1)).mean().item(), 4)}

def log_round(round_id, hypothesis, config, scores):
    round_counter["n"] += 1
    rec = {"round": round_counter["n"], "id": round_id, "hyp": hypothesis,
           "cfg": config, "scores": scores}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

LEADER = []

# ---------- estimators over (possibly whitened) coords ----------
def fit_map(Xtr, Ytr, mode="procrustes", lam=1.0, rank=None):
    if mode == "procrustes":
        M = Xtr.T @ Ytr
        U, _, Vt = torch.linalg.svd(M, full_matrices=False)
        return U @ Vt
    d_in = Xtr.shape[1]
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(d_in), Xtr.T @ Ytr)
    if rank:  # reduced-rank via SVD of predicted matrix product
        M = W @ (Xtr.T @ Xtr)  # approximation of W Mx virtual
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U[:, :rank] @ Vh[:rank, :]
    return W

def procrustes_square(Xtr, Ytr, W):
    M = (Xtr @ W).T @ Ytr
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return W @ U @ Vt

def evaluate(Q, Xtr, Ytr, Xte, Yte, name, hyp, sq=False):
    if sq:
        Xtr_n, Ytr_n = F.normalize(Xtr), F.normalize(Ytr)
        M = (Xtr_n @ Q).T @ Ytr_n
        U, _, Vt = torch.linalg.svd(M, full_matrices=False)
        Qf = Q @ U @ Vt
    else:
        Qf = Q
    return ranks_of(F.normalize(Xte @ Qf) @ F.normalize(Yte).T), Qf

def report(rec):
    LEADER.append((rec["scores"]["r@1"], rec["round"], rec["id"]))
