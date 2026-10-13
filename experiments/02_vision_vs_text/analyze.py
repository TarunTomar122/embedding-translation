"""
Experiment 02 — analysis: vision encoder vs text embedder alignment.

Tests:
  1. raw agreement (CKA, centroid structure)
  2. cross-modal retrieval CLIP->MiniLM via orthogonal Procrustes
     (fit on disjoint train half, eval on test half)
  3. baselines: paired-shuffle (chance), CLIP->CLIP (ceiling)

Run: python analyze.py
"""
import torch
import torch.nn.functional as F
from pathlib import Path

HERE = Path(__file__).parent
d = torch.load(HERE / "results" / "embeddings.pt")
S_img = F.normalize(d["clip"].float())
S_txt = F.normalize(d["mini"].float())
N = S_img.shape[0]


def cka(X, Y, kernel="linear"):
    if kernel == "linear":
        K, L = X @ X.T, Y @ Y.T
    else:
        d2 = (X**2).sum(1)[:, None] + (X**2).sum(1)[None] - 2 * X @ X.T
        K = torch.exp(-d2 / d2.median())
        d2 = (Y**2).sum(1)[:, None] + (Y**2).sum(1)[None] - 2 * Y @ Y.T
        L = torch.exp(-d2 / d2.median())
    K = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    L = L - L.mean(0, keepdim=True) - L.mean(1, keepdim=True) + L.mean()
    return ((K * L).sum() / (K.norm() * L.norm())).item()


def procrustes(X, Y):
    M = X.T @ Y
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt


def recall_ranks(Qry, Doc):
    # Qry [n, dQ] -> Doc [n, dD], row i must retrieve row i
    rec = Qry @ Doc.T
    order = rec.argsort(dim=1, descending=True)
    return (order == torch.arange(rec.shape[0])[:, None]).nonzero()[:, 1]


def report(name, ranks):
    n = len(ranks)
    r = {f"R@{k}": f"{(ranks < k).float().mean():.3f}" for k in (1, 5, 10, 100)}
    print(f"{name:<38} {r}  MRR={float((1/(ranks.float()+1)).mean()):.3f}")


half = N // 2
idx = torch.randperm(N)
tr, te = idx[:half], idx[half:]

print(f"n={N}  ({len(tr)} train / {len(te)} test anchors)\n")
print("--- raw agreement (no projection) ---")
print(f"linear CKA:            {cka(S_img, S_txt):.4f}")
print(f"rbf CKA:               {cka(S_img, S_txt, 'rbf'):.4f}")

def centroid_kernel(X):
    # images: cluster structure from 10k-means-ish -> use text-side clusters?
    return X @ X.T  # placeholder features

print()
# --- cross-modal map: CLIP image -> MiniLM text ---
Q = procrustes(S_img[tr], S_txt[tr])
report("procrustes img->text", recall_ranks(F.normalize(S_img[te] @ Q), S_txt[te]))
Q2 = procrustes(S_txt[tr], S_img[tr])
report("procrustes text->img", recall_ranks(F.normalize(S_txt[te] @ Q2), S_img[te]))

# --- chance baseline: shuffled pairings fitted + evaluated ---
perm = torch.randperm(len(tr))
Q_ch = procrustes(S_img[tr][perm], S_txt[tr])
report("procrustes img->text (SHUFFLED fit)", recall_ranks(F.normalize(S_img[te] @ Q_ch), S_txt[te]))

# --- ceiling: same-model alignment on disjoint halves ---
Qc = procrustes(S_img[tr], S_img[tr])
report("procrustes img->img (CEILING)", recall_ranks(F.normalize(S_img[te] @ Qc), S_img[te]))

# --- ridge regression ( learns scale/rotation generalized) ---
def ridge(X, Y, lam=1.0):
    d = X.shape[1]
    return torch.linalg.solve(X.T @ X + lam * torch.eye(d), X.T @ Y)

Wr = ridge(S_img[tr], S_txt[tr])
report("ridge img->text", recall_ranks(F.normalize(S_img[te] @ Wr), S_txt[te]))
