"""
Experiment 01 — analysis: is DINOv2's space aligned with CLIP's?

Loads results/embeddings.pt and computes:
1. Linear & RBF CKA on the raw spaces                     (structure agreement)
2. Retention (Kornblith-style) on similarity structures
3. Top-k neighbor overlap between the two spaces
4. Linear Procrustes DINO->CLIP: fit on HALF the anchor set, evaluate
   cosine-retrieval Recall@k on the held-out half.

Run: python analyze.py
"""
import torch
from pathlib import Path
import torch.nn.functional as F

HERE = Path(__file__).parent
d = torch.load(HERE / "results" / "embeddings.pt")
clip_f, dino_f, y = d["clip"].float(), d["dino"].float(), d["labels"]
N = clip_f.shape[0]


# ---------- CKA ----------
def cka(X, Y, kernel="linear"):
    if kernel == "linear":
        K = X @ X.T
        L = Y @ Y.T
    else:  # rbf
        d2 = (X**2).sum(1)[:, None] + (X**2).sum(1)[None] - 2 * X @ X.T
        K = torch.exp(-d2 / d2.median())
        d2 = (Y**2).sum(1)[:, None] + (Y**2).sum(1)[None] - 2 * Y @ Y.T
        L = torch.exp(-d2 / d2.median())
    K = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    L = L - L.mean(0, keepdim=True) - L.mean(1, keepdim=True) + L.mean()
    return ((K * L).sum() / (K.norm() * L.norm())).item()


def centroid_kernel(X):
    # kernel induced by distances to class centroids — for retention sanity check
    C = torch.stack([X[y == c].mean(0) for c in range(10)])
    S = F.normalize(C, dim=1)
    M = S @ S.T
    M = M - M.mean()
    return M / M.norm()


# ---------- neighbor structure ----------
def topk_overlap(A, B, ks=(1, 5, 10, 32, 100)):
    assert torch.allclose(A, A)  # normalized already
    SA = (A @ A.T).fill_diagonal_(-9)
    SB = (B @ B.T).fill_diagonal_(-9)
    nA = SA.topk(max(ks), dim=1).indices
    nB = SB.topk(max(ks), dim=1).indices
    out = {}
    for k in ks:
        setA = torch.stack([torch.zeros(N).scatter_(0, nA[i, :k], 1.) for i in range(N)])
        setB = torch.stack([torch.zeros(N).scatter_(0, nB[i, :k], 1.) for i in range(N)])
        out[k] = ((setA * setB).sum(1) / k).mean().item()
    return out


# ---------- projection ----------
def orthogonal_procrustes(X, Y):
    # find Q minimizing ||XQ - Y||_F, Q orthogonal (torch.linalg.matrix_power-free)
    M = X.T @ Y
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt  # semi-orthogonal: dims(X) x dims(Y)


print(f"n={N}\n--- raw agreement (no projection) ---")
print(f"linear CKA:             {cka(clip_f, dino_f):.4f}")
print(f"rbf CKA:                {cka(clip_f, dino_f, 'rbf'):.4f}")
print(f"topk neighbor overlap:  {topk_overlap(clip_f, dino_f)}")

# sanity: agreement of each space's structure with class-centroid structure
print(f"centroid-structure corr (clip vs dino): "
      f"{(centroid_kernel(clip_f) * centroid_kernel(dino_f)).sum():.4f}")

half = N // 2
Xtr, Ytr = dino_f[:half], clip_f[:half]
Xte, Yte = dino_f[half:], clip_f[half:]

Q = orthogonal_procrustes(Xtr, Ytr)
proj_test = F.normalize(Xte @ Q)
rec = proj_test @ Yte.T  # retrieval: dino-space query -> nearest clip embedding
# compute exact ranks
order = rec.argsort(dim=1, descending=True)
ranks = (order == torch.arange(N - half)[:, None]).nonzero()[:, 1]
for k in (1, 5, 10, 100):
    print(f"procrustes distinct->clip Recall@{k:<4}: {(ranks < k).float().mean():.4f}")

print(f"\nbaseline (random space would be ~R@1 = {1.0/(N-half):.5f})")
