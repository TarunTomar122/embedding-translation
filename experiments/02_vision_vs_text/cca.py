"""
Round 5 — CCA-driven bridging on whitened spaces.

CCA finds directions of maximal correlation; mapping X into the shared CCA
subspace and then into Y's whitened coordinates may beat plain Procrustes
(Procrustes matches the whole geometry including noisy directions;
CCA concentrates on the correlated subspace).

Run: python cca.py   (uses whitened embeddings)
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from connectors_v2 import dev

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
def ranks_of(Score):
    order = Score.argsort(1, descending=True)
    r = (order == torch.arange(Score.shape[0], device=Score.device)[:, None]).nonzero()[:, 1]
    return {f"r@{k}": round((r < k).float().mean().item(), 4) for k in (1, 5, 10, 50)} | \
           {"mrr": round((1 / (r.float() + 1)).mean().item(), 4)}

out = {}
k = min(384, 512)  # full rank CCA via generalized eigendecomposition
Sxx = Xtr.T @ Xtr + 1e-3 * torch.eye(512, device=dev)
Syy = Ytr.T @ Ytr + 1e-3 * torch.eye(384, device=dev)
Sxy = Xtr.T @ Ytr
# generalized problem: T = Sxx^-1/2 Sxy Syy^-1/2 ; svd
from connectors_v2 import whiten  # noqa
Sxx_isqrt = torch.linalg.inv(torch.linalg.cholesky(Sxx)).T.float()
Syy_isqrt = torch.linalg.inv(torch.linalg.cholesky(Syy)).T.float()
T = Sxx_isqrt @ Sxy @ Syy_isqrt
Uc, S_, Vc = torch.linalg.svd(T, full_matrices=False)
Ax = Sxx_isqrt @ Uc            # X -> canonical space [512, 384]
Ay = Syy_isqrt @ Vc.T          # Y -> canonical space [384, 384]

# map: from canonical space back into Y via pseudo-inverse ridge (dual direction)
# score in canonical space directly (both spaces already whitened):
out["cca-canonical"] = ranks_of(F.normalize(Xtr @ Ax)[:0] @ 0*0 if False else
                                (F.normalize(Xte @ Ax) @ F.normalize(Yte @ Ay).T))
print("cca-canonical:", out["cca-canonical"])

# also constrained-rank CCA: keep top-m canonical directions
for m in (64, 128, 256):
    Sc = ranks_of(F.normalize(Xte @ Ax[:, :m]) @ F.normalize(Yte @ Ay[:, :m]).T)
    out[f"cca-m{m}"] = Sc
    print(f"cca-m{m}:", Sc)

json.dump(out, open(HERE / "results" / "cca.json", "w"), indent=1)
print("saved")
