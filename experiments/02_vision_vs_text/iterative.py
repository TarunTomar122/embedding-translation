"""
Round 4 (fixed) — iterative linear alignment, composite map tracked cleanly.
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
Ytr_n, Yte_n = F.normalize(Ytr), F.normalize(Yte)

@torch.no_grad()
def ranks_of(Score):
    order = Score.argsort(1, descending=True)
    r = (order == torch.arange(Score.shape[0], device=Score.device)[:, None]).nonzero()[:, 1]
    return {f"r@{k}": round((r < k).float().mean().item(), 4) for k in (1, 5, 10, 50)} | \
           {"mrr": round((1 / (r.float() + 1)).mean().item(), 4)}

out = {}
Wc = torch.eye(S_img.shape[1], device=dev)
Xt_ = Xtr.clone()
for it in range(6):
    d_in = (Xtr.shape[1] if it == 0 else (Xtr @ Wc).shape[1])
    A = Xt_ @ Wc if it == 0 else None
    # fit step on current inputs
    if it == 0:
        W_step = torch.linalg.solve(Xtr.T @ Xtr + 0.01 * torch.eye(d_in, device=dev), Xtr.T @ Ytr_n)
        Wc = W_step
    else:
        Xcur = F.normalize(Xtr @ Wc)
        W_step = torch.linalg.solve(Xcur.T @ Xcur + 0.01 * torch.eye(d_in, device=dev), Xcur.T @ Ytr_n)
        # then procrustes-square the step
        M = (Xcur @ W_step).T @ Ytr_n
        U, _, Vt = torch.linalg.svd(M, full_matrices=False)
        W_step = W_step @ U @ Vt
        Wc = Wc @ W_step
    out[f"step{it}"] = ranks_of(F.normalize(Xte @ Wc) @ Yte_n.T)
    print(f"step{it}:", out[f"step{it}"])

json.dump(out, open(HERE / "results" / "iterative_whitened.json", "w"), indent=1)
print("saved")
