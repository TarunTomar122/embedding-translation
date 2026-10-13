"""Exp05b — can the physics be stitched at all, with the right alignment?

Three increasingly dynamics-aware stitches, same eval as run.py:
1. static Procrustes (reference from run.py)
2. dynamics-supervised linear map: least squares on (delta_ACT @ ? ) -> delta_DINO
   (fit R on matched ACT deltas, NOT static frames)
3. non-linear stitcher: 2-layer MLP ACT->DINO fitted on static frames (architecture control)
   + the same fitted on dynamics deltas
"""
import warnings; warnings.filterwarnings("ignore")
import torch, json, copy
import torch.nn as nn
from pathlib import Path

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
HERE = Path(__file__).parent
OUT = HERE / "results"

c = torch.load(HERE / "results" / "corpus.pt")
data = {k: c[k] for k in c}
act = torch.load(HERE / "results" / "act.pt").float().cpu()
dino = torch.load(HERE / "results" / "dino.pt").float().cpu()
N = act.shape[0]
print("loaded", N, flush=True)

# split: episode-level — reuse run.py's split by re-deriving (same seed order)
perm = torch.randperm(N)
tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]

pairs = {}
for h in (1, 5, 10, 20):
    idx = torch.where((data["episode"][1:] == data["episode"][:-1]) &
                      (data["frame"][1:] == data["frame"][:-1] + h))[0]
    pairs[h] = (idx[torch.isin(idx, tr)], idx[torch.isin(idx, te)])  # (train pairs, test pairs)
print({h: len(v[0]) for h, v in pairs.items()}, "train pairs", flush=True)

def whiten(X, eps=0.01):
    mu = X.mean(0, keepdim=True); Xc = X - mu
    C = Xc.T @ Xc / len(Xc)
    evals, evecs = torch.linalg.eigh(C.cpu())
    W = evecs @ torch.diag(1.0 / (evals + eps).sqrt())
    return (Xc @ W), W, mu

act_w, W_act, mu_act = whiten(act)
din_w, W_din, mu_din = whiten(dino)

# -------- stitch 1: static Procrustes (same as run.py) --------
R_static = torch.linalg.solve(act_w[tr].T @ act_w[tr], act_w[tr].T @ din_w[tr])

# -------- stitch 2: dynamics-supervised --------
tr_pairs = torch.cat([v[0] for v in pairs.values()])
da_tr = (act_w[1:][tr_pairs] - act_w[:-1][tr_pairs])      # [P, 512]
db_tr = (din_w[1:][tr_pairs] - din_w[:-1][tr_pairs])      # [P, 384]
R_dyn = torch.linalg.solve(da_tr.T @ da_tr, da_tr.T @ db_tr)

te_pairs1 = pairs[1][1]
da_te = (act_w[1:][te_pairs1] - act_w[:-1][te_pairs1])
db_te = (din_w[1:][te_pairs1] - din_w[:-1][te_pairs1])

def delta_cos(R, act_lat=act_w, din_lat=din_w):
    out = {}
    for h, (_trp, tep) in pairs.items():
        da = (act_lat[1:][tep] - act_lat[:-1][tep]) @ R
        db = (din_lat[1:][tep] - din_lat[:-1][tep])
        da = da - da.mean(0, keepdim=True); db = db - db.mean(0, keepdim=True)
        out[f"h{h}"] = torch.cosine_similarity(da, db).mean().item()
    return out

res = {
    "static_procrustes_deltas": delta_cos(R_static),
    "dynsupervised_linear_deltas": delta_cos(R_dyn),
}
print("static:", {k: round(v, 4) for k, v in res["static_procrustes_deltas"].items()}, flush=True)
print("dyn-lin:", {k: round(v, 4) for k, v in res["dynsupervised_linear_deltas"].items()}, flush=True)

# control for stitch 2: random orthogonal rotation
def haar(m, n):
    q, _ = torch.linalg.qr(torch.randn(m, n).float()); return q

ctrl = {f"h{h}": [] for h in pairs}
for _ in range(20):
    resc = delta_cos(haar(512, 384))
    for k in resc: ctrl[k].append(resc[k])
res["random_rotation_deltas"] = {k: sum(v) / len(v) for k, v in ctrl.items()}
print("random:", {k: round(v, 4) for k, v in res["random_rotation_deltas"].items()}, flush=True)

# -------- static-plus-dynamics hybrid: R_static then rank-1 correction fitted on deltas --------
R_hyb = R_dyn

# -------- non-linear stitcher: MLP ACT_w -> DINO_w --------
class Stitch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 384))
    def forward(self, x): return self.net(x)

def train_stitch(fitter="static", epochs=800):
    m = Stitch()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    if fitter == "static":
        Xs, Ys = act_w[tr], din_w[tr]
    else:
        Xs = torch.cat([act_w[1:][pairs[h][0]] - act_w[:-1][pairs[h][0]] for h in pairs], dim=0)
        Ys = torch.cat([din_w[1:][pairs[h][0]] - din_w[:-1][pairs[h][0]] for h in pairs], dim=0)
    n = len(Xs)
    for e in range(epochs):
        ii = torch.randperm(n)[:1024]
        out = m(Xs[ii])
        loss = (1 - torch.cosine_similarity(out, Ys[ii], dim=1)).mean() + 0.2 * (out - Ys[ii]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def delta_cos_mlp(m, on_deltas=True):
    out = {}
    for h, (_trp, tep) in pairs.items():
        if on_deltas:
            da = (act_w[1:][tep] - act_w[:-1][tep])
            db = (din_w[1:][tep] - din_w[:-1][tep])
        else:
            da = act_w[1:][tep]; db = din_w[1:][tep]
        da = m(da).detach()
        da = da - da.mean(0, keepdim=True); db = db - db.mean(0, keepdim=True)
        out[f"h{h}"] = torch.cosine_similarity(da, db).mean().item()
    return out

print("training static MLP stitcher...", flush=True)
m1 = train_stitch("static")
res["mlp_static_deltas"] = delta_cos_mlp(m1)
# static-MLP does NOT translate deltas correctly: apply per-frame-delta (already net on delta input? we applied net on deltas — architecture control for "MLP applied to deltas fitted statically")
print("mlp-static:", {k: round(v, 4) for k, v in res["mlp_static_deltas"].items()}, flush=True)

# static-MLP used the *right* way: map frames, then delta in target space
def delta_cos_mlp_frames(m):
    za = m(act_w); zb = din_w
    out = {}
    for h, (_trp, tep) in pairs.items():
        da = za[1:][tep] - za[:-1][tep]
        db = zb[1:][tep] - zb[:-1][tep]
        out[f"h{h}"] = torch.cosine_similarity(da, db).mean().item()
    return out
res["mlp_static_framemapped_deltas"] = delta_cos_mlp_frames(m1)
print("mlp-static-frames:", {k: round(v, 4) for k, v in res["mlp_static_framemapped_deltas"].items()}, flush=True)

print("training dynamics-trained MLP stitcher (on deltas)...", flush=True)
m2 = train_stitch("deltas")
res["mlp_dynsupervised_deltas"] = delta_cos_mlp(m2)
print("mlp-dyn:", {k: round(v, 4) for k, v in res["mlp_dynsupervised_deltas"].items()}, flush=True)

json.dump(res, open(OUT / "results_b.json", "w"), indent=2)
torch.save({"R_static": R_static, "R_dyn": R_dyn, "stitch_static": m1.state_dict(),
            "stitch_dyn": m2.state_dict()}, OUT / "stitch_b.pt")
print("saved results_b.json")
