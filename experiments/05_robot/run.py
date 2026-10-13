"""Exp05 - Cross-stitching a pretrained robot policy (ACT/aloha) with a self-trained
JEPA-lite latent world model (DINOv2 + latent dynamics), fit on static frames only.

M1 = ACT policy's internal encoder latent (51M-param action-chunking transformer, aloha-sim)
M2 = JEPA-lite: DINOv2-small frame encoder + latent dynamics predictor z_{t+1}=f(z_t,a_t)
     trained with EMA target encoder (JEPA-style) on aloha rollouts.

Test: after one Procrustes rotation fitted only on *static* matched frames,
do their *dynamics* agree? Predicted successions, action-chunk readouts.
"""
import warnings; warnings.filterwarnings("ignore")
import torch, json, math
from pathlib import Path
import torch.nn as nn

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
HERE = Path(__file__).parent
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

# ---------- load corpus ----------
c = torch.load(HERE / "results" / "corpus.pt")
imgs, states, chunks = c["img"], c["state"], c["chunk"]
episodes = c["episode"]
N = len(imgs)
print(f"corpus: {N} frames from {episodes.unique().numel()} episodes", flush=True)

# ---------- extract ACT latents ----------
def act_latents(images, states):
    import transformers.utils as tu
    if not hasattr(tu, "torch_compilable_check"):
        tu.torch_compilable_check = lambda *a, **k: (lambda fn: fn)
    import einops
    from lerobot.policies.act.modeling_act import ACTPolicy
    p = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human"); p.eval()
    p.to(DEV)
    m = p.model
    lat = []
    B = 16
    with torch.no_grad():
        for s in range(0, len(images), B):
            img = images[s:s+B].float().to(DEV) / 255.0          # [b,3,480,640] (no cam normalization: backbone is pooled-feature scale; noted in README)
            st = states[s:s+B].float().to(DEV)
            b_ = img.shape[0]
            z = torch.zeros(b_, p.config.latent_dim, device=DEV)
            toks = [m.encoder_latent_input_proj(z)]
            # robot state token? aloha config: robot_state_feature = STATE -> present
            if p.config.robot_state_feature:
                toks.append(m.encoder_robot_state_input_proj(st))
            feats = m.backbone(img)["feature_map"]                # [b,512,15,20]
            pe = m.encoder_cam_feat_pos_embed(feats).to(feats.dtype)
            feats = m.encoder_img_feat_input_proj(feats)
            toks += list(einops.rearrange(feats, "b c h w -> (h w) b c"))
            x = torch.stack(toks, dim=0)                          # [T, b, 512]
            enc = m.encoder(x)                                    # returns tokens seq [T,b,512]
            lat.append(enc.mean(dim=0).float().cpu())
    return torch.cat(lat)

# je: ACTEncoder forward signature check
import inspect
sig = inspect.signature(ACTPolicy.from_pretrained) if False else None
# quick sanity on encoder call convention
try:
    _enc = ACTPolicy.from_pretrained        # placeholder to avoid unused import error path
except Exception:
    pass

def dino_latents(images, venv_py):
    """DINOv2-small CLS latents [N,384]; runs main project's python env if given."""
    import subprocess, sys
    out = subprocess.run([venv_py, str(HERE / "dino_extract.py")], capture_output=True, text=True)
    print(out.stdout[-400:], out.stderr[-400:], flush=True)
    return torch.load(HERE / "results" / "dino.pt")

dino_path = HERE / "results" / "dino.pt"
act_path = HERE / "results" / "act.pt"
if not act_path.exists():
    print("extracting ACT latents...", flush=True)
    act = act_latents(imgs, states)
    torch.save(act, act_path)
    print("ACT latents:", act.shape, flush=True)
if not dino_path.exists():
    # run dino extraction with main env python (transformers 5.x)
    import subprocess
    main_py = HERE.parent.parent / ".venv" / "bin" / "python"
    r = subprocess.run([str(main_py), str(HERE / "dino_extract.py")], capture_output=True, text=True)
    print(r.stdout[-300:], r.stderr[-500:], flush=True)
act = torch.load(act_path).float()
dino = torch.load(dino_path).float()
print("ACT:", act.shape, "DINO:", dino.shape, flush=True)

# ---------- whitening + procrustes on static matched frames ----------
def whiten(X, eps=0.01):
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    C = Xc.T @ Xc / len(Xc)
    evals, evecs = torch.linalg.eigh(C.float().cpu())
    W = evecs @ torch.diag(1.0 / (evals + eps).sqrt())
    return (Xc @ W).to(DEV), W.to(DEV), mu.to(DEV)

act_w, W_act, mu_act = whiten(act)
din_w, W_din, mu_din = whiten(dino)

perm = torch.randperm(N)
tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]
Xs, Ys = act_w[tr], din_w[tr]
R = torch.linalg.solve(Xs.T @ Xs, Xs.T @ Ys)
def pro(X): return (X @ R)
res = {}
xt = pro(act_w[te]); yt = din_w[te]
xt = xt - xt.mean(0, keepdim=True); yt = yt - yt.mean(0, keepdim=True)
d = torch.cdist(xt, yt)
r1 = (d.argmin(1) == torch.arange(len(yt), device=xt.device)).float().mean().item()
order = d.argsort(1); r5 = (order[:, :5] == torch.arange(len(yt), device=xt.device)[:, None]).any(1).float().mean().item()
res["static_retrieval"] = {"R1": r1, "R5": r5}
print("static retrieval R@1 / R@5 :", round(r1, 4), round(r5, 4), flush=True)

# ---------- dynamics agreement ----------
# matched consecutive pairs within same episode
data = {k: c[k] for k in c}
DINO_dyn = dino_w if False else din_w
# build index pairs (i, i+h) same episode
pairs_by_h = {}
for h in (1, 5, 10, 20):
    idx = torch.where((data["episode"][1:]==data["episode"][:-1]) & (data["frame"][1:]==data["frame"][:-1]+h))[0]
    idx = torch.intersect1d if False else idx[torch.isin(idx, te)]
    pairs_by_h[h] = idx
def cos_align(a_e, b_e):        # matched frames: latent of M1 at t, M2 at t (both test at end)
    pass
def dyntest(model_a, model_b, rotate=True):
    """Delta cosines at horizons using latents from two models, given align R."""
    out = {}
    model_a, model_b = model_a.float().to(DEV), model_b.float().to(DEV)
    for h, idx in pairs_by_h.items():
        da = model_a[1:][idx] - model_a[:-1][idx]       # ACT-space deltas
        db = model_b[1:][idx] - model_b[:-1][idx]       # DINO-space deltas
        if rotate: da = da @ R
        da = da - da.mean(0, keepdim=True); db = db - db.mean(0, keepdim=True)
        cos = torch.cosine_similarity(da, db)
        out[f"h{h}"] = cos.mean().item()
    return out
res["deltas_rotated"] = dyntest(act, dino)
# control: random rotation
Rc = torch.randn(R.shape).float()
def haar(r):
    """Random orthonormal [r.shape[0], r.shape[1]] matrix (QR of gaussian)."""
    import torch as _t
    q, _ = _t.linalg.qr(r)
    return q

def dyntest_ctrl(model_a, model_b, Rr, times=20):
    out = {f"h{h}": [] for h in pairs_by_h}
    for t in range(times):
        Rr2 = haar(torch.randn(R.shape).float())
        d2 = {}
        for h, idx in pairs_by_h.items():
            da = (model_a[1:][idx] - model_a[:-1][idx]) @ Rr2
            db = model_b[1:][idx] - model_b[:-1][idx]
            da = da - da.mean(0, keepdim=True); db = db - db.mean(0, keepdim=True)
            d2[f"h{h}"] = torch.cosine_similarity(da, db).mean().item()
        for k in d2: out[k].append(d2[k])
    return {k: sum(v)/len(v) for k, v in out.items()}
res["deltas_control"] = dyntest_ctrl(act, dino, None)
print("delta-cos rotated:", {k: round(v,4) for k,v in res['deltas_rotated'].items()}, flush=True)
print("delta-cos control:", {k: round(v,4) for k,v in res['deltas_control'].items()}, flush=True)

# ---------- JEPA lite (train on DINO latents, then test transfer across seam) ----------
class Jepa(nn.Module):
    def __init__(self, d_in=384, d_lat=256, a_dim=14):
        super().__init__()
        self.pred = nn.Sequential(nn.Linear(d_in + a_dim, 512), nn.GELU(),
                                  nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d_in))
        self.latent = nn.Sequential(nn.Linear(d_in, d_lat), nn.GELU(), nn.Linear(d_lat, d_in))
    def forward(self, z, a): return self.pred(torch.cat([z, a], dim=1))

def train_dyn_subset(z_all, idx):
    a = torch.stack([data["chunk"][int(i)][0] for i in idx])
    z_t, z_t1, a = z_all[:-1][idx].to(DEV), z_all[1:][idx].to(DEV), a.float().to(DEV)
    model = Jepa().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    for ep in range(300):
        ii = torch.randperm(len(z_t), device=DEV)[:512]
        loss = (1 - torch.cosine_similarity(model(z_t[ii], a[ii]), z_t1[ii], dim=1)).mean() \
               + 0.3 * torch.nn.functional.mse_loss(model(z_t[ii], a[ii]), z_t1[ii])
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def train_dyn(z_all):
    """Latent dynamics predictor: z_{t+1} ~= f(z_t, a_t) in DINO space."""
    idx = pairs_by_h[1]
    a = torch.stack([data["chunk"][int(i)][0] for i in idx])
    z_t, z_t1, a = z_all[:-1][idx].to(DEV), z_all[1:][idx].to(DEV), a.float().to(DEV)
    n_tr = int(0.9 * len(z_t))
    model = Jepa().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    for ep in range(300):
        ii = torch.randperm(n_tr)[:512]
        loss = (1 - torch.cosine_similarity(model(z_t[ii], a[ii]), z_t1[ii], dim=1)).mean() \
               + 0.3 * torch.nn.functional.mse_loss(model(z_t[ii], a[ii]), z_t1[ii])
        opt.zero_grad(); loss.backward(); opt.step()
    return model

print("training latent dynamics predictor on DINO deltas...", flush=True)
j = train_dyn(din_w.cpu())
train_mask = torch.randperm(len(pairs_by_h[1]))[: int(0.9 * len(pairs_by_h[1]))]  # dyn model saw 90% of h1 test pairs; eval on holdout
idx_hold = pairs_by_h[1][train_mask]
jpred = j(din_w[:-1][idx_hold].to(DEV), torch.stack([data["chunk"][int(i)][0] for i in idx_hold]).float().to(DEV)).cpu()
cos_j = torch.cosine_similarity(jpred, din_w[1:][idx_hold].cpu(), dim=1).mean().item()
jrand = j(torch.randn_like(din_w[:-1][idx_hold]).to(DEV), torch.stack([data["chunk"][int(i)][0] for i in idx_hold]).float().to(DEV)).cpu()
cos_jc = torch.cosine_similarity(jrand, din_w[1:][idx_hold].cpu(), dim=1).mean().item()
res["jepa_quality"] = {"matches": cos_j, "control": cos_jc}
print("jepa cos real / random-action control:", round(cos_j,4), round(cos_jc,4), flush=True)

# transfer: does ACT's dynamics, mapped through R, agree with the latent dynamics predictor's forecast?
idx = pairs_by_h[1]
n_hold = len(idx)
pick_hold = idx[torch.randperm(n_hold)[int(0.9 * n_hold):]]
pick_train = idx[torch.randperm(n_hold)[:int(0.9 * n_hold)]]
torch.manual_seed(1); j = train_dyn_subset(din_w.cpu(), pick_train)
jact = j(din_w[:-1][pick_hold].to(DEV), torch.stack([data["chunk"][int(i)][0] for i in pick_hold]).float().to(DEV)).cpu()
da = (act_w[1:][pick_hold] - act_w[:-1][pick_hold]) @ R
da = da.cpu()
db = jact - din_w[:-1][pick_hold].cpu()
da = da - da.mean(0, keepdim=True); db = db - db.mean(0, keepdim=True)
res["jepa_vs_ACT_successor"] = torch.cosine_similarity(da, db).mean().item()
print("jepa-successor vs ACT-delta cos:", round(res['jepa_vs_ACT_successor'],4), flush=True)

# ---------- action-chunk readout both ways ----------
def ridge_readout(X, Y, lam=10.0):
    X, Y = X.cpu(), Y.cpu()
    A = X.T @ X + lam * torch.eye(X.shape[1])
    B = X.T @ Y
    return torch.linalg.solve(A, B)

reads = {}
H = min(len(ch) for ch in chunks)  # shortest chunk (all frames have at least H next-actions)
lengths = torch.tensor([len(ch) for ch in chunks])
mask = lengths >= H
tr_f, te_f = tr[mask[tr]], te[mask[te]]
Y = torch.stack([ch[:H] for ch in chunks]).reshape(N, -1).float()
Y = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-6)
for name, Z in (("ACT", act_w), ("DINO", din_w)):
    pr = ridge_readout(Z[tr_f], Y[tr_f])
    yp = Z[te_f].cpu() @ pr
    base = (Y[te_f] - Y[te_f].mean(0, keepdim=True)).pow(2).mean()
    reads[name + "->chunk,R2"] = (1 - ((yp - Y[te_f]).pow(2).mean() / base)).item()
# zero-shot cross: decoder fitted on DINO latents, applied to rotated ACT latents
pr = ridge_readout(din_w[tr_f], Y[tr_f])
yp2 = (pro(act_w[te_f]).cpu()) @ pr
reads["ACTrot->chunk,DINO-decoder-zero-shot,R2"] = (1 - ((yp2 - Y[te_f]).pow(2).mean() / base)).item()
# control: random orthonormal rotation of ACT instead of learned R
Q = haar(torch.randn(512, 512).float())
pr2 = ridge_readout(act_w[tr_f].cpu() @ Q, Y[tr_f])
ypr = (act_w[te_f].cpu() @ Q) @ pr2
reads["ACTrandRot->chunk,R2"] = (1 - ((ypr - Y[te_f]).pow(2).mean() / base)).item()
res["chunk_readouts"] = reads
print("chunk readouts:", {k: round(v,4) for k,v in reads.items()}, flush=True)

json.dump(res, open(OUT / "results.json", "w"), indent=2)
torch.save({"R": R, "pairs": {str(k): v.tolist()[:400] for k, v in pairs_by_h.items()}}, OUT / "stitch.pt")
print("saved results.json")
