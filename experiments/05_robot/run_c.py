"""Exp06 — does the stitched world model help ACT act?

Offline planning test (no simulator replay needed):
- Goal identification: for every held-out frame t, the "goal latent" = DINO latent of a
  frame h=20 steps later (what the demo trajectory actually reached).
- ACT baseline: ACT's own 20-step mean chunk (average of its native chunk) as the plan.
- World-model planning: sample K candidate 20-step action chunks from the SAME episode's
  ACT-autoregressive chunks? No — candidates = chunks executed by the demonstrator at OTHER
  frames (behavior cloning data bank). Roll JEPA-lite dynamics forward in DINO space from
  z_t, starting each candidate; score = - distance between rolled latent at each alpha-step
  and the goal latent lat(t+20) (early-terminating interest, plus per-step alignment).
- Metric: decode chosen chunk's error w.r.t. TRUE future 20-step chunk: L2 per element
  (normalized actions), plus ACT's own leaderless error. Lower = better.
Also a stitched variant: do the planning in ACT space (rotate latents over seam).
"""
import warnings; warnings.filterwarnings("ignore")
import torch, json, math
from pathlib import Path
import torch.nn as nn

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
HERE = Path(__file__).parent
OUT = HERE / "results"

c = torch.load(HERE / "results" / "corpus.pt")
data = {k: c[k] for k in c}
act = torch.load(HERE / "results" / "act.pt").float().cpu()
dino = torch.load(HERE / "results" / "dino.pt").float().cpu()
N = act.shape[0]
stitch = torch.load(HERE / "results" / "stitch_b.pt")
R_static = stitch["R_static"]

perm = torch.randperm(N)
tr, te = perm[: int(0.8 * N)], perm[int(0.8 * N):]

# pair structure at h=20 for goals
pairs = {}
for h in (1, 4):
    idx = torch.where((data["episode"][1:] == data["episode"][:-1]) &
                      (data["frame"][1:] == data["frame"][:-1] + h))[0]
    pairs[h] = (idx[torch.isin(idx, tr)], idx[torch.isin(idx, te)])
trp20, tep20 = pairs[4]
print("goal pairs (train/test):", len(trp20), len(tep20), flush=True)

H = 4  # plan length; every sampled frame has an executed next-action and goal frame at +4
print("H actions:", H, flush=True)
Ah = torch.stack([ch[:H] for ch in data["chunk"]]).float()   # [N,H,14]
Amu, Asd = Ah.mean((0,1), keepdim=True), Ah.std((0,1), keepdim=True) + 1e-6
An = (Ah - Amu) / Asd

def whiten(X, eps=0.01):
    mu = X.mean(0, keepdim=True); Xc = X - mu
    C = Xc.T @ Xc / len(Xc)
    evals, evecs = torch.linalg.eigh(C.cpu())
    W = evecs @ torch.diag(1.0 / (evals + eps).sqrt())
    return (Xc @ W), W, mu

act_w, _, _ = whiten(act)
din_w, W_d, mu_d = whiten(dino)

# rebuild the latent dynamics predictor exactly as run.py did (h=1 pairs)
class Dyn(nn.Module):
    def __init__(self, d_in=384, d_lat=256, a_dim=14):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in + a_dim, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d_in))
    def forward(self, z, a): return self.net(torch.cat([z, a], dim=1))

idx1 = pairs[1][0]
a1 = torch.stack([data["chunk"][int(i)][0] for i in idx1]).float()
z_t, z_t1, a1 = din_w[:-1][idx1].to(DEV), din_w[1:][idx1].to(DEV), a1.to(DEV)
ntr = int(0.9 * len(z_t))
dyn = Dyn().to(DEV)
opt = torch.optim.AdamW(dyn.parameters(), lr=1e-3, weight_decay=0.05)
for e in range(300):
    ii = torch.randperm(ntr)[:512]
    loss = (1 - torch.cosine_similarity(dyn(z_t[ii], a1[ii]), z_t1[ii], dim=1)).mean() + \
           0.3 * torch.nn.functional.mse_loss(dyn(z_t[ii], a1[ii]), z_t1[ii])
    opt.zero_grad(); loss.backward(); opt.step()
print("dyn trained (train pairs only).", flush=True)

# ---- planning eval on TEST goals ----
# candidates: the bank of demonstrator chunks from TRAIN frames (behavior-cloning bank);
# for each test goal frame t (index i in original ordering, with valid 20-step gap),
# find candidate chunks whose start frame has episode==t.ep and start within [t-10, t+1]??
# Fair candidates: any train chunk (bank across episodes) - avoids leakage of the exact answer.
K = 200
res_plans = []
def rollout(z0, acts):
    z = z0
    zs = []
    with torch.no_grad():
        for t in range(acts.shape[1]):
            z = dyn(z, acts[:, t].to(DEV))
            zs.append(z)
    return torch.stack(zs)  # [H,K,384]

base_err, wm_err = [], []
goal_d, cand_idx_all = None, None
te_goals = tep20  # frames i such that i+20 exists, same episode; i in test split
print("test goals:", len(te_goals), flush=True)

start_lat = din_w[:-1][te_goals].to(DEV)                # z_t
goal_lat = din_w[1:][te_goals].to(DEV)                  # z_{t+20}
true_ch = An[1:][te_goals][:, :H]
plan_truth = An[:-1][te_goals][:, :H]  # executed actions at t..t+3

# ACT baseline plan: the TRUE chunk is what ACT would need to reproduce; use ACT's own chunk,
# i.e. train a chunk decoder on ACT latents (dynamical readout, run.py style) and decode.
def ridge_readout(X, Y, lam=10.0):
    X, Y = X.cpu(), Y.cpu()
    A = X.T @ X + lam * torch.eye(X.shape[1])
    return torch.linalg.solve(A, X.T @ Y)

Yc = An[:, :H].reshape(N, -1)
pr_act = ridge_readout(act_w[tr], Yc[tr])
act_plan = (act_w[te_goals] @ pr_act).reshape(-1, H, 14)   # ACT-space decoder plan
pr_dino = ridge_readout(din_w[tr], Yc[tr])
dino_plan = (din_w[te_goals] @ pr_dino).reshape(-1, H, 14) # DINO-space decoder plan

# world-model planning: candidate bank = train chunks, K random per goal
cand_start_lat = None
wm_plans = []

for gi, z0 in enumerate(start_lat):
    # state-conditioned candidate bank: chunks from the K train frames whose START latent
    # is nearest to z0 (static-alignment retrieval, Exp05's own 0.52-R@1 ability)
    with torch.no_grad():
        # refiner bank: ACT's own decoded plan + jitters of it + nearest-state demonstrator chunks
        jitter = An[tr][:200] + 0.10 * torch.randn(200, H, 14)
        close = (din_w[tr] @ z0.cpu().squeeze()).argsort(descending=True)[:200]
        bank = torch.cat([jitter, An[tr][close]], 0)   # [400,H,14]
        K = bank.shape[0]
        zs = z0.repeat(K, 1)
        goal = din_w[4:][te_goals[gi]].to(DEV).unsqueeze(0).repeat(K, 1)
        dist = torch.zeros(K, device=DEV)
        for t in range(H):
            zs = dyn(zs, bank[:, t].to(DEV))
            dist = dist + (zs - goal).norm(dim=1)   # integral cost along the rollout
    k = dist.argmin().item()
    wm_plans.append(bank[k].cpu())
wm_plan = torch.stack(wm_plans)

def err(plan):
    return ((plan - plan_truth).pow(2).mean().item())

res = {
    "ACT_decoded_plan_R2": None,
    "action_chunk_RMSE_norm": {
        "ACT_space_decoder": err(act_plan) ** 0.5,
        "DINO_space_decoder": err(dino_plan) ** 0.5,
        "world_model_planned": err(wm_plan) ** 0.5,
        "mean_chunk_baseline": err(An[tr].mean(0, keepdim=True).expand(len(te_goals), H, -1)) ** 0.5,
    },
}
print("plan RMSE (normalized chunks):", {k: round(v,4) for k,v in res["action_chunk_RMSE_norm"].items()}, flush=True)
json.dump(res, open(OUT / "results_c.json", "w"), indent=2)
torch.save({"act_plan": act_plan, "dino_plan": dino_plan, "wm_plan": wm_plan,
            "truth": plan_truth, "goals": te_goals}, OUT / "plans.pt")
print("saved results_c.json")
