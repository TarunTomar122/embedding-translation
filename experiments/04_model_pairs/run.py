"""Exp 04 run: same unsupervised recipe on 3 image<->text pairs, fair comparison.
Rank metric is the CORRECT diagonal-rank (same as exp02 loop.ranks_of):
  o = Score.argsort(1, descending=True).argsort? -> use (o == arange) trick like loop.py
Recipe: whiten(eps=0.01) -> procrustes + ridge-fused scores -> train-side CSLS k variants swept.
Split: seed 0, first half fit / second half eval (same as exp02 loop).
"""
import sys, json, torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "02_vision_vs_text"))
from connectors_v2 import whiten
spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
    "loop", Path(__file__).parent.parent / "02_vision_vs_text" / "loop.py")
loop = __import__("importlib.util", fromlist=["util"]).module_from_spec(spec)
spec.loader.exec_module(loop)
ranks_of = loop.ranks_of
TR, TE = loop.TR, loop.TE
NRM = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)

def score_pair(name, X, Y, clip_B=False, lam=0.3):
    torch.manual_seed(0); n = len(X)
    idx = torch.randperm(n); tr, te = idx[:n//2], idx[n//2:]
    Xw, Yw = whiten(X), whiten(Y)
    Xn, Yn = NRM(Xw[tr]), NRM(Yw[tr])
    U, _, Vt = torch.linalg.svd(Xn.T @ Yn, full_matrices=False)
    if clip_B:
        WB = torch.linalg.solve(Xn.T @ Xn + lam*torch.eye(Xn.shape[1]), Xn.T @ Yn)
        U2, _, Vt2 = torch.linalg.svd((Xn @ WB).T @ Yn, full_matrices=False)
        S = NRM(Xw[te]) @ (U@Vt) @ NRM(Yw[te]).T
        S2 = NRM(Xw[te]) @ (WB@(U2@Vt2)) @ NRM(Yw[te]).T
        z = lambda m: (m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-9)
        S = z(S) + 0.5*z(S2)
    else:
        S = NRM(Xw[te]) @ (U@Vt) @ NRM(Yw[te]).T
    out = {}
    out["plain"] = ranks_of(S)["r@1"]
    # train-side CSLS
    rS = NRM(Xw[te]) @ NRM(Xw[tr]).T
    rT = NRM(Yw[te]) @ NRM(Yw[tr]).T
    for k in (10, 50, 150):
        cs_i = rS.topk(k, dim=1).values.mean(1)
        cs_t = rT.topk(k, dim=1).values.mean(1)
        out[f"csls{k}"] = ranks_of(2*S - cs_i[:, None] - cs_t[None, :])["r@1"]
    print(name, out, flush=True)
    return {"pair": name, **out}

if __name__ == "__main__":
    d2 = torch.load(Path(__file__).parent.parent/"02_vision_vs_text"/"results"/"embeddings.pt")
    catT = torch.load(Path(__file__).parent.parent/"02_vision_vs_text"/"results"/"captions_all5.pt")
    res = [
        score_pair("A_clipb32_vs_minilm", d2["clip"].float(), F.normalize(catT.mean(1), dim=-1)),
        score_pair("B_siglip_vs_mpnet",
                   (q := torch.load(Path(__file__).parent/"results"/"pairs.pt"))["siglip"].float(),
                   q["mpnet_mean5"].float()),
        score_pair("C_clipb16_vs_minilm",
                   (p := torch.load(Path(__file__).parent/"results"/"pairs.pt"))["clip16"].float(),
                   p["mpnet_mean5"].float()),
        score_pair("D_clipb16_vs_mpnet", p["clip16"].float(), p["mpnet_mean5"].float(), clip_B=False),
    ]
    json.dump(res, open(Path(__file__).parent/"results"/"pair_scores.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
