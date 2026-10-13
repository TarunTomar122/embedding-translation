# Exp05 — Stitching a robot policy with a latent world model

The robotics-grade version of this project's core question: **do a pretrained robot policy
and an independently-trained latent world model share enough latent geometry that one map
plugs them together — and does the *physics* survive the seam?**

## The two machines

| | Model | What it is | Latent |
|---|---|---|---|
| M1 | **ACT** (`lerobot/act_aloha_sim_transfer_cube_human`, 51M params) | Action Chunking Transformer (Zhao et al. 2023, arXiv:2304.13705), aloha cube-transfer simulation: resnet18 → 5-layer transformer encoder → CVAE decoder emitting 100-step action chunks | 512-d, mean over encoder tokens (zero-VAE-latent + robot-state + 300 camera patch tokens) |
| M2 | **JEPA-lite** (self-trained) | DINOv2-small frame encoder (384-d CLS) + MLP dynamics head `z_{t+1} = f(z_t, a_t)`, cosine+MSE, latent prediction only | 384-d |

Data: 2,983 aligned (frame, state, next-actions) tuples from 50 `lerobot/aloha_sim_transfer_cube_human`
episodes, episode-level 80/20 split. All whitening/ZCA eps=0.01, CPU (`eigh` trap).

## Main results (run.py)

| Test | Number | Control | Verdict |
|---|---|---|---|
| Static-frame retrieval (unseen episodes), R@1 / R@5 | **0.521 / 0.724** | chance ≈ 0.001 | pictures agree |
| Rotated motion-delta cosine, whitened, h = 1 / 5 / 10 (20) steps | **0.058 / 0.218 / 0.309** (n=4) | −0.001 / −0.002 / 0.003 (random rotation) | some physics comes across, growing with horizon |
| JEPA-lite dynamics quality, cos(predicted, true next latent) | **0.918** | 0.029 | the world model itself is good |
| Dynamics forecast vs ACT's actual latent change, cos | **0.074** | ≈ 0 | weak stitching-level agreement |
| Action-chunk readout R² (ridge, 12×14-d chunks) | ACT **0.93**, DINO **0.88** | — | both spaces carry action info |
| Zero-shot: DINO's decoder applied to rotated ACT latents | **0.89** | identity 0.93 | mild transfer through the stitch |

## Exp05b — how far can the physics be pushed across? (`run_b.py`)

Same data; four increasingly sophisticated stitches, all evaluated on unseen episodes
(test pairs: 260 / 57 / 19 / 4 at h = 1 / 5 / 10 / 20 — h=20 is n-toy, ignore it):

| Stitch fitted with | h=1 | h=5 | h=10 |
|---|---|---|---|
| nothing (random rotation) | −0.001 | −0.002 | 0.004 |
| static frames, linear Procrustes | 0.058 | 0.218 | 0.309 |
| **motion deltas only, linear LS** | 0.007 | 0.034 | 0.036 |
| static frames, 2-layer MLP, deltas via mapped frames | **0.094** | **0.374** | **0.454** |
| **motion deltas only, MLP** | 0.011 | 0.041 | 0.043 |

Three honest findings:

1. **Static training does carry some dynamics.** The first pass of this experiment scored
   rotated deltas ≈ 0.000 and read as "pictures align, physics doesn't". That was an
   evaluation bug: deltas had been computed on *raw* (unwhitened) latents, where a shared
   rotation can't act coherently. On whitened latents the static-fit map aligns motion
   deltas at 0.22–0.31 cosine (h=5–10) — clearly above control, clearly below static retrieval.
2. **A non-linear stitcher trained on the same static frames does better still** —
   0.37/0.45 at h=5/10, roughly doubling the physics that crosses the seam. Same
   supervision (static pairs only!), extra expressivity: the aligned directions live on a
   slight bend, not a plane.
3. **Fitting directly on motion deltas fails, both linear and MLP** (≈ 0.01–0.04, at or
   barely above control). Deltas are noisy and high-variance relative to static geometry;
   supervised least-squares on them fits noise. The dynamics info is carried by the
   *objects*, not by regressing onto the motion directly.

## The headline, corrected

**Pictures transfer better than physics — but physics partly survives, and the static-only
recipe is enough to get a real chunk of it.** The alignment quality for *what the world looks
like* (retrieval 0.52) is far ahead of *how the world moves* (delta-cos 0.09 at one step,
converging toward ~0.45 as the delta becomes more "positional" at longer horizons). Long-horizon
deltas are quasi-positional (state geometry), which is exactly why they align best — one-step
reaction dynamics remain the least shared structure. This is, inside this project, the
strongest evidence that JEPA-style predictive training encodes something that plain
contrastive/static alignment does not expose linearly.

## Reproduce

```bash
cd experiments/05_robot
uv venv env --python 3.12
uv pip install --python env/bin/python torch numpy transformers==4.56.2 "huggingface-hub<1.0" datasets pillow lerobot
brew install ffmpeg@7   # torchcodec needs FFmpeg ≤ 8 shared libs
DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib ./env/bin/python extract.py   # ~10 min, 2.7 GB corpus
../../.venv/bin/python dino_extract.py                                        # DINOv2 latents
./env/bin/python run.py                                                       # main stitch + metrics
./env/bin/python run_b.py                                                     # stitch ablations (Exp05b)
```

Files: `extract.py` (LeRobotDataset → corpus), `dino_extract.py`, `run.py`, `run_b.py`,
`results/{corpus,act,dino,stitch,stitch_b}.pt`, `results/results{,_b}.json`.

## Exp06 — does the world model help ACT act? (`run_c.py`)

Offline planning test on 57 unseen goal states (goal = the latent the demo actually reached
4 sampled frames later. Plan candidates scored by rolling the (stitched) DINO dynamics model
forward from the current latent; scored against ground-truth executed chunks (RMSE on
state-normalized 4×14-d actions):

| Plan | RMSE |
|---|---|
| ACT's own latent → chunk decoder (ridge) | **0.250** |
| DINO latent → chunk decoder (ridge) | 0.335 |
| mean train chunk | 0.964 |
| world-model planning over a state-conditioned candidate bank | 1.34 |
| world-model planning over demonstrator+jitter bank | 1.47 |

**The world loses.** So we asked why, with an oracle-in-bank diagnostic: when the TRUE
action chunk is among 201 candidates, the rollout cost ranks it on average **94.7 / 201**
(≈ chance 100); direct comparison scores 1.0 by construction. Mechanism: the dynamics
predictor maps different chunks to nearby latent futures — its 0.92 one-step cosine
accuracy lives in continuous latent similarity, not in *chunk-level discrimination*.
Rollout noise compounds; the discriminative signal that does come through the seam is in
the objects (Exp05b), not in ranking futures.

**Honest Exp06 verdict: at this laptop scale, the stitched world model does not improve
action selection over direct reading of the model's own latents.** Retry-worthy with:
(a) full 20k-frame corpus instead of our 3k subsample, (b) a generative chunk head
(diffusion, as in ACT/Diffusion-Policy proper) instead of a nearest-neighbor bank,
(c) far-horizon goals where Exp05b showed the seam is strongest.
