# Exp05 — Stitching a robot policy with a latent world model

University-grade question, small-scale test: **do a pretrained robot policy and an
independently-trained latent world model share enough latent geometry that one rotation
plugs them together?**

## The two machines

| | Model | What it is | Latent |
|---|---|---|---|
| M1 | **ACT** (`lerobot/act_aloha_sim_transfer_cube_human`, 51M params) | Action Chunking Transformer (Zhao et al. 2023, arXiv:2304.13705) trained on aloha cube-transfer simulation: resnet18 visual backbone → 5-layer transformer encoder → CVAE decoder emitting 100-step action chunks | 512-d, mean over encoder tokens (zero-VAE-latent token + robot-state token + 300 camera patch tokens) |
| M2 | **JEPA-lite** (self-trained) | DINOv2-small frame encoder (384-d CLS) + small MLP dynamics head `z_{t+1} = f(z_t, a_t)` trained on the rollouts with cosine+MSE, JEPA-style latent prediction | 384-d |

Data: 2,983 aligned (frame, state, next-actions) tuples from 50 `lerobot/aloha_sim_transfer_cube_human`
episodes, episode-level 80/20 split so the test set is whole episodes the models were never aligned on.

The one trick being tested: fit a **single Procrustes rotation on static matched frames only**
(also ZCA-whiten eps=0.01 first, same recipe as exps 02–04). No dynamics supervision at all
in the fit. Then ask how much *dynamics* comes across the seam.

## Results

| Test | Number | Control | Verdict |
|---|---|---|---|
| Static-frame retrieval (test episodes), R@1 / R@5 | **0.521 / 0.724** | chance ≈ 0.001 | ✅ pictures agree |
| Rotated motion-delta cosine, h = 1 / 5 / 10 / 20 steps | **0.003 / −0.002 / 0.009 / 0.000** | −0.002 (random rotation) | ❌ motion does not agree |
| JEPA-lite dynamics quality, cos(predicted, true next latent) | **0.918** | 0.029 (random start state, same action) | world model itself is good |
| Dynamics-predictor forecast vs ACT's actual latent change (cos) | **0.074** | ≈ 0 | weak physics agreement |
| Action-chunk readout R² (ridge, 12×14-d chunks) | ACT **0.93**, DINO **0.88** | — | both spaces carry action info |
| Zero-shot: DINO's decoder applied to rotated ACT latents | **0.89** | random-rotation same-fit 0.93* | mild transfer through the stitch |

\* The random-rotation "control" for readouts is weak by construction: an orthogonal rotation
plus a refit decoder is exactly invariant, so R² stays identical — it only shows the decoder
was refit, not that the mapping is good. The meaningful control is the DINO decoder transfer:
0.89 with fitted rotation vs the ACT-native 0.93.

## The headline

**Images align, physics doesn't.**

A rotation fitted purely on *what the world looks like* carries ACT's visual semantics into
DINOv2 space almost as well as our CLIP↔MiniLM translations carried text
(exp03 went further: it dropped zero-shot CIFAR-10 accuracy by only 0.2%).
But ACT's *dynamics* — how its latent changes when the arms move — survives that same
rotation essentially not at all (delta-cosine ≈ 0.00 at every horizon, same as a random
rotation).

This is the first picture in our series where the answer is "no", and it sharpens what the
earlier wins actually proved:

- Exp03/04 showed **object-level/semantic structure** is shared across independently trained encoders.
- Exp05 shows **temporal-dynamical structure** is not — at least not linearly, and not
  from a static-frame fit.

Which is consistent with the JEPA literature's own claims (e.g. V-JEPA 2, arXiv:2506.09985):
predictive world models are useful precisely because their latent *dynamics* are their
product — that is the part that has to be learned by predicting, not recovered by rotating.

So: plugging a pretrained policy's face onto an unused world model with one linear map
gives you the picture, not the physics.

## Reproduce

```bash
cd experiments/05_robot
uv venv env --python 3.12
uv pip install --python env/bin/python torch numpy transformers==4.56.2 "huggingface-hub<1.0" datasets pillow lerobot
brew install ffmpeg@7   # torchcodec needs FFmpeg ≤ 8 shared libs
DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib ./env/bin/python extract.py   # ~10 min, 2.7 GB corpus
../../.venv/bin/python dino_extract.py                                        # DINOv2 latents
./env/bin/python run.py                                                       # ~6 min on MPS
```

Files: `extract.py` (LeRobotDataset → corpus), `dino_extract.py`, `run.py` (latents + stitch + all
metrics), `results/corpus.pt`, `results/{act,dino}.pt`, `results/results.json`.
