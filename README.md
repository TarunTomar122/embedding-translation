# two models, one picture

Can an embedding space from one model be translated into an embedding space from a
*completely different* model — without letting the two models ever train together?

This project answers it at laptop scale with real measurements, not vibes:

- **Model 1:** CLIP ViT-B/32 image tower — eats photos, spits 512 numbers.
- **Model 2:** MiniLM-L6-v2 — eats sentences, spits 384 numbers. Has never seen a photo.
- **Test data:** MS-COCO val (5,000 image–caption groups), 2,500 fit / 2,500 held out.

**Result:** a single learned rotation (plus whitening, plus three fused maps, plus a re-ranker)
gets the true caption to rank #1 of 2,500 held-out candidates **56% of the time**. Raw rotation
gets 0.243. Chance is 0.0004. CLIP itself — trained end-to-end *for this duty* — gets 0.337 on
this test. The translation beats the translation-aligned model.

## Where to look

| file / dir | what it is |
|---|---|
| `experiments/01_clip_vs_dino/` | warm-up: CLIP vs DINOv2 on images (CKA 0.67, retrieval R@1 0.52) |
| `experiments/02_vision_vs_text/` | **the main event** — CLIP↔MiniLM cross-modal translation |
| `experiments/02_vision_vs_text/site/index.html` | open in a browser — the friendly story version with real COCO photos, a live 2D rotation demo, and honest failure cases |
| `experiments/02_vision_vs_text/RESULTS.md` | the full research diary: 217 logged rounds, all wins AND all dead ends (CCA, self-training, diffusion smoothing, PLS, RRR, partial whitening, …) |
| `experiments/02_vision_vs_text/results/loop.jsonl` | every single round's config + score, machine-readable |
| `embedding spaces/experiments/02_vision_vs_text/README.md` | experiment-01 and -02 md write-ups |

## The recipe, end to end

1. **Embed** 5,000 COCO images with CLIP, each image's five human captions with MiniLM
   (mean of 5 caps → 1 consensus anchor per image).
2. **Whiten** both spaces (ZCA, eps=0.01, fitted once on all rows, never per-split) —
   worth +0.19 R@1, the single biggest gain.
3. **Fit ONE rotation** (orthogonal Procrustes) on 2,500 anchor pairs: point every picture
   at its caption.
4. **Fuse three bridges** (Procrustes + ridge + InfoNCE-MLP), z-score each, weights (1.0, 0.5, 0.6).
5. **CSLS re-rank** (k=3, both sides, MUSE-2018 style): penalize "popular hub" captions.
6. Measure exact R@1 on the untouched 2,500 pairs.

## Why 100% is not possible here — measured, not argued

- ~5% of eval captions have a **near-duplicate** (cosine > 0.92) caption that belongs to a
  *different image*. No translator can split true twins.
- A linear map fitted **on the eval set itself** (unlimited cheating) tops out at 0.490.
  Every technique in the "map + cosine" class saturates below 0.6 on this data.
- The remaining gap is dataset ambiguity + small model resolution, not tuning room.

## Honest list of things that did NOT work

CCA (0.022), self-training with pseudo-anchors (0.471, unstable), graph-diffusion
re-ranking (0.326), partial whitening, iterative ridge alternation, PCA-whitening,
pair-rich caption splits, PLS, reduced-rank regression. Full detail in `RESULTS.md`.

## Reproduce

```bash
# Python 3.14 venv: embedding spaces/.venv (torch, transformers, datasets, matplotlib)
cd "experiments/02_vision_vs_text"
../../.venv/bin/python analyze.py      # step-by-step analysis of results
../../.venv/bin/python loop.py --count 200  # the 217-round harness
open site/index.html                   # the story version of all this
```

## Related work

- Platonic Representation Hypothesis (Huh et al. 2024) — big models converge; this project
  asks whether *small* ones do too (they do, but need the extra whitening).
- Relative Representations (Moschella et al. 2022) — "anchor spaces" instead of joint modes.
- Representation Alignment Rests on Linear Structure (2026) — the linear-bridge argument.
- ACT / policy stitching (Zhao et al. 2023; exp05) — deployed offline: semantic alignment
  holds; dynamics does not, without being fitted on dynamics.

## Traps encountered (documented honestly)

- MPS: `torch.linalg.eigh` silently returns garbage on float32 — do all whitening on CPU.
- Whitening with per-split matrices collapses retrieval (different bases).
- `@torch.no_grad()` around a training loop trains nothing (obvious bug, embarrassing freq).
- COCO val split only has 5,000 examples, not 25,000 like the training set.

## Status

- 217 rounds on record, 0.243 → 0.560 logged stepwise.
- Ceilings measured, both confirmed independently.
- Results and traps all written into `RESULTS.md`.
- **Exp05 (`experiments/05_robot/`)**: stitched a pretrained ACT robot policy with a self-trained
  latent world model on aloha-sim rollouts. Static frames align (R@1 0.52) and motion deltas
  partly align through the same static-only stitch (cos 0.22–0.31 at h=5–10; 0.37–0.45 with a
  non-linear stitcher) — but far less than the pictures do, and fitting a map on motion deltas
  directly fails (≈0.01–0.04). Object structure is broadly shared; one-step reaction dynamics
  are the least shared, prediction-training-specific part.
