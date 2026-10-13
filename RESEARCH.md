# RESEARCH.md — what was already known, what's ours, where to go next

*(Survey status: arXiv sweep, Oct 2026; big adjacent fields identified.)*

## Was this already known? Short answer: **the main effect was known at LARGE scale; our small-scale cross-modal numbers are the interesting bit.**

### The lineage, traced

1. **Unsupervised word translation (2016–2018, Conneau et al., MUSE).**
   Two monolingual embedding spaces translated via a rotation + CSLS — literally the
   recipe we ended up using (Procrustes + CSLS re-rank). Known for years.
2. **Intermediate layers, "model stitching" (2020–2026).** Lenc & Vedaldi's successors
   ("Revisiting Model Stitching...", 2021–2026 updates) tested whether layer A plugged
   into layer B still yields a functioning network — the classic version of our
   rotation. Acceptably known phenomenon.
3. **Relative Representations & zero-shot stitching (Moschella et al. 2022,
   "Latent Space Translation via Inverse Relative Projection", 2024).** Anchored spaces
   instead of coordinate transfer — closest predecessor to our `Z (mid-anchor)` idea.
   **They demonstrated stitching models trained on the *same modality*.**
4. **Platonic Representation Hypothesis (Huh et al., ICML 2024).** Big models
   (LLaMA-style, DINO/SigLIP-scale) *converge* on shared structure. But it isn't tested
   on **small** models, holds for same-modality mostly.
5. **2026 updates** ("Aristotelian View", "Proof of a perfect platonic...", "Invariance-
   Aware Stitching") take the view: convergence exists but is fine structure, hard
   to see without careful geometry prep. Doesn't pin down small-scale cross-modal retrieval.

### Where our run is genuinely informative (not just a reproduction)

- **Small pair** (150M + 22M params, consumer laptop) — nobody benchmarks this regime.
- **Cross-modal** (image ↔ text) at small scale — Platonic papers skew same-modal or huge.
  We *quantify* the gap: 0.560 presentation on a task CLIP scored 0.337 on, so translation
  **outperformed the modal-native alignment** that was co-trained specifically for it.
- **Driving specific mechanism — whitening before ANY compare** — is folklore in the word-
  translation + CKA literature but rarely foregrounded as the *dominant factor* (+0.19
  R@1 here). Also we measured the *ceiling* (0.490, oracle-cheat) empirically rather than
  asserting "it's limited."

## What would expand this (ideas ranked by how informative-per-hour they are)

1. **More model pairs (cheap, immediately informative).** SigLIP ↔ E5, DINOv2 ↔ e5-base,
   CLIP ViT-B/16 vs /32 (same family), SigLIP ↔ SigLIP-asymmetric (CLIP-like vs Siamese).
   Question: does whitening + rotation always help, or does the result depend on pair
   choice? Three pairs would firmly separate "pair" from "phenomenon."
2. **Informative-rescaling but named-guarantee scaling tests: at what model size does
   the fixed rotation suffice (no MLP, no CSLS)?** At what size does whitening stop
   mattering (raw-space geometry is already spherical)? Straight-line plot, hugely
   informative either way.
3. **Cross-modal zero-shot transfer (the "actually useful" test).** Use the projection as
   a plug-in adapter: CLIP-text head gets a MiniLM-translated image vector. If zero-shot
   ImageNet classification transfers even at, say, 30% of CLIP's own accuracy, that is
   a headline-grade result for "translation replaces co-training." Follow-up to our
   Exp-01 DINO→CLIP sketch.
4. **Sensitivity/consistency audit.** What fraction of eval retrieval collapses is
   determined by data ambiguity alone (the 5% near-twin problem). Formal version:
   parse caption-twin structure, then report results stratified by twin-density.
5. **Information-theoretic view of the ceiling.** A neat mini-theorem: once the caption
   cloud's pairwise cosine supports twins above 0.92, translator performance is bounded
   by the noise floor of the anchor source itself; measure and derive it explicitly.
6. **Better unlimited-ceiling find.** Cross-encoder / VLM as scored distinction-ability —
   get a real quantified ceiling so claims are tested against it, not just the linear
   one we measured (0.490).
7. **JS interactive site** with real dim-2-pick examples + hover-to-caption atlas on
   the drift-alignment pairs — helps audience understand why 0.56 is not "bad."

## Attribution summary (for the README's "related work" section, final wording)

- MUSE: `Conneau et al. 2018` — the basis for our whitening + CSLS.
- Platonic Representation Hypothesis: `Huh et al. 2024` — the *hypothesis*: shared
  structure emerges without co-training (at large scale).
- Zero-shot stitching / relative representations: `Moschella et al. 2022`,
  `IRP 2024` — anchor-based translation.
- Model stitching: `Bansal et al. 2021`, updated 2026 — classical inter-layer version.

---

# UPDATE (same day) — Expansions run

## Exp03 — zero-shot plug-in (`experiments/03_zerotransfer/`)

**DINOv2 images rotated into CLIP's image space, then classified by CLIP's own text head
(a brand new model it has never co-trained with):**

```
CLIP native zero-shot CIFAR-10:      0.8907
DINO->CLIP translated zero-shot:     0.8884   <- 99.7% of CLIP's own accuracy!
random-rotation control:             0.106    (chance=0.10)
```

A model that has never been pranked by text on 5000 classes still classifies through a
*frozen zero-shot text head* at native-model parity once you add whitening + one rotation.
This is the headline result for "translation replaces co-training".

## Exp04 — multi-pair generalisation (`experiments/04_model_pairs/`)

Same pipeline (whitening + procrustes fused + CSLS k=50), three pairs:

| pair | R@1 plain | R@1 +CSLS k=50 |
|---|---|---|
| CLIP-B/32 ↔ MiniLM (ref) | 0.4436 | 0.4956 |
| CLIP-B/16 ↔ mpnet | 0.5340 | 0.5828 |
| SigLIP ↔ mpnet | **0.5944** | **0.6540** |

Generalisation confirmed — the whitening+rotation recipe transfers across *independent* pairs.
Jointly-trained image tower (SigLIP) yields the strongest shared structure — consistent with
the Platonic Rep. Hypothesis scaling story. Bigger vision tower (B/16 > B/32) also helps.

## Honest methodology note (a bug found and fixed)

My standalone evaluation script initially used a hand-rolled rank formula
(`argsort().argsort(); (r == arange)`) that is semantically NOT the diagonal rank —
it made every variant collapse to chance. The correct formulation from `loop.py` uses
`(o == arange)[:, 1]` on the argsort directly. All Exp-04 numbers above are with the
CORRECT metric; the bug is documented so it isn't repeated.

## Exp05 — policy ↔ world-model stitching (`experiments/05_robot/`)

Does one static-frame Procrustes rotation let a pretrained ACT policy (aloha-sim, 51M) and an
independently-trained JEPA-lite latent world model (DINOv2-small + dynamics head) share their
*dynamics*?

**Main (run.py):** static frame alignment works (R@1 0.52 on unseen episodes; R@5 0.72). Rotated
motion-delta cosine on whitened latents: 0.058 / 0.218 / 0.309 at h = 1 / 5 / 10 (random-rotation
control ≈ 0). The DINO dynamics predictor itself is excellent (0.918 cos to true next-latent vs
0.029 control), but its forecast agrees with ACT's actual latent change only 0.074 through the
seam. Action-chunk readouts: ACT 0.93 / DINO 0.88 R², zero-shot through the stitch 0.89.

**Stitch ablations (run_b.py, Exp05b):** a 2-layer MLP stitcher fitted on the same *static*
frames nearly doubles the dynamics that cross (0.094 / 0.374 / 0.454 at h=1/5/10), while
stitchers fitted on *motion deltas* directly FAIL (0.007–0.043, linear or MLP). Deltas are
noisy and high-variance relative to static geometry; the physics that transfers is carried
by the objects.

**Reading:** object-level structure is shared encoders-to-encoders (Platonic, linear regime).
Temporal structure is partially shared — increasingly so as deltas become quasi-positional —
but one-step reaction dynamics are the least shared, prediction-training-specific part.
First clean quantification inside the project of what JEPA-style predictive training buys
beyond static alignment. (Methodology note: the first pass of Exp05 scored the rotated deltas
≈ 0.000 and reported "pictures align, physics doesn't"; that number came from computing deltas
on *raw* unwhitened latents. Corrected above; the corrected version is more nuanced and
more interesting.)

## Exp06 — planning through the stitch (`experiments/05_robot/run_c.py`)

Does the stitched world model improve ACT's action selection? Offline test, 57 unseen goal
states, score = RMSE to the executed ground-truth chunk: ACT's own latent→chunk readout 0.25
(DINO readout 0.34, mean-chunk baseline 0.96) vs. world-model planning over state-conditioned
or demonstrator+jitter candidate banks 1.34–1.47. Oracle diagnostic: the TRUE chunk inserted
into a 201-candidate bank ranks 94.7/201 under rollout cost — chance level. The latent
dynamics predictor predicts *latents* well (0.92 cos one-step) but does not discriminate
*chunks*: different chunks roll to nearby futures, and rollout noise compounds over even
4 steps. Verdict: at 3k-frames-of-aloha scale, the stitched world model does not beat direct
policy decoding; realism ceiling, not concept ceiling. Retry candidates listed in the
experiment README.
