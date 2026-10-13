# Experiment 02: CLIP image tower vs MiniLM text embeddings

**Question:** do a vision encoder and a pure-text model — trained on nothing
shared, with no contrastive bond between them — encode the *same concept* for
the same content?

**Models:**
- `openai/clip-vit-base-patch32` — CLIP image tower (~88M visual params)
- `sentence-transformers/all-MiniLM-L6-v2` (~22M BERT-family text embedder,
  trained on completely different data/objective)

**Anchor set:** COCO validation images, each paired with the mean of its
5 human captions embedded by MiniLM. ~4k pairs.

## Tests

1. Raw structure agreement: CKA (linear/rbf), centroid-structure correlation
2. Cross-model retrieval: CLIP image embed query → MiniLM text space via
   orthogonal Procrustes fitted on disjoint train half. Recall@k.
3. Baselines: same-experiment with *shuffled* pairing (should collapse to
   chance) and CLIP→CLIP (ceiling).

## Prior art check (from arXiv survey, 2026-10)

- Platonic Representation Hypothesis (Huh 2024): CLIP↔BERT alignment at
  foundation scale — our replicated-method reference.
- Relative representations (Moschella 2022): zero-shot stitching.
- "What Converges in the PRH" (2026): structure conserved, geometry not.
- Open niche: small-model scale (<2B) + failure-mode mapping.

## Results

Done — see RESULTS.md. Highlights (n=5000 COCO anchors, held-out):

- Raw CKA surface agreement 0.65–0.68 (structure shared with NO fitting)
- Raw orthogonal Procrustes retrieval: R@1 = 0.25
- **Best: ZCA-whiten (eps 0.01) + orthogonal Procrustes → R@1 ≈ 0.43±0.01,
  R@10 ≈ 0.85, MRR ≈ 0.57** (chance R@1 = 0.0004)
- Nonlinear bridges win on the raw sphere but LOSE after whitening → residual
  misalignment is genuine, not scale artifacts
- CCA: catastrophic overfit at this scale (R@1 ≈ 0.01)
- Iterative ridge/procrustes alternation: dead end
- Anchor quality > count: mean-of-5 captions beats 4x more noisy pairs
