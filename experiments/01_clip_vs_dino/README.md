# Experiment 01: CLIP ViT-B/32 vs DINOv2 ViT-S/14

**Question:** do two image encoders trained with radically different
objectives (contrastive language-image vs self-supervised distillation)
learn aligned embedding spaces?

**Models:**
- `openai/clip-vit-base-patch32` — CLIP, ~150M params (vision tower)
- `facebook/dinov2-small` (~22M params, ViT-S/14, self-supervised, no text)

**Anchor set:** CIFAR-10 (10k test images) — same raw inputs to both models.

## Plan

1. Extract penultimate features from both models on 2-10k images.
2. Measure raw agreement (no projection):
   - Linear & RBF CKA
   - Procrustes similarity on the pairwise distance structures
   - retrieval overlap for each image's top-k neighbors across the two spaces
3. Fit a projection from DINO->CLIP space (Orthogonal Procrustes, ridge, CCA)
   and evaluate cross-space retrieval.
4. Visuals: CKA heatmaps across layer pairs, neighbor-overlap vs k curve.

## Results

_(fill in as we run)_
