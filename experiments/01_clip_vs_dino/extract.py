"""
Experiment 01 — embedding-space alignment: CLIP ViT-B/32 vs DINOv2 ViT-S/14.

Extracts penultimate features for both models on the CIFAR-10 test set and
saves them + labels as tensors under results/.

Run:  python extract.py [--n 10000] [--batch 128]
"""
import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import CLIPVisionModelWithProjection, Dinov2Model

HERE = Path(__file__).parent
OUT = HERE / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN_DINO = [0.485, 0.456, 0.406]
STD_DINO = [0.229, 0.224, 0.225]


def load_models():
    clip = (
        CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        .to(DEVICE).eval()
    )
    dino = Dinov2Model.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()
    return clip, dino


@torch.no_grad()
def extract(dataset, n, batch):
    import numpy as np
    hs = torch.nn.functional.normalize  # shorthand
    clip_feats, dino_feats, labels = [], [], []
    clip_m, dino_m = load_models()
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    for i in range(0, n, batch):
        chunk = dataset.select(range(i, min(i + batch, n)))
        imgs = [x["img"].convert("RGB") for x in chunk]
        lbl = chunk["label"]

        # CLIP preprocessing (224, official normalization)
        px = proc(images=imgs, return_tensors="pt", do_rescale=True).to(DEVICE)
        c = clip_m(**px).image_embeds  # projected -> 512-d
        clip_feats.append(c.cpu())
        labels.extend(lbl)

        # DINOv2 preprocessing
        imgs224 = torch.stack([
            (torch.from_numpy(np.array(im.resize((224, 224))))
             .float().permute(2, 0, 1) / 255.0) for im in imgs
        ])
        mean = torch.tensor(MEAN_DINO); std = torch.tensor(STD_DINO)
        imgs224 = (imgs224 - mean[:, None, None]) / std[:, None, None]
        d = dino_m(pixel_values=imgs224.to(DEVICE)).last_hidden_state[:, 0]  # CLS 384-d
        dino_feats.append(d.cpu())
        print(f"  {i + len(imgs)}/{n}", flush=True)
    return hs(torch.cat(clip_feats)), hs(torch.cat(dino_feats)), torch.tensor(labels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("uoft-cs/cifar10", split="test")
    clip_f, dino_f, y = extract(ds, a.n, a.batch)
    torch.save({"clip": clip_f, "dino": dino_f, "labels": y}, OUT / "embeddings.pt")
    print("saved:", OUT / "embeddings.pt", "| clip", clip_f.shape, "| dino", dino_f.shape)
