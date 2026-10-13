"""
Experiment 02 — extract CLIP image embeddings and MiniLM caption embeddings
for the same COCO images. Saves results/embeddings.pt with:
  clip   [N, 512]  normalized image embeddings (CLIP ViT-B/32 tower)
  mini   [N, 384]  normalized mean-of-5-captions embeddings (MiniLM-L6-v2)
  labels []        cocoid (分组无所谓, information only)

Run: python extract.py [--n 4000] [--batch 32]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, AutoTokenizer, AutoModel

OUT = Path(__file__).parent / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

@torch.no_grad()
def mean_pool(model_output, attention_mask):
    tok = model_output[0]
    m = attention_mask.unsqueeze(-1).float()
    return (tok * m).sum(1) / m.sum(1).clamp(min=1e-9)

@torch.no_grad()
def main(n, batch):
    OUT.mkdir(exist_ok=True)
    print("loading models...")
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    mini = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE).eval()

    ds = load_dataset("Multimodal-Fatima/COCO_captions_validation", split="validation", streaming=True)
    CL, MI = [], []
    buf_img, buf_txt, groups = [], [], []
    count = 0
    groups = []  # (n_caps) per image, for caption averaging
    def flush():
        nonlocal buf_img, buf_txt, groups
        if not buf_img: return
        px = proc(images=buf_img, return_tensors="pt").to(DEVICE)
        cl = clip(**px).image_embeds
        flat = [c for caps in buf_txt for c in caps]
        groups = [len(caps) for caps in buf_txt]
        enc = tok(flat, padding=True, truncation=True, max_length=75, return_tensors="pt").to(DEVICE)
        mi = F.normalize(mean_pool(mini(**enc), enc["attention_mask"]), dim=1)  # [sum(groups), 384]
        # mean of per-caption embeddings -> one vector per image
        mi = torch.stack([mi[s:s+g].mean(0) for s, g in zip(np.cumsum([0]+groups[:-1]), groups)])
        CL.append(F.normalize(cl.float().cpu()))
        MI.append(F.normalize(mi.float().cpu()))
        buf_img, buf_txt = [], [],

    for ex in ds:
        caps = ex["sentences_raw"][:5]
        buf_img.append(ex["image"].convert("RGB"))
        buf_txt.append(caps)
        count += 1
        if count % batch == 0:
            flush()
            print(f"  {count}", flush=True)
        if count >= n: break
    flush()

    clip_f, mini_f = torch.cat(CL), torch.cat(MI)
    torch.save({"clip": clip_f, "mini": mini_f}, OUT / "embeddings.pt")
    print("saved:", clip_f.shape, mini_f.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    main(a.n, a.batch)
