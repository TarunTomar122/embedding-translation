"""Extract CLIP + DINOv2 embeddings for CIFAR-10 test AND train splits (zero-shot plug-in exp)."""
import sys, torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import CLIPVisionModelWithProjection, Dinov2Model, CLIPImageProcessor

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path(__file__).parent / "results"
MEAN_DINO = (0.480, 0.456, 0.406); STD_DINO = (0.268, 0.258, 0.276)
normt = torch.nn.functional.normalize

@torch.no_grad()
def extract(ds, batch=64):
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    dino = Dinov2Model.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    cf, df, y = [], [], []
    for i in range(0, len(ds), batch):
        chunk = ds.select(range(i, min(i + batch, len(ds))))
        imgs = [x["img"].convert("RGB") for x in chunk]
        px = proc(images=imgs, return_tensors="pt", do_rescale=True).to(DEVICE)
        cf.append(clip(**px).image_embeds.cpu())
        xs = torch.stack([(torch.from_numpy(np.array(im.resize((224, 224)))).float().permute(2, 0, 1) / 255.) for im in imgs])
        xs = (xs - torch.tensor(MEAN_DINO)[:, None, None]) / torch.tensor(STD_DINO)[:, None, None]
        df.append(dino(pixel_values=xs.to(DEVICE)).last_hidden_state[:, 0].cpu())
        y.extend(x["label"] for x in chunk)
        if (i // batch) % 20 == 0: print(f"{i}/{len(ds)}", flush=True)
    return normt(torch.cat(cf)), normt(torch.cat(df)), torch.tensor(y)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "test"
    ds = load_dataset("uoft-cs/cifar10", split=which)
    c, d, y = extract(ds)
    torch.save({"clip": c, "dino": d, "labels": y}, OUT / f"cifar_{which}.pt")
    print("saved", OUT / f"cifar_{which}.pt", c.shape)
