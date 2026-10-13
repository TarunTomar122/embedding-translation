"""Exp 04 extract: extra (image-tower, text-encoder) pairs on the same 5000 COCO images.

Pairs (all image towers independent of the text encoder used):
  B: SigLIP-base-patch16-224 image tower  <->  all-mpnet-base-v2 (text-only, 110M)
  C: CLIP ViT-B/16 image tower            <->  MiniLM-L6 (same text as Exp02)
Reuses cached COCO images from Exp02 extract via the HF dataset again.
"""
import itertools, json, sys, torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, SiglipImageProcessor, AutoTokenizer, AutoModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ROOT = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
normt = torch.nn.functional.normalize

def embed_images(model_pair, imgs_list):
    clip16 = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).eval()
    proc16 = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    siglip = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    outs = {"clip16": [], "siglip": []}
    mdl_s = None
    try:
        from transformers import SiglipVisionModel
        mdl_s = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE).eval()
    except Exception as e:
        print("siglip unavailable:", e, flush=True)
    with torch.no_grad():
        for i in range(0, len(imgs_list), 32):
            chunk = imgs_list[i:i+32]
            px = proc16(images=chunk, return_tensors="pt", do_rescale=True).to(DEVICE)
            outs["clip16"].append(clip16(**px).image_embeds.cpu())
            if mdl_s is not None:
                ps = siglip(images=chunk, return_tensors="pt", do_rescale=True).to(DEVICE)
                outs["siglip"].append(mdl_s(**ps).pooler_output.cpu())
            if (i//32) % 20 == 0: print(f"img {i}/{N}", flush=True)
    outs["clip16"] = normt(torch.cat(outs["clip16"]))
    if outs["siglip"]: outs["siglip"] = normt(torch.cat(outs["siglip"]))
    return outs

def embed_texts_mpnet(sents):
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    mdl = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sents), 64):
            b = sents[i:i+64]
            enc = tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            o = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = (o * mask).sum(1) / mask.sum(1)          # mean-pool
            out.append(normt(emb).cpu())
            if (i//64) % 40 == 0: print(f"txt {i}/{len(sents)}", flush=True)
    return torch.cat(out)

if __name__ == "__main__":
    ds = load_dataset("Multimodal-Fatima/COCO_captions_validation", split="validation", streaming=True)
    imgs, caps = [], []
    for ex in itertools.islice(ds, N):
        imgs.append(ex["image"].convert("RGB"))
        caps.append([c.strip() for c in ex["sentences_raw"][:5]])
    im = embed_images(None, imgs)
    flat = [c for grp in caps for c in grp]
    mpnet_flat = embed_texts_mpnet(flat)
    mpnet = mpnet_flat.reshape(len(caps), 5, -1)
    torch.save({
        "clip16": im.get("clip16"), "siglip": im.get("siglip"),
        "mpnet_mean5": normt(mpnet.mean(1)), "mpnet_cap1": mpnet[:, 0],
    }, ROOT / "results" / "pairs.pt")
    json.dump(caps, open(ROOT / "results" / "captions.json", "w"))
    print("saved", {k: (v.shape if torch.is_tensor(v) else None) for k, v in torch.load(ROOT/"results"/"pairs.pt").items()})
