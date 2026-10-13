"""Exp 03: zero-shot plug-in — project DINOv2 image space into CLIP's image space,
then let CLIP's own TEXT head classify the image (independent models, one rotation)."""
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent
dev = "mps" if torch.backends.mps.is_available() else "cpu"
NRM = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
PROMPT_TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a photo of the {}.",
                    "a cropped photo of a {}.", "a photo of one {}."]

@torch.no_grad()
def clip_text_embeds():
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    mdl = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    texts = [t.format(c) for c in CIFAR_CLASSES for t in PROMPT_TEMPLATES]
    enc = tok(texts, padding=True, return_tensors="pt").to(dev)
    E = mdl(**enc).text_embeds.cpu()
    return NRM(E.reshape(10, len(PROMPT_TEMPLATES), -1).mean(1))   # [10, 512]

def predict(img_proj, proto):
    return (NRM(img_proj) @ proto.T).argmax(-1)

def main():
    tr = torch.load(ROOT / "results" / "cifar_train.pt", map_location="cpu")
    te = torch.load(ROOT / "results" / "cifar_test.pt", map_location="cpu")
    proto = clip_text_embeds()
    yte = te["labels"]
    torch.manual_seed(0)

    # (a) CLIP native zero-shot
    acc_native = (te["clip"] @ proto.T).argmax(-1).eq(yte).float().mean().item()

    # (b) fit whitened procruste DINO->CLIP on 10k training images
    n_fit = 10000
    Xtr = tr["dino"][:n_fit]; Ytr = tr["clip"][:n_fit]
    def wh(Z, eps=0.01):
        mu = Z.mean(0, keepdim=True); Zc = Z - mu
        C = Zc.T @ Zc / len(Zc)
        ev, V = torch.linalg.eigh(C)          # CPU — MPS eigh is broken
        ev = ev.clamp_min(eps)
        W = (V * ev.rsqrt()) @ V.T.clamp(0).to(torch.float32) if False else (V * ev.rsqrt()) @ V.T
        return (Z - mu) @ W
    Xw, Yw = wh(Xtr), wh(Ytr)
    M = NRM(Xw).T @ NRM(Yw)
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    din_wte = NRM(wh(te["dino"]))
    acc_trans = predict(NRM(din_wte @ Q), proto).eq(yte).float().mean().item()

    # (c) random rotation control
    G = torch.randn(384, 512)
    U1, _, Vt1 = torch.linalg.svd(G, full_matrices=False)
    R1 = U1 @ Vt1                      # 384 x 512 partial isometry
    acc_rand = predict(NRM(NRM(te["dino"]) @ R1), proto).eq(yte).float().mean().item()

    print({
        "clip_native_zero_shot": round(acc_native, 4),
        "dino_translated_zero_shot": round(acc_trans, 4),
        "random_rotation_control": round(acc_rand, 4),
        "chance": 0.10,
    })

if __name__ == "__main__":
    main()
