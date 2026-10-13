"""DINOv2-small CLS latents for the aloha corpus. Run with main project env (transformers 5.x)."""
import warnings; warnings.filterwarnings("ignore")
import torch
from pathlib import Path
HERE = Path(__file__).parent
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()

c = torch.load(HERE / "results" / "corpus.pt")
imgs = c["img"]  # [N,3,480,640] uint8
feat = torch.nn.functional.adaptive_avg_pool2d if True else None
out = []
with torch.no_grad():
    for s in range(0, len(imgs), 64):
        x = imgs[s:s+64].float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True).to(DEV)
        h = m(x).last_hidden_state[:, 0]  # CLS
        out.append(h.cpu())
z = torch.cat(out)
torch.save(z.float(), HERE / "results" / "dino.pt")
print("dino latents:", z.shape)
