import os
if "/opt/homebrew/opt/ffmpeg@7/lib" not in os.environ.get("DYLD_LIBRARY_PATH", ""):
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/ffmpeg@7/lib"
"""Extract aligned (frame, state, action-chunk) corpus from lerobot/aloha_sim_save_cube_human.
Caches: images (uint8 480x640), states, action chunks, episode ids, frame indices.
"""
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
N_TARGET = 3000

def main():
    ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
    n = ds.meta.total_frames
    rng = np.random.default_rng(0)
    picks = np.sort(rng.choice(n, size=N_TARGET, replace=False))
    imgs, states, chunks, epids, fids, dones = [], [], [], [], [], []
    for j, i in enumerate(picks):
        f = ds[int(i)]
        ep = int(f["episode_index"])
        # gather next-20 action chunk from within same episode
        cs = []
        for k in range(20):
            g = ds[int(i) + k] if int(i) + k < n else None
            if g is None or int(g["episode_index"]) != ep: break
            cs.append(g["action"].float())
        if len(cs) < 4: continue
        imgs.append(f["observation.images.top"].clone())
        states.append(f["observation.state"].float())
        chunks.append(torch.stack(cs))
        epids.append(ep); fids.append(int(i))
        dones.append(torch.tensor(f["next.done"], dtype=torch.float32))
        if j % 200 == 0: print(j, "/", N_TARGET, flush=True)
    torch.save({
        "img": torch.stack(imgs),          # [N,3,480,640] uint8
        "state": torch.stack(states),      # [N,14]
        "chunk": chunks,                   # listiến of [<=20,14]
        "episode": torch.tensor(epids), "frame": torch.tensor(fids),
        "done": torch.stack(dones),
    }, OUT / "corpus.pt")
    print("saved", OUT / "corpus.pt", len(imgs), "frames")

if __name__ == "__main__":
    main()
