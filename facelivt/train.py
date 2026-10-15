"""Train FaceLiVTv2 on CelebA with CosFace."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from facelivt.data.celeba import CelebAIdDataset, resolve_celeba_root
from facelivt.models.facelivt_net import build_facelivt, count_params


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    train_ds = CelebAIdDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 128)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = build_facelivt(
        train_ds.num_classes,
        variant=str(cfg["model"].get("variant", "xs")),
        s=float(cfg["model"].get("cosface_s", 16.0)),
        m=float(cfg["model"].get("cosface_m", 0.25)),
    ).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    epochs = int(cfg["train"].get("epochs", 12))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        pbar = tqdm(loader, desc=f"facelivt {epoch}/{epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(images, labels)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        sched.step()
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss, "lr": float(sched.get_last_lr()[0])})
        print(f"[facelivt] epoch={epoch} loss={mean_loss:.4f} lr={history[-1]['lr']:.6f} params={n_params}")

    elapsed = time.time() - t0
    torch.save(
        {
            "model": model.state_dict(),
            "num_classes": train_ds.num_classes,
            "variant": cfg["model"].get("variant", "xs"),
            "n_params": n_params,
            "config": cfg,
        },
        ckpt_dir / "last.pt",
    )
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "epochs": epochs,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "n_params": n_params,
        "variant": cfg["model"].get("variant", "xs"),
        "elapsed_sec": elapsed,
        "history": history,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s params={n_params}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
