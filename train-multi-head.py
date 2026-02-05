# train.py
# Train multi-head FIQA (quality regression for fr/gaze/kpt scores) with validation every n epochs.
#
# Expects dataset.py in the same directory providing:
#   - MultiScoreDataset
#   - build_default_transforms
#
# Logs:
#   - train/val loss
#   - per-task metrics: MAE, RMSE, Pearson, Spearman, R2
#
# Notes:
#   - "quality" head in model.py outputs a single scalar; here we train it as the *overall* quality
#     computed as mean(fr,gaze,kpt) (you can change weights).
#   - We also supervise each task head by predicting the same scalar? No:
#       - We supervise three scores by mapping model outputs:
#         quality -> overall
#         gaze head -> gaze score (scalar)
#         keypoint head -> kpt score (scalar)
#         fr embedding head -> fr score (scalar) via an extra regressor (added below)
#     Since your model.py currently returns fr_emb (embedding) and kpts (coords), we add small
#     "score projection" heads inside train.py (common in multi-task setups) without modifying model.py.

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MultiScoreDataset, build_default_transforms
from model import MultiHeadFIQA


# ------------------------- Metrics utils -------------------------

def _to_1d(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().view(-1).cpu()


def mae(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    p, t = _to_1d(pred), _to_1d(tgt)
    return float(torch.mean(torch.abs(p - t)))


def rmse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    p, t = _to_1d(pred), _to_1d(tgt)
    return float(torch.sqrt(torch.mean((p - t) ** 2) + 1e-12))


def r2_score(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    p, t = _to_1d(pred), _to_1d(tgt)
    ss_res = torch.sum((t - p) ** 2)
    ss_tot = torch.sum((t - torch.mean(t)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def pearsonr(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    p, t = _to_1d(pred), _to_1d(tgt)
    p = p - p.mean()
    t = t - t.mean()
    denom = (torch.sqrt(torch.sum(p * p)) * torch.sqrt(torch.sum(t * t)) + 1e-12)
    return float(torch.sum(p * t) / denom)


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    # simple rank with average for ties (approx via stable sort + tie grouping)
    x = x.clone()
    sorted_vals, idx = torch.sort(x)
    ranks = torch.empty_like(idx, dtype=torch.float32)
    n = x.numel()

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1].item() == sorted_vals[i].item():
            j += 1
        # average rank for ties, 1-based ranks
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[idx[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    p, t = _to_1d(pred), _to_1d(tgt)
    rp = _rankdata(p)
    rt = _rankdata(t)
    return pearsonr(rp, rt)


# ------------------------- Logging -------------------------

class SimpleLogger:
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        if self.log_path is not None:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ------------------------- Train config -------------------------

@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str

    backbone: str = "small"         # "small" or "large"
    pretrained: bool = False
    image_size: int = 224
    in_channels: int = 3

    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 50
    val_every: int = 5              # validate every n epochs

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # Multi-task loss weights
    w_quality: float = 1.0
    w_fr: float = 1.0
    w_gaze: float = 1.0
    w_kpt: float = 1.0

    # Overall quality target = weighted mean of three scores
    q_w_fr: float = 1.0
    q_w_gaze: float = 1.0
    q_w_kpt: float = 1.0

    # outputs
    out_dir: str = "./runs/fiqa_multitask"
    log_file: str = "train.log"
    save_best_name: str = "best.pt"
    save_last_name: str = "last.pt"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ------------------------- Auxiliary regressors -------------------------

class ScoreProjector(nn.Module):
    """
    Project a vector output to a scalar score in [0,1] (sigmoid).
    Used to map:
      - fr_emb (D) -> fr_score
      - kpts (K,2) -> kpt_score (flatten then MLP)
    """
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return (N,1) in [0,1]
        return torch.sigmoid(self.net(x))


# ------------------------- Core -------------------------

def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_tf = build_default_transforms(image_size=cfg.image_size, train=True)
    val_tf = build_default_transforms(image_size=cfg.image_size, train=False)

    train_ds = MultiScoreDataset(root_dir=cfg.train_dir, transforms=train_tf, strict=False)
    val_ds = MultiScoreDataset(root_dir=cfg.val_dir, transforms=val_tf, strict=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def compute_overall_quality_target(
    fr: torch.Tensor, gaze: torch.Tensor, kpt: torch.Tensor, cfg: TrainConfig
) -> torch.Tensor:
    w = torch.tensor([cfg.q_w_fr, cfg.q_w_gaze, cfg.q_w_kpt], device=fr.device, dtype=fr.dtype)
    s = torch.stack([fr, gaze, kpt], dim=1)  # (N,3)
    q = (s * w.view(1, 3)).sum(dim=1) / (w.sum() + 1e-12)
    return q  # (N,)


def loss_fn(pred: torch.Tensor, tgt: torch.Tensor, kind: str = "smoothl1") -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, tgt)
    # default smoothl1
    return F.smooth_l1_loss(pred, tgt)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    fr_proj: nn.Module,
    kpt_proj: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    fr_proj.eval()
    kpt_proj.eval()

    all_q_p, all_q_t = [], []
    all_fr_p, all_fr_t = [], []
    all_gaze_p, all_gaze_t = [], []
    all_kpt_p, all_kpt_t = [], []

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)  # (N,3)
        t_fr, t_gaze, t_kpt = y[:, 0], y[:, 1], y[:, 2]
        t_q = compute_overall_quality_target(t_fr, t_gaze, t_kpt, cfg)

        out = model(img, return_dict=True, normalize_fr=True)
        p_q = out["quality"].view(-1)                    # (N,)
        p_gaze = out["gaze"].view(img.size(0), -1)       # (N,G)
        # If gaze_out_dim>1, reduce to scalar score via mean; can be customized
        p_gaze_s = p_gaze.mean(dim=1)

        p_fr = fr_proj(out["fr_emb"]).view(-1)           # (N,)
        p_kpt = kpt_proj(out["kpts"].view(img.size(0), -1)).view(-1)  # (N,)

        lq = loss_fn(p_q, t_q)
        lfr = loss_fn(p_fr, t_fr)
        lg = loss_fn(p_gaze_s, t_gaze)
        lk = loss_fn(p_kpt, t_kpt)
        loss = cfg.w_quality * lq + cfg.w_fr * lfr + cfg.w_gaze * lg + cfg.w_kpt * lk

        total_loss += float(loss.item())
        n_batches += 1

        all_q_p.append(p_q.detach().cpu());   all_q_t.append(t_q.detach().cpu())
        all_fr_p.append(p_fr.detach().cpu()); all_fr_t.append(t_fr.detach().cpu())
        all_gaze_p.append(p_gaze_s.detach().cpu()); all_gaze_t.append(t_gaze.detach().cpu())
        all_kpt_p.append(p_kpt.detach().cpu()); all_kpt_t.append(t_kpt.detach().cpu())

    def cat(xs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=0) if len(xs) else torch.empty(0)

    q_p, q_t = cat(all_q_p), cat(all_q_t)
    fr_p, fr_t = cat(all_fr_p), cat(all_fr_t)
    g_p, g_t = cat(all_gaze_p), cat(all_gaze_t)
    k_p, k_t = cat(all_kpt_p), cat(all_kpt_t)

    metrics = {
        "loss": total_loss / max(1, n_batches),

        "q_mae": mae(q_p, q_t),
        "q_rmse": rmse(q_p, q_t),
        "q_pearson": pearsonr(q_p, q_t) if q_p.numel() >= 2 else 0.0,
        "q_spearman": spearmanr(q_p, q_t) if q_p.numel() >= 2 else 0.0,
        "q_r2": r2_score(q_p, q_t) if q_p.numel() >= 2 else 0.0,

        "fr_mae": mae(fr_p, fr_t),
        "fr_rmse": rmse(fr_p, fr_t),
        "fr_pearson": pearsonr(fr_p, fr_t) if fr_p.numel() >= 2 else 0.0,
        "fr_spearman": spearmanr(fr_p, fr_t) if fr_p.numel() >= 2 else 0.0,
        "fr_r2": r2_score(fr_p, fr_t) if fr_p.numel() >= 2 else 0.0,

        "gaze_mae": mae(g_p, g_t),
        "gaze_rmse": rmse(g_p, g_t),
        "gaze_pearson": pearsonr(g_p, g_t) if g_p.numel() >= 2 else 0.0,
        "gaze_spearman": spearmanr(g_p, g_t) if g_p.numel() >= 2 else 0.0,
        "gaze_r2": r2_score(g_p, g_t) if g_p.numel() >= 2 else 0.0,

        "kpt_mae": mae(k_p, k_t),
        "kpt_rmse": rmse(k_p, k_t),
        "kpt_pearson": pearsonr(k_p, k_t) if k_p.numel() >= 2 else 0.0,
        "kpt_spearman": spearmanr(k_p, k_t) if k_p.numel() >= 2 else 0.0,
        "kpt_r2": r2_score(k_p, k_t) if k_p.numel() >= 2 else 0.0,
    }
    return metrics


def train_one_epoch(
    model: nn.Module,
    fr_proj: nn.Module,
    kpt_proj: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    fr_proj.train()
    kpt_proj.train()

    total_loss = 0.0
    total_lq = 0.0
    total_lfr = 0.0
    total_lg = 0.0
    total_lk = 0.0
    n_batches = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)  # (N,3)
        t_fr, t_gaze, t_kpt = y[:, 0], y[:, 1], y[:, 2]
        t_q = compute_overall_quality_target(t_fr, t_gaze, t_kpt, cfg)

        out = model(img, return_dict=True, normalize_fr=True)

        p_q = out["quality"].view(-1)
        p_gaze = out["gaze"].view(img.size(0), -1)
        p_gaze_s = p_gaze.mean(dim=1)  # scalar gaze-score proxy

        p_fr = fr_proj(out["fr_emb"]).view(-1)
        p_kpt = kpt_proj(out["kpts"].view(img.size(0), -1)).view(-1)

        lq = loss_fn(p_q, t_q)
        lfr = loss_fn(p_fr, t_fr)
        lg = loss_fn(p_gaze_s, t_gaze)
        lk = loss_fn(p_kpt, t_kpt)

        loss = cfg.w_quality * lq + cfg.w_fr * lfr + cfg.w_gaze * lg + cfg.w_kpt * lk

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(fr_proj.parameters()) + list(kpt_proj.parameters()),
                max_norm=cfg.grad_clip,
            )

        optimizer.step()

        total_loss += float(loss.item())
        total_lq += float(lq.item())
        total_lfr += float(lfr.item())
        total_lg += float(lg.item())
        total_lk += float(lk.item())
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "loss_q": total_lq / max(1, n_batches),
        "loss_fr": total_lfr / max(1, n_batches),
        "loss_gaze": total_lg / max(1, n_batches),
        "loss_kpt": total_lk / max(1, n_batches),
    }


def save_ckpt(
    path: str,
    model: nn.Module,
    fr_proj: nn.Module,
    kpt_proj: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model": model.state_dict(),
            "fr_proj": fr_proj.state_dict(),
            "kpt_proj": kpt_proj.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(str(out_dir / cfg.log_file))

    device = torch.device(cfg.device)

    train_loader, val_loader = build_loaders(cfg)

    # Base model
    model = MultiHeadFIQA(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        in_channels=cfg.in_channels,
    ).to(device)

    # Build score projectors:
    #   fr_emb: D -> 1
    #   kpts: (K,2) -> 1
    # Infer dims from a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(1, cfg.in_channels, cfg.image_size, cfg.image_size).to(device)
        out = model(dummy, return_dict=True)
        fr_dim = out["fr_emb"].shape[1]
        kpt_dim = int(out["kpts"].numel())  # 1*K*2

    fr_proj = ScoreProjector(in_dim=fr_dim, hidden=max(128, fr_dim // 2)).to(device)
    kpt_proj = ScoreProjector(in_dim=kpt_dim, hidden=max(256, kpt_dim // 4)).to(device)

    # Optimizer
    params = list(model.parameters()) + list(fr_proj.parameters()) + list(kpt_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Scheduler (cosine)
    total_steps = cfg.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    best_val = float("inf")  # best by val loss (lower is better)

    logger.log(f"Device: {device}")
    logger.log(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    logger.log(f"Total epochs: {cfg.epochs}, validate every {cfg.val_every} epoch(s)")
    logger.log(f"Output dir: {out_dir}")

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_stats = train_one_epoch(model, fr_proj, kpt_proj, train_loader, optimizer, cfg, device)

        # step scheduler per-iteration count (approx using batch loop length)
        # Here we step by number of batches (CosineAnnealingLR expects per-step calls if T_max=steps)
        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1

        lr_now = optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"lr={lr_now:.6g} | "
            f"train_loss={train_stats['loss']:.5f} "
            f"(q={train_stats['loss_q']:.5f}, fr={train_stats['loss_fr']:.5f}, "
            f"gaze={train_stats['loss_gaze']:.5f}, kpt={train_stats['loss_kpt']:.5f}) | "
            f"time={time.time() - t0:.1f}s"
        )
        logger.log(msg)

        # Validate
        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.epochs)
        if do_val:
            val_metrics = evaluate(model, fr_proj, kpt_proj, val_loader, cfg, device)

            logger.log(
                "VAL | "
                f"loss={val_metrics['loss']:.5f} | "
                f"q: MAE={val_metrics['q_mae']:.4f}, RMSE={val_metrics['q_rmse']:.4f}, "
                f"PLCC={val_metrics['q_pearson']:.4f}, SRCC={val_metrics['q_spearman']:.4f}, R2={val_metrics['q_r2']:.4f} | "
                f"fr: MAE={val_metrics['fr_mae']:.4f}, RMSE={val_metrics['fr_rmse']:.4f}, "
                f"PLCC={val_metrics['fr_pearson']:.4f}, SRCC={val_metrics['fr_spearman']:.4f}, R2={val_metrics['fr_r2']:.4f} | "
                f"gaze: MAE={val_metrics['gaze_mae']:.4f}, RMSE={val_metrics['gaze_rmse']:.4f}, "
                f"PLCC={val_metrics['gaze_pearson']:.4f}, SRCC={val_metrics['gaze_spearman']:.4f}, R2={val_metrics['gaze_r2']:.4f} | "
                f"kpt: MAE={val_metrics['kpt_mae']:.4f}, RMSE={val_metrics['kpt_rmse']:.4f}, "
                f"PLCC={val_metrics['kpt_pearson']:.4f}, SRCC={val_metrics['kpt_spearman']:.4f}, R2={val_metrics['kpt_r2']:.4f}"
            )

            # Save best by val loss
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_ckpt(
                    str(out_dir / cfg.save_best_name),
                    model,
                    fr_proj,
                    kpt_proj,
                    optimizer,
                    epoch=epoch,
                    best_metric=best_val,
                )
                logger.log(f"Saved BEST checkpoint (val_loss={best_val:.5f}) -> {out_dir / cfg.save_best_name}")

        # Always save last
        save_ckpt(
            str(out_dir / cfg.save_last_name),
            model,
            fr_proj,
            kpt_proj,
            optimizer,
            epoch=epoch,
            best_metric=best_val,
        )

    logger.log("Training done.")


if __name__ == "__main__":
    # Edit these two paths:
    #   train_dir: contains labels.txt and images/
    #   val_dir: contains labels.txt and images/
    cfg = TrainConfig(
        train_dir="./train_dir",
        val_dir="./val_dir",
        # val_every=5, epochs=50, ...
    )
    main(cfg)

