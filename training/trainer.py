"""
Training loop with validation, checkpointing, early stopping, resume and history.

Model selection uses the VALIDATION split only (best = highest val NC, ties by
lower val loss).  The test split is never touched here.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.blind_extractor import BlindWatermarkExtractor, build_model, count_parameters
from training.losses import WatermarkLoss
from training.metrics import batch_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "models" / "checkpoints"


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, epoch: int = 0, history=None,
                    best_metric: float = -1.0, train_config: Optional[dict] = None, extra: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state": model.state_dict(), "model_config": model.config, "epoch": epoch,
             "history": history or [], "best_metric": best_metric, "train_config": train_config or {},
             "optimizer_state": optimizer.state_dict() if optimizer else None,
             "scheduler_state": scheduler.state_dict() if scheduler else None,
             "torch_version": torch.__version__, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             **(extra or {})}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device = torch.device("cpu")):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain a model first: python run.py --phase ann-train --quick")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device) -> Dict[str, float]:
    model.eval()
    tot, n, acc = 0.0, 0, {}
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot += loss_fn(out, y).item() * x.shape[0]
        n += x.shape[0]
        for k, v in batch_metrics(out, y).items():
            acc[k] = acc.get(k, 0.0) + v.sum().item()
    return {"loss": tot / max(n, 1), **{k: v / max(n, 1) for k, v in acc.items()}}


class Trainer:
    def __init__(self, train_ds, val_ds, epochs: int = 15, batch_size: int = 16, lr: float = 2e-3,
                 weight_decay: float = 1e-4, patience: int = 5, nc_weight: float = 0.1, seed: int = 42,
                 device: str = "auto", ckpt_dir: Path = CKPT_DIR, results_dir: Optional[Path] = None,
                 model_cfg: Optional[dict] = None, num_threads: Optional[int] = None,
                 log: Callable[[str], None] = print, max_train_batches: Optional[int] = None):
        set_seed(seed)
        if num_threads or (os.cpu_count() and not torch.cuda.is_available()):
            torch.set_num_threads(num_threads or os.cpu_count())
        self.device = get_device(device)
        self.cfg = dict(epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay, patience=patience,
                        nc_weight=nc_weight, seed=seed, device=str(self.device), optimizer="AdamW",
                        scheduler="ReduceLROnPlateau(mode=max on val NC, factor=0.5, patience=2)",
                        loss=f"BCEWithLogits + {nc_weight} * (1 - softNC)", selection="max val NC",
                        n_train=len(train_ds), n_val=len(val_ds), max_train_batches=max_train_batches)
        g = torch.Generator(); g.manual_seed(seed)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        self.model: BlindWatermarkExtractor = build_model(model_cfg).to(self.device)
        self.loss_fn = WatermarkLoss(nc_weight)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="max", factor=0.5, patience=2)
        self.ckpt_dir = Path(ckpt_dir)
        self.results_dir = Path(results_dir) if results_dir else None
        self.log = log
        self.max_train_batches = max_train_batches
        self.history, self.start_epoch, self.best, self.bad_epochs = [], 0, -1.0, 0

    def resume(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        if ck.get("optimizer_state"):
            self.opt.load_state_dict(ck["optimizer_state"])
        if ck.get("scheduler_state"):
            self.sched.load_state_dict(ck["scheduler_state"])
        self.history = ck.get("history", [])
        self.start_epoch = int(ck.get("epoch", 0))
        self.best = float(ck.get("best_metric", -1.0))
        self.log(f"Resumed from {path} at epoch {self.start_epoch} (best val NC {self.best:.4f})")
        return True

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        tot, n, acc = 0.0, 0, {}
        for b, (x, y, _) in enumerate(self.train_loader):
            if self.max_train_batches and b >= self.max_train_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss = self.loss_fn(out, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
            for k, v in batch_metrics(out.detach(), y).items():
                acc[k] = acc.get(k, 0.0) + v.sum().item()
        return {"loss": tot / max(n, 1), **{k: v / max(n, 1) for k, v in acc.items()}}

    def fit(self, on_epoch: Optional[Callable[[Dict], None]] = None) -> Dict:
        self.log(f"Training on {self.device} | params={count_parameters(self.model):,} | cfg={self.cfg}")
        epochs = self.cfg["epochs"]
        stopped_early = False
        for ep in range(self.start_epoch, epochs):
            t0 = time.time()
            tr = self.train_one_epoch()
            va = evaluate_loader(self.model, self.val_loader, self.loss_fn, self.device)
            self.sched.step(va["nc"])
            rec = {"epoch": ep + 1, "lr": self.opt.param_groups[0]["lr"], "epoch_seconds": round(time.time() - t0, 1),
                   **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
            self.history.append(rec)
            improved = va["nc"] > self.best + 1e-4
            if improved:
                self.best, self.bad_epochs = va["nc"], 0
                save_checkpoint(self.ckpt_dir / "best_model.pth", self.model, None, None, ep + 1,
                                self.history, self.best, self.cfg)
            else:
                self.bad_epochs += 1
            save_checkpoint(self.ckpt_dir / "last_model.pth", self.model, self.opt, self.sched, ep + 1,
                            self.history, self.best, self.cfg)
            self.log(f"epoch {ep+1}/{epochs} loss {tr['loss']:.4f} | val loss {va['loss']:.4f} "
                     f"NC {va['nc']:.4f} BER {va['ber']:.4f} acc {va['accuracy']:.4f} | "
                     f"{rec['epoch_seconds']}s {'*best*' if improved else ''}")
            if self.results_dir:
                self.write_history()
            if on_epoch:
                on_epoch(rec)
            if self.bad_epochs >= self.cfg["patience"]:
                self.log(f"Early stopping: no val-NC improvement for {self.bad_epochs} epochs")
                stopped_early = True
                break
        return {"history": self.history, "best_val_nc": self.best, "stopped_early": stopped_early,
                "epochs_completed": len(self.history), "config": self.cfg}

    def write_history(self):
        d = self.results_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "training_history.json").write_text(json.dumps(self.history, indent=2))
        if self.history:
            with open(d / "training_history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
                w.writeheader(); w.writerows(self.history)
