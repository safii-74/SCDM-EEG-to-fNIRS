"""Training loop for SCDM with LR scheduling, gradient clipping, and EMA."""
from __future__ import annotations
import copy
import math
import torch


class EMA:
    """Exponential moving average of model parameters for stable sampling."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone().detach() for k, v in sd.items()}


def _cosine_lr(epoch: int, warmup: int, total: int, peak_lr: float, min_lr: float = 1e-6):
    """Cosine annealing with linear warmup."""
    if epoch < warmup:
        return min_lr + (peak_lr - min_lr) * epoch / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, scdm, optimizer, device="cpu", grad_accum_steps: int = 1,
                 grad_clip_norm: float = 0.0, ema_decay: float = 0.0,
                 warmup_epochs: int = 0, total_epochs: int = 500):
        self.scdm = scdm.to(device)
        self.opt = optimizer
        self.device = device
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip_norm = grad_clip_norm
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.peak_lr = optimizer.param_groups[0]['lr']
        self.ema = EMA(scdm, ema_decay) if ema_decay > 0 else None

    def _to_device(self, planes):
        return {k: v.to(self.device) for k, v in planes.items()}

    def _set_lr(self, epoch):
        lr = _cosine_lr(epoch, self.warmup_epochs, self.total_epochs,
                        self.peak_lr)
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr

    def train_epoch(self, loader):
        self.scdm.train()
        total = 0.0
        self.opt.zero_grad()
        for step, (e0, f0, planes, _) in enumerate(loader):
            e0, f0 = e0.to(self.device), f0.to(self.device)
            loss = self.scdm.loss(e0, f0, self._to_device(planes))
            (loss / self.grad_accum_steps).backward()
            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.scdm.parameters(), self.grad_clip_norm)
                self.opt.step()
                self.opt.zero_grad()
                if self.ema is not None:
                    self.ema.update(self.scdm)
            total += loss.item() * e0.size(0)
        if (step + 1) % self.grad_accum_steps != 0:
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.scdm.parameters(), self.grad_clip_norm)
            self.opt.step()
            self.opt.zero_grad()
            if self.ema is not None:
                self.ema.update(self.scdm)
        return total / len(loader.dataset)

    def fit(self, loader, epochs, save_path="scdm.pt", save_every: int = 100):
        history = []
        best_loss = float('inf')
        for ep in range(epochs):
            lr = self._set_lr(ep)
            loss = self.train_epoch(loader)
            history.append(loss)

            if loss < best_loss:
                best_loss = loss
                self._save_checkpoint(save_path, ep, loss, best=True)

            if (ep + 1) % save_every == 0:
                self._save_checkpoint(
                    save_path.replace('.pt', f'_ep{ep+1}.pt'), ep, loss)

            print(f"epoch {ep + 1:4d}/{epochs}  loss {loss:.4f}  "
                  f"lr {lr:.2e}  best {best_loss:.4f}", flush=True)
        return history

    def _save_checkpoint(self, path, epoch, loss, best=False):
        ckpt = {
            'model': self.scdm.state_dict(),
            'optimizer': self.opt.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }
        if self.ema is not None:
            ckpt['ema'] = self.ema.state_dict()
        torch.save(ckpt, path)
        if best:
            best_path = path.replace('.pt', '_best.pt')
            if best_path == path:
                best_path = path.replace('.pt', '') + '_best.pt'
            torch.save(ckpt, best_path)
