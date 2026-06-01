from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
import torch
from torch import nn
from env import Env

# =============================
# Device
# =============================

def resolve_device(d: str) -> str:
    if d == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        try:
            if torch.backends.mps.is_available():
                return 'mps'
        except AttributeError:
            pass
        return 'cpu'
    return d

# =============================
# Evaluation & logging
# =============================

def _predict_logits(model: nn.Module, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    device_t = torch.device(device)
    model.eval()
    
    # Determine a safe batch size to avoid OOM
    batch_size = 128
    if hasattr(model, "bert"):
        batch_size = 128   # BERT requires much more VRAM

    with torch.no_grad():
        logits_list = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size].to(device_t)
            logits_list.append(model(batch_X).cpu())
        return torch.cat(logits_list, dim=0)


def evaluate_binary(logits: torch.Tensor, y: torch.Tensor):
    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    y_true = y.cpu().numpy().reshape(-1)
    y_pred = (probs >= 0.5).astype(np.float32)
    return float((y_true == y_pred).mean())


def evaluate_multiclass(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Accuracy for a multiclass task (softmax → argmax)."""
    if logits.dim() == 1:
        # Binary model called by mistake — fallback
        return evaluate_binary(logits, y)
    y_pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
    y_true = y.cpu().numpy().reshape(-1)
    return float((y_true == y_pred).mean())


def evaluate_env(model: nn.Module, env: Env, device: str = "cpu", max_samples: int = 2000) -> float:
    """Automatically dispatches to binary or multiclass evaluation based on model output shape."""
    if max_samples is not None and len(env.X) > max_samples:
        # Sub-sample evaluation to avoid massive delays
        indices = torch.randperm(len(env.X))[:max_samples]
        X_eval = env.X[indices]
        y_eval = env.y[indices]
    else:
        X_eval = env.X
        y_eval = env.y

    logits = _predict_logits(model, X_eval, device=device)
    if logits.dim() == 2 and logits.shape[1] > 1:
        return evaluate_multiclass(logits, y_eval)
    return evaluate_binary(logits, y_eval)


def evaluate_group(model: nn.Module, envs: List[Env], device: str = "cpu"):
    accs = []
    for e in envs:
        acc = evaluate_env(model, e, device=device)
        accs.append(acc)
    return float(np.mean(accs))


def evaluate_and_log_step(tag: str, step: int, model: nn.Module,
                          train_envs: List[Env], val_envs: List[Env], test_env: Env,
                          device: str = "cpu", loss_val: Optional[float] = None):
    parts = [f"[{tag}] step {step}"]
    if loss_val is not None:
        parts.append(f"loss={loss_val:.4f}")
    tr_acc = evaluate_group(model, train_envs, device=device)
    parts.append(f"Train(ID): acc={tr_acc:.3f}")
    va_acc = evaluate_group(model, val_envs, device=device)
    parts.append(f"Val(ID): acc={va_acc:.3f}")
    te_acc = evaluate_env(model, test_env, device=device)
    parts.append(f"Test(OOD): acc={te_acc:.3f}")
    print(" | ".join(parts))
