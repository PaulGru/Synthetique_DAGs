from __future__ import annotations
import os, json, math
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score

from data_synth import Env

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
# Évaluation & logs
# =============================

def _predict_logits(model: nn.Module, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    device_t = torch.device(device)
    model.eval()
    with torch.no_grad():
        return model(X.to(device_t))


def evaluate_binary(model: nn.Module, env: Env, device: str = "cpu"):
    logits = _predict_logits(model, env.X, device=device)
    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    y_true = env.y.cpu().numpy().reshape(-1)
    y_pred = (probs >= 0.5).astype(np.float32)
    acc = accuracy_score(y_true, y_pred)
    return float(acc)


def evaluate_group(model: nn.Module, envs: List[Env], device: str = "cpu"):
    accs = []
    for e in envs:
        acc = evaluate_binary(model, e, device=device)
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
    te_acc = evaluate_binary(model, test_env, device=device)
    parts.append(f"Test(OOD): acc={te_acc:.3f}")
    print(" | ".join(parts))


# =============================
# Sauvegarde CSV des envs (inspection)
# =============================

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _maybe(env: Env, key: str, default=None):
    v = getattr(env, "meta", None)
    if isinstance(v, dict) and key in v:
        return v[key]
    return default

def _summary_from_arrays(X, y):
    X = _to_np(X); y = _to_np(y).reshape(-1)
    C = X[:, 1].reshape(-1)
    return {
        "n": int(len(y)),
        "prop_C_egal_Y": float((C == y).mean()),
        "prop_C_trompeuse": float((C != y).mean()),
    }
