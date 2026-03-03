"""
models_bios.py — Modèles ERM et IRM pour Bias in Bios (classification multi-classes).

Différences par rapport à models_training.py :
- CrossEntropyLoss au lieu de BCEWithLogitsLoss
- out_dim = N_CLASSES (28 professions)
- History enrichi avec gender_gap sur le test OOD
- Pas de LR schedule hardcodé par dataset_name
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset

from data_bios import Env, N_CLASSES
from utils_bios import (
    evaluate_multiclass,
    evaluate_group,
    evaluate_by_gender,
    evaluate_and_log_step,
    resolve_device,
)


# =============================================================================
# Modèles
# =============================================================================

class LogisticReg(nn.Module):
    """Régression logistique multi-classes (Linear → CrossEntropy)."""

    def __init__(self, d_in: int, n_classes: int = N_CLASSES):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes, bias=True)

    def forward(self, x):
        return self.linear(x)  # (B, C) — logits bruts


class SmallMLP(nn.Module):
    """MLP léger multi-classes."""

    def __init__(
        self,
        d_in: int,
        hidden: int = 256,
        n_layers: int = 1,
        dropout: float = 0.1,
        bn: bool = False,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(in_dim, hidden))
            if bn:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, C)


def _build_model(model_kind: str, d_in: int, device: torch.device, **kwargs) -> nn.Module:
    if model_kind == "logreg":
        return LogisticReg(d_in=d_in).to(device)
    return SmallMLP(
        d_in=d_in,
        hidden=kwargs.get("mlp_hidden", 256),
        n_layers=kwargs.get("mlp_layers", 1),
        dropout=kwargs.get("mlp_dropout", 0.1),
        bn=kwargs.get("mlp_bn", False),
    ).to(device)


def _init_history() -> Dict:
    return {
        "step": [],
        "loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "gender_gap": [],  # |acc_F - acc_M| sur le test OOD
    }


def _move_envs(envs: List[Env], device: torch.device) -> List[Env]:
    return [
        Env(
            X=e.X.to(device),
            y=e.y.to(device),
            meta=e.meta,
        )
        for e in envs
    ]


# =============================================================================
# ERM
# =============================================================================

def train_erm(
    envs: List[Env],
    steps: int = 10000,
    lr: float = 1e-3,
    batch: int = 256,
    seed: int = 0,
    device: str = "cpu",
    eval_every: int = 500,
    val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    model_kind: str = "logreg",
    mlp_hidden: int = 256,
    mlp_layers: int = 1,
    mlp_dropout: float = 0.1,
    mlp_bn: bool = False,
) -> tuple:

    history = _init_history()
    torch.manual_seed(seed)
    dev = torch.device(resolve_device(device))

    envs = _move_envs(envs, dev)
    if val_envs:
        val_envs = _move_envs(val_envs, dev)
    if test_env:
        test_env = Env(test_env.X.to(dev), test_env.y.to(dev), test_env.meta)

    d_in = int(envs[0].X.shape[1])
    model = _build_model(
        model_kind, d_in, dev,
        mlp_hidden=mlp_hidden, mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout, mlp_bn=mlp_bn,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    # Concaténer tous les environnements d'entraînement
    X_all = torch.cat([e.X for e in envs], dim=0)
    y_all = torch.cat([e.y for e in envs], dim=0).long()
    loader = DataLoader(TensorDataset(X_all, y_all), batch_size=batch, shuffle=True, drop_last=False)
    it = iter(loader)

    for t in range(steps):
        try:
            Xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            Xb, yb = next(it)

        model.train()
        logits = model(Xb)
        loss = ce(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if eval_every and (t + 1) % eval_every == 0 and val_envs and test_env:
            tr_acc = evaluate_group(model, envs, str(dev))
            va_acc = evaluate_group(model, val_envs, str(dev))
            te_acc = evaluate_multiclass(model, test_env, str(dev))
            gg = evaluate_by_gender(model, test_env, str(dev))["gap"]

            history["step"].append(t + 1)
            history["loss"].append(loss.item())
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["test_acc"].append(te_acc)
            history["gender_gap"].append(gg)

            evaluate_and_log_step("ERM", t + 1, model, envs, val_envs, test_env,
                                  str(dev), loss_val=loss.item())

    return model, history


# =============================================================================
# IRM (IRMv1)
# =============================================================================

def train_irm(
    envs: List[Env],
    steps: int = 10000,
    lr: float = 1e-3,
    batch: int = 256,
    irm_lambda: float = 500.0,
    seed: int = 0,
    device: str = "cpu",
    eval_every: int = 500,
    val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    model_kind: str = "logreg",
    mlp_hidden: int = 256,
    mlp_layers: int = 1,
    mlp_dropout: float = 0.1,
    mlp_bn: bool = False,
) -> tuple:

    history = _init_history()
    torch.manual_seed(seed)
    dev = torch.device(resolve_device(device))

    envs = _move_envs(envs, dev)
    if val_envs:
        val_envs = _move_envs(val_envs, dev)
    if test_env:
        test_env = Env(test_env.X.to(dev), test_env.y.to(dev), test_env.meta)

    d_in = int(envs[0].X.shape[1])
    phi = _build_model(
        model_kind, d_in, dev,
        mlp_hidden=mlp_hidden, mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout, mlp_bn=mlp_bn,
    )
    opt = torch.optim.Adam(phi.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(reduction="mean")

    env_data = [(e.X, e.y.long()) for e in envs]
    E = len(envs)
    warmup_steps = max(500, int(steps * 0.1))

    for t in range(steps):
        phi.train()

        emp_risk = torch.tensor(0.0, device=dev)
        penalties = []

        for X_e, y_e in env_data:
            logits = phi(X_e)

            # Risque empirique
            loss_e = ce(logits, y_e)
            emp_risk = emp_risk + loss_e

            # Pénalité IRM : trick du scalaire
            scale = torch.tensor(1.0, device=dev, requires_grad=True)
            loss_scaled = ce(logits * scale, y_e)
            grad_scale = grad(loss_scaled, [scale], create_graph=True)[0]
            penalties.append(grad_scale ** 2)

        emp_risk = emp_risk / E
        penalty = torch.stack(penalties).mean()

        # Warmup linéaire de lambda
        alpha = min(1.0, t / float(warmup_steps))
        lambda_t = alpha * irm_lambda

        objective = emp_risk + lambda_t * penalty
        if lambda_t > 1.0:
            objective = objective / lambda_t

        opt.zero_grad()
        objective.backward()
        opt.step()

        if eval_every and (t + 1) % eval_every == 0 and val_envs and test_env:
            tr_acc = evaluate_group(phi, envs, str(dev))
            va_acc = evaluate_group(phi, val_envs, str(dev))
            te_acc = evaluate_multiclass(phi, test_env, str(dev))
            gg = evaluate_by_gender(phi, test_env, str(dev))["gap"]

            history["step"].append(t + 1)
            history["loss"].append(emp_risk.item())
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["test_acc"].append(te_acc)
            history["gender_gap"].append(gg)

            evaluate_and_log_step("IRM", t + 1, phi, envs, val_envs, test_env,
                                  str(dev), loss_val=emp_risk.item())

    return phi, history
