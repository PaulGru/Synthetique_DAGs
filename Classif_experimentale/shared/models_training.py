from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from data_synth import Env
from utils_irm import (resolve_device, evaluate_binary, evaluate_group,
                       evaluate_and_log_step, evaluate_env)

def compute_accuracy(model: nn.Module, envs: List[Env], device: str = "cpu") -> float:
    if not envs:
        return 0.0
    return float(np.mean([evaluate_env(model, env, device=device) for env in envs]))


def _group_balanced_sampler(
    y: torch.Tensor,
    A: torch.Tensor | None,
    num_samples: int,
) -> WeightedRandomSampler:
    """
    Retourne un WeightedRandomSampler qui équilibre les groupes (Y × A).

    Si A est fourni : 4 groupes (Y=0/1 × A=0/1), chaque groupe reçoit un poids
    inversement proportionnel à sa fréquence → représentation égale en espérance.
    Si A=None : équilibre uniquement sur Y (2 classes).

    Échantillonne avec replacement — les exemples minoritaires apparaissent
    plusieurs fois par epoch, ce qui est équivalent à une loss pondérée.
    """
    y_bin = y.long().view(-1)
    if A is not None:
        groups = y_bin * 2 + A.long().view(-1)   # 0=neg_sae, 1=neg_aae, 2=pos_sae, 3=pos_aae
        n_groups = 4
    else:
        groups = y_bin
        n_groups = 2
    counts = torch.bincount(groups, minlength=n_groups).float().clamp(min=1)
    weights = (1.0 / counts[groups]).tolist()
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

# =============================
# Modèles
# =============================

class LogisticReg(nn.Module):
    def __init__(self, d_in: int = 2, bn: bool = False, out_dim: int = 1):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in) if bn else None
        self.linear = nn.Linear(d_in, out_dim, bias=not bn)
    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        out = self.linear(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


class SmallMLP(nn.Module):
    """MLP à 2 couches cachées (hidden → hidden//2 → out_dim)."""
    def __init__(self, d_in: int, hidden: int = 512, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )
    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


# =============================
# =============================

def train_erm(
    envs: List[Env], steps: int = 500, lr: float = 1e-3, batch: int = 256,
    seed: int = 0, device: str = "cpu",
    eval_every: int = 0, val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    logreg_bn: bool = False,
    balanced_sampling: bool = False,
    dataset_name: str = "synthetic_semi_anti_causal",
    n_classes: int = 2,
    use_mlp: bool = False,
    mlp_hidden: int = 512,
    mlp_dropout: float = 0.1,
):
    history = {'step': [], 'loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [], 'w_z': [], 'w_y': [], 'w_full': []}

    multiclass = n_classes > 2

    torch.manual_seed(seed)
    device = torch.device(resolve_device(device))
    envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in envs]
    if val_envs is not None:
        val_envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in val_envs]
    if test_env is not None:
        test_env = Env(test_env.X.to(device), test_env.y.to(device), getattr(test_env, 'y_true', None), getattr(test_env, 'meta', None))

    d_in = int(envs[0].X.shape[1])

    # Création du modèle
    out_dim = n_classes if multiclass else 1
    if use_mlp:
        model = SmallMLP(d_in=d_in, hidden=mlp_hidden, out_dim=out_dim, dropout=mlp_dropout).to(device)
    else:
        model = LogisticReg(d_in=d_in, bn=logreg_bn, out_dim=out_dim).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss() if multiclass else nn.BCEWithLogitsLoss()

    X_all = torch.cat([e.X for e in envs], dim=0)
    if multiclass:
        y_all = torch.cat([e.y for e in envs], dim=0).long().view(-1)
    else:
        y_all = torch.cat([e.y for e in envs], dim=0).float().view(-1)

    if balanced_sampling:
        A_all = None
        if all(e.meta and "A" in e.meta for e in envs):
            A_all = torch.cat([e.meta["A"].to(device) for e in envs], dim=0)
        sampler = _group_balanced_sampler(y_all, A_all, num_samples=len(y_all))
        loader = DataLoader(TensorDataset(X_all, y_all), batch_size=batch, sampler=sampler, drop_last=False)
    else:
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
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()

        # LR decay pour les datasets synthétiques (même schedule pour tous)
        if dataset_name.startswith("synthetic_"):
            if t == 10000:
                for pg in opt.param_groups: pg['lr'] = 1e-3
            elif t == 20000:
                for pg in opt.param_groups: pg['lr'] = 5e-4
            elif t == 30000:
                for pg in opt.param_groups: pg['lr'] = 1e-4


        if eval_every and ((t+1) % eval_every == 0) and (val_envs is not None) and (test_env is not None):
            # Eval
            train_acc = compute_accuracy(model, envs, device=str(device))
            val_acc = compute_accuracy(model, val_envs, device=str(device)) if val_envs else 0.0
            test_acc = compute_accuracy(model, [test_env], device=str(device)) if test_env else 0.0
    
            history['step'].append(t+1)
            history['loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            if isinstance(model, LogisticReg):
                w = model.linear.weight.detach().cpu().numpy()[0]
                history['w_full'].append(model.linear.weight.detach().cpu().numpy().copy())
                if hasattr(envs[0], 'meta') and envs[0].meta and 'dir_sem' in envs[0].meta:
                    # Données NLP avec directions sémantique/confondante connues :
                    # projeter w sur chaque direction pour mesurer leur usage relatif
                    dir_sem  = envs[0].meta['dir_sem']
                    dir_conf = envs[0].meta['dir_conf']
                    history['w_z'].append(float(np.abs(np.dot(w, dir_sem))))
                    history['w_y'].append(float(np.abs(np.dot(w, dir_conf))))
                elif hasattr(envs[0], 'meta') and envs[0].meta and 'dim_z' in envs[0].meta:
                    # Données synthétiques 2D : séparer la part causale (w_z) et confondante (w_y)
                    dim_z    = envs[0].meta['dim_z']
                    history['w_z'].append(float(np.linalg.norm(w[:dim_z])))
                    history['w_y'].append(float(np.linalg.norm(w[dim_z:])))
                else:
                    # Embedding haute dimension (ex. Moji/DistilBERT 768-d) :
                    # - w_z = ‖w‖₂  : norme totale → décisivité du classifieur
                    # - w_y = ‖b‖₂  : norme du biais (compatible binaire ET multiclasse)
                    b = model.linear.bias
                    history['w_z'].append(float(np.linalg.norm(w)))
                    history['w_y'].append(float(b.detach().cpu().norm().item()) if b is not None else 0.0)
            else:
                history['w_z'].append(0.0)
                history['w_y'].append(0.0)

            evaluate_and_log_step("ERM", t+1, model, envs, val_envs, test_env, device=str(device), loss_val=float(loss.item()))

    return model, history

# =============================
# IRM (IRMv1)
# =============================

def _irm_penalty(loss_e_list: List[torch.Tensor], w_pen: torch.Tensor, normalize_by_dim: int = 1):

    grads = [grad(le, w_pen, create_graph=True)[0] for le in loss_e_list]
    penalty = sum((g ** 2).sum() for g in grads)
    return penalty / normalize_by_dim  # Normalisation critique!


def train_irm(
    envs: List[Env], steps: int = 500, lr: float = 1e-3, batch: int = 256,
    irm_lambda: float = 5000.0, warmup_steps: int = 0,
    seed: int = 0, device: str = "cpu",
    eval_every: int = 0, val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    logreg_bn: bool = False,
    balanced_sampling: bool = False,
    dataset_name: str = "synthetic_semi_anti_causal",
    n_classes: int = 2,
    use_mlp: bool = False,
    mlp_hidden: int = 512,
    mlp_dropout: float = 0.1,
):
    history = {'step': [], 'loss': [], 'penalty': [], 'train_acc': [], 'val_acc': [], 'test_acc': [], 'w_z': [], 'w_y': [], 'w_full': []}

    multiclass = n_classes > 2

    torch.manual_seed(seed)
    device = torch.device(resolve_device(device))
    envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in envs]
    if val_envs is not None:
        val_envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in val_envs]
    if test_env is not None:
        test_env = Env(test_env.X.to(device), test_env.y.to(device), getattr(test_env, 'y_true', None), getattr(test_env, 'meta', None))

    d_in = int(envs[0].X.shape[1])

    # Création du modèle
    out_dim = n_classes if multiclass else 1
    if use_mlp:
        phi = SmallMLP(d_in=d_in, hidden=mlp_hidden, out_dim=out_dim, dropout=mlp_dropout).to(device)
    else:
        phi = LogisticReg(d_in=d_in, bn=logreg_bn, out_dim=out_dim).to(device)

    opt = torch.optim.Adam(phi.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction='mean') if multiclass else nn.BCEWithLogitsLoss(reduction='mean')

    if multiclass:
        env_raw = [(e.X, e.y.view(-1).long()) for e in envs]
    else:
        env_raw = [(e.X, e.y.view(-1).float()) for e in envs]

    E = len(envs)

    # Per-environment DataLoaders — même batch_size que ERM pour un batching aligné
    env_loaders = []
    for e_idx, (X_e, y_e) in enumerate(env_raw):
        if balanced_sampling:
            A_e = envs[e_idx].meta.get("A") if envs[e_idx].meta else None
            sampler = _group_balanced_sampler(y_e, A_e, num_samples=len(y_e))
            loader_e = DataLoader(TensorDataset(X_e, y_e), batch_size=batch, sampler=sampler, drop_last=False)
        else:
            loader_e = DataLoader(TensorDataset(X_e, y_e), batch_size=batch, shuffle=True, drop_last=False)
        env_loaders.append(loader_e)
    env_iters = [iter(loader) for loader in env_loaders]

    # Warmup proportionnel: 10% des steps total
    warmup_steps = max(500, int(steps * 0.1))

    for t in range(steps):
        phi.train()

        emp_risk = 0.0
        penalties = []

        for e_idx in range(E):
            # Mini-batch par environnement (aligné avec ERM)
            try:
                Xb_e, yb_e = next(env_iters[e_idx])
            except StopIteration:
                env_iters[e_idx] = iter(env_loaders[e_idx])
                Xb_e, yb_e = next(env_iters[e_idx])

            # 1. Risque empirique
            logits = phi(Xb_e)
            if not multiclass:
                logits = logits.squeeze()
            loss_emp = loss_fn(logits, yb_e)
            emp_risk = emp_risk + loss_emp

            # 2. Pénalité IRM pour cet environnement
            scale = torch.tensor(1.0, device=device, requires_grad=True)
            loss_scaled = loss_fn(logits * scale, yb_e)
            grad_scale = grad(loss_scaled, [scale], create_graph=True)[0]
            penalty_e = grad_scale ** 2
            penalties.append(penalty_e)

        # Moyennes
        emp_risk = emp_risk / E
        penalty = torch.stack(penalties).mean()

        
        if t < warmup_steps:
            # warmup linéaire de 0 à irm_lambda
            alpha = t / float(warmup_steps)
            lambda_t = alpha * irm_lambda
        else:
            # après warmup : lambda constant
            lambda_t = irm_lambda

        objective = emp_risk + lambda_t * penalty
        
        if lambda_t > 1.0:
            objective = objective / lambda_t

        opt.zero_grad(); objective.backward(); opt.step()

        if eval_every and ((t+1) % eval_every == 0) and (val_envs is not None) and (test_env is not None):
            train_acc = compute_accuracy(phi, envs, device=str(device))
            val_acc = compute_accuracy(phi, val_envs, device=str(device)) if val_envs else 0.0
            test_acc = compute_accuracy(phi, [test_env], device=str(device)) if test_env else 0.0
            
            history['step'].append(t+1)
            history['loss'].append(emp_risk.item())
            history['penalty'].append(penalty.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            if isinstance(phi, LogisticReg):
                w = phi.linear.weight.detach().cpu().numpy()[0]
                history['w_full'].append(phi.linear.weight.detach().cpu().numpy().copy())
                if hasattr(envs[0], 'meta') and envs[0].meta and 'dir_sem' in envs[0].meta:
                    dir_sem  = envs[0].meta['dir_sem']
                    dir_conf = envs[0].meta['dir_conf']
                    history['w_z'].append(float(np.abs(np.dot(w, dir_sem))))
                    history['w_y'].append(float(np.abs(np.dot(w, dir_conf))))
                elif hasattr(envs[0], 'meta') and envs[0].meta and 'dim_z' in envs[0].meta:
                    dim_z = envs[0].meta['dim_z']
                    history['w_z'].append(float(np.linalg.norm(w[:dim_z])))
                    history['w_y'].append(float(np.linalg.norm(w[dim_z:])))
                else:
                    # Embedding haute dimension (ex. Moji/DistilBERT 768-d) :
                    # - w_z = ‖w‖₂  : norme totale → décisivité du classifieur
                    # - w_y = ‖b‖₂  : norme du biais (compatible binaire ET multiclasse)
                    b = phi.linear.bias
                    history['w_z'].append(float(np.linalg.norm(w)))
                    history['w_y'].append(float(b.detach().cpu().norm().item()) if b is not None else 0.0)
            else:
                history['w_z'].append(0.0); history['w_y'].append(0.0)

            evaluate_and_log_step("IRM", t+1, phi, envs, val_envs, test_env, device=str(device), loss_val=float(emp_risk.item()))

    return phi, history
