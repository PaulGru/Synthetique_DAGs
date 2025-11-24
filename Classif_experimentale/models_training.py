from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import TensorDataset, DataLoader

from data_synth import Env
from utils_irm import (resolve_device, evaluate_binary, evaluate_group,
                       evaluate_and_log_step)

# =============================
# Modèles
# =============================

class LogisticReg(nn.Module):
    def __init__(self, d_in: int = 2):
        super().__init__()
        self.linear = nn.Linear(d_in, 1, bias=True)
    def forward(self, x):
        return self.linear(x).squeeze(-1)

class SmallMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, n_layers: int = 1,
                 dropout: float = 0.0, bn: bool = False, out_dim: int = 1):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(in_dim, hidden)]
            if bn: layers.append(nn.BatchNorm1d(hidden))
            layers += [nn.ReLU()]
            if dropout > 0.0: layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out

class EnvHead1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        return self.linear(z).squeeze(-1)

# =============================
# ERM
# =============================

def train_erm(
    envs: List[Env], steps: int = 500, lr: float = 1e-3, batch: int = 256,
    seed: int = 0, device: str = "cpu",
    eval_every: int = 0, val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    model_kind: str = "mlp",
    mlp_hidden: int = 256, mlp_layers: int = 1, mlp_dropout: float = 0.1, mlp_bn: bool = False
):

    torch.manual_seed(seed)
    device = torch.device(resolve_device(device))
    envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in envs]
    if val_envs is not None:
        val_envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in val_envs]
    if test_env is not None:
        test_env = Env(test_env.X.to(device), test_env.y.to(device), getattr(test_env, 'y_true', None), getattr(test_env, 'meta', None))

    d_in = int(envs[0].X.shape[1])
    if model_kind == "logreg":
        model = LogisticReg(d_in=d_in).to(device)
    else:
        model = SmallMLP(d_in=d_in, hidden=mlp_hidden, n_layers=mlp_layers,
                        dropout=mlp_dropout, bn=mlp_bn, out_dim=1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    X_all = torch.cat([e.X for e in envs], dim=0)
    y_all = torch.cat([e.y for e in envs], dim=0).float().view(-1)
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
        loss = bce(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()

        if eval_every and ((t+1) % eval_every == 0) and (val_envs is not None) and (test_env is not None):
            evaluate_and_log_step("ERM", t+1, model, envs, val_envs, test_env, device=str(device), loss_val=float(loss.item()))

    return model

# =============================
# IRM (IRMv1)
# =============================

def _irm_penalty(loss_e_list: List[torch.Tensor], w_pen: torch.Tensor):
    grads = [grad(le, w_pen, create_graph=True)[0] for le in loss_e_list]
    return sum((g ** 2).sum() for g in grads)


def train_irm(
    envs: List[Env], steps: int = 500, lr: float = 1e-3, batch: int = 256,
    irm_lambda: float = 5000.0, warmup_steps: int = 0,
    seed: int = 0, device: str = "cpu",
    eval_every: int = 0, val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    model_kind: str = "mlp",
    mlp_hidden: int = 256, mlp_layers: int = 1,
    mlp_dropout: float = 0.1, mlp_bn: bool = False
):

    torch.manual_seed(seed)
    device = torch.device(resolve_device(device))
    envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in envs]
    if val_envs is not None:
        val_envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in val_envs]
    if test_env is not None:
        test_env = Env(test_env.X.to(device), test_env.y.to(device), getattr(test_env, 'y_true', None), getattr(test_env, 'meta', None))

    d_in = int(envs[0].X.shape[1])
    if model_kind == "logreg":
        phi = LogisticReg(d_in=d_in).to(device)
    else:
        phi = SmallMLP(d_in=d_in, hidden=mlp_hidden, n_layers=mlp_layers,
                       dropout=mlp_dropout, bn=mlp_bn, out_dim=1).to(device)

    opt = torch.optim.Adam(phi.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction='mean')

    def make_loader(env):
        ds = TensorDataset(env.X, env.y.view(-1).float())
        return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    loaders = [make_loader(e) for e in envs]
    iters = [iter(ld) for ld in loaders]

    def next_batch(e_idx):
        try:
            return next(iters[e_idx])
        except StopIteration:
            iters[e_idx] = iter(loaders[e_idx])
            return next(iters[e_idx])

    E = len(envs)
    penalty_start_step = 1500   # step à partir duquel la pénalité est activée
    warmup_steps = 3500         # durée du warmup après ce step

    for t in range(steps):
        phi.train()
        w_pen = torch.tensor(1.0, device=device, requires_grad=True)

        loss_e_list = []
        emp_risk = 0.0
        for e_idx in range(E):
            Xb, yb = next_batch(e_idx)
            logits_emp = phi(Xb).squeeze()
            loss_emp = bce(logits_emp, yb.squeeze(-1))
            emp_risk = emp_risk + loss_emp

            logits_pen = logits_emp * w_pen
            loss_pen = bce(logits_pen, yb)
            loss_e_list.append(loss_pen)

        penalty = _irm_penalty(loss_e_list, w_pen)

        # scheduling de lambda_t
        if t < penalty_start_step:
            # avant penalty_start_step : aucune pénalité
            lambda_t = 0.0
        elif penalty_start_step + warmup_steps:
            alpha = (t - penalty_start_step) / float(warmup_steps)
            lambda_t = alpha * irm_lambda
        else:
            lambda_t = irm_lambda

        objective = (emp_risk / E) + lambda_t * penalty

        opt.zero_grad(); objective.backward(); opt.step()

        if eval_every and ((t+1) % eval_every == 0) and (val_envs is not None) and (test_env is not None):
            evaluate_and_log_step("IRM", t+1, phi, envs, val_envs, test_env, device=str(device), loss_val=float((emp_risk / E).item()))

    return phi


# =============================
# EIRM (IRM Games, BRD)
# =============================

def train_eirm(envs_train: List[Env], envs_val: Optional[List[Env]], test_env: Optional[Env],
               steps: int = 400, lr: float = 1e-4, br_updates: int = 1, batch: int = 256,
               weight_decay: float = 0.0, weight_phi_decay: float = 5e-4, eval_every: int = 50,
               device: str = 'cpu', phi_kind: str = "mlp", seed: int = 0,
               mlp_hidden: int = 256, mlp_layers: int = 1, mlp_dropout: float = 0.1, mlp_bn: bool = False,
               phi_lr_factor: float = 1.0, phi_period: Optional[int] = None
) -> Tuple[nn.Module, List[nn.Module]]:

    device_t = torch.device(resolve_device(device))
    E = len(envs_train)
    d_in = envs_train[0].X.size(1)

    if phi_period is None:
        phi_period = E + 1

    if phi_kind == "logreg":
        # Représentation Φ scalaire
        phi = LogisticReg(d_in=d_in).to(device_t)
        heads = [EnvHead1D().to(device_t) for _ in range(E)]
    else:
        phi = SmallMLP(d_in=d_in, hidden=mlp_hidden, n_layers=mlp_layers,
                       dropout=mlp_dropout, bn=mlp_bn, out_dim=1).to(device_t)
        rep_dim = 1
        heads = [SmallMLP(d_in=rep_dim, hidden=mlp_hidden, n_layers=mlp_layers,
            dropout=mlp_dropout, bn=mlp_bn, out_dim=1).to(device_t) for _ in range(E)]

    # têtes 1D par environnement
    # heads = [EnvHead1D().to(device_t) for _ in range(E)]
    # rep_dim = 1
    # heads = [SmallMLP(d_in=rep_dim, hidden=mlp_hidden, n_layers=1,
    #     dropout=mlp_dropout, bn=mlp_bn, out_dim=1).to(device_t) for _ in range(E)]

    opt_phi = torch.optim.Adam(phi.parameters(), lr=lr * phi_lr_factor, weight_decay=weight_phi_decay)
    opts = [torch.optim.Adam(h.parameters(), lr=lr, weight_decay=weight_decay) for h in heads]
    bce = nn.BCEWithLogitsLoss()

    def make_loader(env):
        ds = TensorDataset(env.X.to(device_t), env.y.float().view(-1).to(device_t))
        return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    loaders = [make_loader(e) for e in envs_train]
    iters = [iter(ld) for ld in loaders]

    def next_batch(e_idx):
        try:
            return next(iters[e_idx])
        except StopIteration:
            iters[e_idx] = iter(loaders[e_idx])
            return next(iters[e_idx])

    X_all = torch.cat([e.X.to(device_t) for e in envs_train], dim=0)
    y_all = torch.cat([e.y.float().view(-1).to(device_t) for e in envs_train], dim=0)

    def sample_batch_pool(bs):
        idx = torch.randint(0, X_all.size(0), (bs,), device=device_t)
        return X_all[idx], y_all[idx]

    def ensemble_logits(z):
        z = as_col(z)
        out = 0.0
        for h in heads:
            out = out + h(z)
        return out / len(heads)

    def eval_env_concat(env_list):
        X = torch.cat([e.X.to(device_t) for e in env_list], dim=0)
        y = torch.cat([e.y.float().view(-1).to(device_t) for e in env_list], dim=0)
        z = phi(X)
        logits = ensemble_logits(z).squeeze(-1)
        loss = bce(logits, y).item()
        y_hat = (torch.sigmoid(logits) >= 0.5).float()
        acc = (y_hat == y).float().mean().item()
        return {"acc": acc, "loss": loss, "n": X.size(0)}

    def eval_one_env(env_obj):
        X = env_obj.X.to(device_t)
        y = env_obj.y.float().view(-1).to(device_t)
        z = phi(X)
        logits = ensemble_logits(z).squeeze(-1)
        loss = bce(logits, y).item()
        y_hat = (torch.sigmoid(logits) >= 0.5).float()
        acc = (y_hat == y).float().mean().item()
        return {"acc": acc, "loss": loss, "n": X.size(0)}

    def as_col(z: torch.Tensor) -> torch.Tensor:
        # z peut être (N,) ou déjà (N,1) → toujours (N,1)
        return z.reshape(-1, 1)

    for t in range(1, steps + 1):
        player = (t - 1) % phi_period

        if player < E:
            # (A) Best-response sur la tête `player`
            e_idx = player
            for _ in range(max(1, br_updates)):
                Xb, yb = next_batch(e_idx)
                Zb = as_col(phi(Xb))  # phi n'est PAS dans no_grad : on l'utilise juste en forward

                # autres têtes figées
                with torch.no_grad():
                    others = 0.0
                    for j, hj in enumerate(heads):
                        if j == e_idx:
                            continue
                        others = others + hj(Zb)

                logit = (heads[e_idx](Zb) + others) / E
                loss  = bce(logit.squeeze(-1), yb)

                opts[e_idx].zero_grad(set_to_none=True)
                loss.backward()
                opts[e_idx].step()

        else:
            # (B) Tour de φ : mise à jour de la représentation
            Xp, yp = sample_batch_pool(batch)
            Zp = as_col(phi(Xp))
            logits_avg = ensemble_logits(Zp)
            loss_phi = bce(logits_avg.squeeze(-1), yp)

            opt_phi.zero_grad(set_to_none=True)
            loss_phi.backward()
            opt_phi.step()

        # (C) Logs périodiques (inchangé)
        if (t % eval_every == 0) or (t == 1) or (t == steps):
            tr = eval_env_concat(envs_train)
            va = eval_env_concat(envs_val) if envs_val else {"acc": float('nan'), "loss": float('nan')}
            te = eval_one_env(test_env) if test_env else {"acc": float('nan'), "loss": float('nan')}
            print(
                f"[EIRM] step {t:>4} | "
                f"train-acc={tr['acc']:.3f} loss={tr['loss']:.4f} | "
                f"val-acc={va['acc']:.3f} loss={va['loss']:.4f} | "
                f"test-acc={te['acc']:.3f} loss={te['loss']:.4f}"
            )

    return phi, heads
