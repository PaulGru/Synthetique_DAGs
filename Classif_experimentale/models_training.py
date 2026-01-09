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

def compute_accuracy(model: nn.Module, envs: List[Env], device: str = "cpu") -> float:
    if not envs:
        return 0.0
    accuracies = []
    for env in envs:
        acc = evaluate_binary(model, env, device=device)
        accuracies.append(acc)
    return np.mean(accuracies)

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
    mlp_hidden: int = 256, mlp_layers: int = 1, mlp_dropout: float = 0.1, mlp_bn: bool = False,
    dataset_name: str = "synthetic_semi_anti_causal"
):
    history = {'step': [], 'loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [], 'w_z': [], 'w_y': [], 'align_z': [], 'align_y': [], 'dist_z': [], 'dist_y': []}

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

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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

        if dataset_name == "synthetic_confounding":
            if t == 10000:
                for pg in opt.param_groups: pg['lr'] = 1e-3
            elif t == 20000:
                for pg in opt.param_groups: pg['lr'] = 5e-4
            elif t == 30000:
                for pg in opt.param_groups: pg['lr'] = 1e-4
        elif dataset_name == "synthetic_semi_anti_causal":
            if t == 10000:
                for pg in opt.param_groups: pg['lr'] = 1e-3
            elif t == 20000:
                for pg in opt.param_groups: pg['lr'] = 5e-4
            elif t == 30000:
                for pg in opt.param_groups: pg['lr'] = 1e-4
        elif dataset_name == "synthetic_selection":
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
                # Extract dimension info from first training env if available
                dim_z = envs[0].meta.get('dim_z', 1) if hasattr(envs[0], 'meta') and envs[0].meta else 1
                
                # Split weights and compute norms
                w_z_part = w[:dim_z]
                w_y_part = w[dim_z:]
                
                history['w_z'].append(float(np.linalg.norm(w_z_part)))
                history['w_y'].append(float(np.linalg.norm(w_y_part)))
                
                # Compute alignment with true weights (cosine similarity)
                # Extract w_true and u from environment metadata
                w_true = envs[0].meta.get('w_true', None) if hasattr(envs[0], 'meta') and envs[0].meta else None
                u = envs[0].meta.get('u', None) if hasattr(envs[0], 'meta') and envs[0].meta else None
                
                if w_true is not None and len(w_z_part) > 0:
                    # Cosine similarity: (w · w_true) / (||w|| × ||w_true||)
                    norm_w = np.linalg.norm(w_z_part)
                    norm_true = np.linalg.norm(w_true)
                    if norm_w > 1e-8 and norm_true > 1e-8:
                        align_z = np.dot(w_z_part, w_true) / (norm_w * norm_true)
                    else:
                        align_z = 0.0
                    history['align_z'].append(float(align_z))
                else:
                    history['align_z'].append(0.0)
                
                if u is not None and len(w_y_part) > 0:
                    norm_w = np.linalg.norm(w_y_part)
                    norm_u = np.linalg.norm(u)
                    if norm_w > 1e-8 and norm_u > 1e-8:
                        align_y = np.dot(w_y_part, u) / (norm_w * norm_u)
                    else:
                        align_y = 0.0
                    history['align_y'].append(float(align_y))
                else:
                    history['align_y'].append(0.0)
                
                # Compute Euclidean distance between learned and true weights
                if w_true is not None and len(w_z_part) > 0:
                    dist_z = np.linalg.norm(w_z_part - w_true)
                    history['dist_z'].append(float(dist_z))
                else:
                    history['dist_z'].append(0.0)
                
                if u is not None and len(w_y_part) > 0:
                    dist_y = np.linalg.norm(w_y_part - u)
                    history['dist_y'].append(float(dist_y))
                else:
                    history['dist_y'].append(0.0)
            else:
                history['w_z'].append(0.0)
                history['w_y'].append(0.0)
                history['align_z'].append(0.0)
                history['align_y'].append(0.0)
                history['dist_z'].append(0.0)
                history['dist_y'].append(0.0)

            evaluate_and_log_step("ERM", t+1, model, envs, val_envs, test_env, device=str(device), loss_val=float(loss.item()))

    return model, history

# =============================
# IRM (IRMv1)
# =============================

def _irm_penalty(loss_e_list: List[torch.Tensor], w_pen: torch.Tensor, normalize_by_dim: int = 1):
    """Calcule la pénalité IRM avec normalisation par dimension.
    
    ✅ FIX: Ajout du paramètre normalize_by_dim pour éviter l'explosion
    de la pénalité en haute dimension (la norme au carré croît avec d).
    """
    grads = [grad(le, w_pen, create_graph=True)[0] for le in loss_e_list]
    penalty = sum((g ** 2).sum() for g in grads)
    return penalty / normalize_by_dim  # Normalisation critique!


def train_irm(
    envs: List[Env], steps: int = 500, lr: float = 1e-3, batch: int = 256,
    irm_lambda: float = 5000.0, warmup_steps: int = 0,
    seed: int = 0, device: str = "cpu",
    eval_every: int = 0, val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    model_kind: str = "mlp",
    mlp_hidden: int = 256, mlp_layers: int = 1,
    mlp_dropout: float = 0.1, mlp_bn: bool = False,
    dataset_name: str = "synthetic_semi_anti_causal"
):
    history = {'step': [], 'loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [], 'w_z': [], 'w_y': [], 'align_z': [], 'align_y': [], 'dist_z': [], 'dist_y': []}

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

    # ✅ FIX: Ajout de weight decay pour régularisation L2
    opt = torch.optim.Adam(phi.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='mean')

    # ✅ FIX: Utiliser tout le dataset comme dans l'implémentation officielle Facebook Research
    # Cela donne une estimation plus stable de la pénalité IRM (pas de bruit d'échantillonnage)
    env_data = [(e.X, e.y.view(-1).float()) for e in envs]

    E = len(envs)
    
    # Warmup proportionnel: 10% des steps total
    warmup_steps = max(500, int(steps * 0.1))

    for t in range(steps):
        phi.train()

        # ✅ IMPLÉMENTATION OFFICIELLE IRM (Facebook Research)
        # Source: https://github.com/facebookresearch/InvariantRiskMinimization
        # 
        # Utilise TOUT le dataset à chaque step pour une estimation stable de la pénalité.
        
        emp_risk = 0.0
        penalties = []
        
        for e_idx in range(E):
            X_e, y_e = env_data[e_idx]
            
            # 1. Risque empirique
            logits = phi(X_e).squeeze()
            loss_emp = bce(logits, y_e)
            emp_risk = emp_risk + loss_emp
            
            # 2. Pénalité IRM pour cet environnement
            # Créer un scale unique pour mesurer le gradient
            scale = torch.tensor(1.0, device=device, requires_grad=True)
            loss_scaled = bce(logits * scale, y_e)
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
        
        # ✅ FIX: Rescaling comme dans l'implémentation officielle Facebook Research
        # Source: https://github.com/facebookresearch/InvariantRiskMinimization
        # Quand lambda_t > 1, on divise l'objectif par lambda_t pour garder
        # des gradients dans une plage raisonnable et éviter l'effondrement des poids.
        if lambda_t > 1.0:
            objective = objective / lambda_t

        opt.zero_grad(); objective.backward(); opt.step()

        # NOTE: LR scheduling désactivé pour IRM car il causait l'effondrement des poids.
        # Avec le rescaling Facebook Research, un LR constant fonctionne mieux.


        if eval_every and ((t+1) % eval_every == 0) and (val_envs is not None) and (test_env is not None):
            train_acc = compute_accuracy(phi, envs, device=str(device))
            val_acc = compute_accuracy(phi, val_envs, device=str(device)) if val_envs else 0.0
            test_acc = compute_accuracy(phi, [test_env], device=str(device)) if test_env else 0.0
            
            history['step'].append(t+1)
            history['loss'].append(emp_risk.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            if isinstance(phi, LogisticReg):
                w = phi.linear.weight.detach().cpu().numpy()[0]
                # Extract dimension info from first training env if available
                dim_z = envs[0].meta.get('dim_z', 1) if hasattr(envs[0], 'meta') and envs[0].meta else 1
                
                # Split weights and compute norms
                w_z_part = w[:dim_z]
                w_y_part = w[dim_z:]
                
                history['w_z'].append(float(np.linalg.norm(w_z_part)))
                history['w_y'].append(float(np.linalg.norm(w_y_part)))
                
                # Compute alignment with true weights (cosine similarity)
                w_true = envs[0].meta.get('w_true', None) if hasattr(envs[0], 'meta') and envs[0].meta else None
                u = envs[0].meta.get('u', None) if hasattr(envs[0], 'meta') and envs[0].meta else None
                
                if w_true is not None and len(w_z_part) > 0:
                    norm_w = np.linalg.norm(w_z_part)
                    norm_true = np.linalg.norm(w_true)
                    if norm_w > 1e-8 and norm_true > 1e-8:
                        align_z = np.dot(w_z_part, w_true) / (norm_w * norm_true)
                    else:
                        align_z = 0.0
                    history['align_z'].append(float(align_z))
                else:
                    history['align_z'].append(0.0)
                
                if u is not None and len(w_y_part) > 0:
                    norm_w = np.linalg.norm(w_y_part)
                    norm_u = np.linalg.norm(u)
                    if norm_w > 1e-8 and norm_u > 1e-8:
                        align_y = np.dot(w_y_part, u) / (norm_w * norm_u)
                    else:
                        align_y = 0.0
                    history['align_y'].append(float(align_y))
                else:
                    history['align_y'].append(0.0)
                
                # Compute Euclidean distance between learned and true weights
                if w_true is not None and len(w_z_part) > 0:
                    dist_z = np.linalg.norm(w_z_part - w_true)
                    history['dist_z'].append(float(dist_z))
                else:
                    history['dist_z'].append(0.0)
                
                if u is not None and len(w_y_part) > 0:
                    dist_y = np.linalg.norm(w_y_part - u)
                    history['dist_y'].append(float(dist_y))
                else:
                    history['dist_y'].append(0.0)
            else:
                history['w_z'].append(0.0); history['w_y'].append(0.0)
                history['align_z'].append(0.0); history['align_y'].append(0.0)
                history['dist_z'].append(0.0); history['dist_y'].append(0.0)

            evaluate_and_log_step("IRM", t+1, phi, envs, val_envs, test_env, device=str(device), loss_val=float(emp_risk.item()))

    return phi, history


# =============================
# IB-IRM (Invariant Information Bottleneck)
# =============================

def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extrait la représentation Φ(X) depuis le modèle.
    
    Pour MLP : features de la dernière couche cachée (avant classificateur final)
    Pour LogReg : X directement
    """
    if isinstance(model, LogisticReg):
        return x
    elif isinstance(model, SmallMLP):
        # On veut récupérer l'avant-dernière couche (juste avant la projection finale)
        # Structurallement SmallMLP.net est un Sequential.
        # La dernière couche est Linear(hidden, out_dim).
        # On veut tout sauf la dernière couche.
        out = x
        for layer in model.net[:-1]:
            out = layer(out)
        return out
    else:
        # Fallback pour d'autres modèles non supportés explicitement
        return x

def train_ib_irm(
    envs: List[Env], steps: int = 500, lr: float = 1e-3, batch: int = 256,
    irm_lambda: float = 5000.0, ib_gamma: float = 1e-2, 
    warmup_steps: int = 0,
    seed: int = 0, device: str = "cpu",
    eval_every: int = 0, val_envs: Optional[List[Env]] = None,
    test_env: Optional[Env] = None,
    model_kind: str = "mlp",
    mlp_hidden: int = 256, mlp_layers: int = 1,
    mlp_dropout: float = 0.1, mlp_bn: bool = False,
    dataset_name: str = "synthetic_semi_anti_causal"
):
    history = {'step': [], 'loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [], 
               'w_z': [], 'w_y': [], 'align_z': [], 'align_y': [], 'dist_z': [], 'dist_y': [],
               'var_penalty': []}

    torch.manual_seed(seed)
    device = torch.device(resolve_device(device))
    envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in envs]
    if val_envs is not None:
        val_envs = [Env(e.X.to(device), e.y.to(device), getattr(e, 'y_true', None), getattr(e, 'meta', None)) for e in val_envs]
    if test_env is not None:
        test_env = Env(test_env.X.to(device), test_env.y.to(device), getattr(test_env, 'y_true', None), getattr(test_env, 'meta', None))

    d_in = int(envs[0].X.shape[1])
    
    if model_kind == "logreg":
        print("WARNING: IB-IRM avec Logistic Regression n'est pas recommandé car Var(Phi) est constant ou linéaire.")
        phi = LogisticReg(d_in=d_in).to(device)
    else:
        phi = SmallMLP(d_in=d_in, hidden=mlp_hidden, n_layers=mlp_layers,
                       dropout=mlp_dropout, bn=mlp_bn, out_dim=1).to(device)

    opt = torch.optim.Adam(phi.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction='mean')

    env_data = [(e.X, e.y.view(-1).float()) for e in envs]
    E = len(envs)
    
    warmup_steps = max(500, int(steps * 0.1))

    for t in range(steps):
        phi.train()
        
        emp_risk = 0.0
        penalties = []
        all_logits = []

        for e_idx in range(E):
            X_e, y_e = env_data[e_idx]
            
            # 1. Risque empirique
            logits = phi(X_e).squeeze()
            loss_emp = bce(logits, y_e)
            emp_risk = emp_risk + loss_emp
            
            # 2. Pénalité IRM
            scale = torch.tensor(1.0, device=device, requires_grad=True)
            loss_scaled = bce(logits * scale, y_e)
            grad_scale = grad(loss_scaled, [scale], create_graph=True)[0]
            penalty_e = grad_scale ** 2
            penalties.append(penalty_e)
            
            # 3. Collecte logits pour Var(Phi)
            # Dans la formulation IRM stricte, Phi(X) est la sortie du réseau (logits)
            all_logits.append(logits)
        
        # Moyennes
        emp_risk = emp_risk / E
        penalty = torch.stack(penalties).mean()
        
        # 4. Pénalité de Variance (IB)
        # Var(Phi) = variance des logits (sorties de Phi)
        all_logits_cat = torch.cat(all_logits, dim=0)
        # Centrage
        logits_centered = all_logits_cat - all_logits_cat.mean()
        # Variance empirique
        var_penalty = (logits_centered ** 2).mean() 
        
        # Gestion du warmup pour gamma (IB)
        if t < warmup_steps:
            alpha = t / float(warmup_steps)
            gamma_t = alpha * ib_gamma
        else:
            gamma_t = ib_gamma

        # Loss finale
        objective = emp_risk + irm_lambda * penalty + gamma_t * var_penalty

        if irm_lambda > 1.0:
            objective = objective / irm_lambda

        opt.zero_grad(); objective.backward(); opt.step()

        if eval_every and ((t+1) % eval_every == 0) and (val_envs is not None) and (test_env is not None):
            train_acc = compute_accuracy(phi, envs, device=str(device))
            val_acc = compute_accuracy(phi, val_envs, device=str(device)) if val_envs else 0.0
            test_acc = compute_accuracy(phi, [test_env], device=str(device)) if test_env else 0.0
            
            history['step'].append(t+1)
            history['loss'].append(emp_risk.item())
            history['var_penalty'].append(var_penalty.item())
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            # Note: On ne log pas les poids pour IB-IRM avec MLP (simplification)
            history['w_z'].append(0.0); history['w_y'].append(0.0)
            history['align_z'].append(0.0); history['align_y'].append(0.0)
            history['dist_z'].append(0.0); history['dist_y'].append(0.0)

            evaluate_and_log_step("IB-IRM", t+1, phi, envs, val_envs, test_env, device=str(device), loss_val=float(emp_risk.item()))

    return phi, history