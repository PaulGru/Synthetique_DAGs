from __future__ import annotations
import os, json, math
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
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
    # acc = accuracy_score(y_true, y_pred)
    acc = (y_true == y_pred).mean()
    return float(acc)


def evaluate_multiclass(model: nn.Module, env: Env, device: str = "cpu") -> float:
    """Accuracy pour une tâche multiclasse (softmax → argmax)."""
    logits = _predict_logits(model, env.X, device=device)
    if logits.dim() == 1:
        # Modèle binaire appelé par erreur — fallback
        return evaluate_binary(model, env, device=device)
    y_pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
    y_true = env.y.cpu().numpy().reshape(-1)
    return float((y_true == y_pred).mean())


def evaluate_env(model: nn.Module, env: Env, device: str = "cpu") -> float:
    """Dispatch automatique binary/multiclass selon la forme du modèle."""
    logits = _predict_logits(model, env.X, device=device)
    if logits.dim() == 2 and logits.shape[1] > 1:
        return evaluate_multiclass(model, env, device=device)
    return evaluate_binary(model, env, device=device)


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


# =============================
# Visualisation des données univariés
# =============================

def visualize_1d_simple(envs, env_names, filename="plot/data_1d_simple.png"):
    """
    Visualisation simplifiée quand dim_z=1 et dim_y=1.
    Axe X = La feature causale brute.
    Axe Y = La feature spurieuse brute.
    """
    n_envs = len(envs)
    fig, axes = plt.subplots(n_envs, 2, figsize=(16, 5 * n_envs))
    if n_envs == 1: axes = np.expand_dims(axes, axis=0)

    print(f"\n📊 Génération de la visualisation 1D : {filename} ...")

    for i, (env, name) in enumerate(zip(envs, env_names)):
        # Récupération directe des données (N, 2)
        # Pas besoin de w_true ou u ici, on regarde la donnée brute.
        X = env.X.cpu().numpy()
        Y = env.y.cpu().numpy().flatten()
        
        # Colonne 0 = Causal, Colonne 1 = Spurieux
        X_causal_raw = X[:, 0]
        X_spurious_raw = X[:, 1]
        
        # --- GRAPHIQUE 1 : Nuage de points (Scatter) ---
        ax_scatter = axes[i, 0]
        sns.scatterplot(
            x=X_causal_raw, 
            y=X_spurious_raw, 
            hue=Y, 
            palette={0: "blue", 1: "red"},
            alpha=0.6, 
            ax=ax_scatter
        )
        ax_scatter.set_title(f"{name} : Données Brutes (2D)")
        ax_scatter.set_xlabel("Feature Causale (x0)")
        ax_scatter.set_ylabel("Feature Spurieuse (x1)")
        ax_scatter.axvline(0, color='grey', linestyle='--', alpha=0.5)
        ax_scatter.legend(title="Label Y")

        # --- GRAPHIQUE 2 : Histogramme Spurieux (Axe Y) ---
        ax_hist = axes[i, 1]
        
        # On regarde la distribution de la feature spurieuse (colonne 1)
        # pour la classe Y=1
        sns.histplot(
            X_spurious_raw[Y == 1], 
            kde=True, 
            color="red", 
            stat="density", 
            bins=30, 
            ax=ax_hist,
            label="Y=1"
        )
        sns.histplot(
            X_spurious_raw[Y == 0], 
            kde=True, 
            color="blue", 
            stat="density", 
            bins=30, 
            ax=ax_hist,
            alpha=0.3,
            label="Y=0"
        )

        # p_spur = env.meta['p_spur']
        ax_hist.set_title(f"Distribution de la coordonnée spurieuse (x1)")
        ax_hist.set_xlabel("Valeur de x1")
        ax_hist.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("✅ Visualisation 1D terminée.")
    plt.close()


# =============================
# Visualisation des données multivariés
# =============================

def visualize_anti_causal_data(envs, env_names, filename="plot/data_distribution.png"):
    """
    Visualise la structure 'Mixture de Gaussiennes' des données anti-causales.
    Projette les données haute dimension sur les axes directeurs (w_true et u).
    """
    n_envs = len(envs)
    fig, axes = plt.subplots(n_envs, 2, figsize=(16, 5 * n_envs))
    if n_envs == 1: axes = np.expand_dims(axes, axis=0)

    print(f"\n📊 Génération de la visualisation des données : {filename} ...")

    for i, (env, name) in enumerate(zip(envs, env_names)):
        # 1. Récupération des données et conversion en Numpy
        X = env.X.cpu().numpy()
        Y = env.y.cpu().numpy().flatten()
        
        # 2. Récupération des métadonnées vitales pour la projection
        # dim_z est nécessaire pour séparer X_z de X_y
        dim_z = env.meta['dim_z']
        
        # Les vecteurs directeurs (shape: (dim,))
        w_true = env.meta['w_true'].flatten()
        u = env.meta['u'].flatten()
        
        # 3. Séparation Causal / Spurieux
        X_z_raw = X[:, :dim_z]
        X_y_raw = X[:, dim_z:]
        
        # 4. PROJECTION (Crucial pour voir quelque chose en haute dimension)
        # On calcule le "Score Causal" (alignement avec w_true)
        # et le "Score Spurieux" (alignement avec u)
        # Formule projection scalaire : (v . u) / ||u|| (ici vecteurs unitaires ou presque)
        causal_score = X_z_raw @ w_true 
        spurious_score = X_y_raw @ u 
        
        # --- GRAPHIQUE 1 : Scatter Plot (Causal vs Spurieux) ---
        ax_scatter = axes[i, 0]
        sns.scatterplot(
            x=causal_score, 
            y=spurious_score, 
            hue=Y, 
            palette={0: "blue", 1: "red"},
            alpha=0.5, 
            s=10,
            ax=ax_scatter
        )
        ax_scatter.set_title(f"{name} : Espace Latent Projeté")
        ax_scatter.set_xlabel(f"Score Causal ($X_z \\cdot w_{{true}}$)")
        ax_scatter.set_ylabel(f"Score Spurieux ($X_y \\cdot u$)")
        ax_scatter.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax_scatter.legend(title="Label Y")

        # --- GRAPHIQUE 2 : Histogramme des 'Bosses' (Score Spurieux | Y=1) ---
        ax_hist = axes[i, 1]
        
        # On isole les points où Y=1 (Les rouges)
        spurious_y1 = spurious_score[Y == 1]
        
        # On trace la distribution
        sns.histplot(
            spurious_y1, 
            kde=True, 
            color="red", 
            stat="density", 
            bins=40, 
            ax=ax_hist,
            label="Y=1"
        )
        
        # Pour voir le contraste, on met Y=0 en bleu (optionnel mais utile)
        spurious_y0 = spurious_score[Y == 0]
        sns.histplot(
            spurious_y0, 
            kde=True, 
            color="blue", 
            stat="density", 
            bins=40, 
            ax=ax_hist, 
            alpha=0.3,
            label="Y=0"
        )

        # Affichage du paramètre p_spur
        p_spur = env.meta['p_spur']
        ax_hist.set_title(f"{name} ($p_{{spur}}={p_spur}$) : Distribution Feature Spurieuse")
        ax_hist.set_xlabel(f"Projection sur u")
        ax_hist.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print("✅ Visualisation terminée.")
    plt.close()