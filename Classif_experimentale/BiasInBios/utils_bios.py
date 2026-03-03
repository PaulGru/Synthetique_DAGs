"""
utils_bios.py — Utilitaires d'évaluation et de visualisation pour Bias in Bios.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_bios import Env, N_CLASSES, IDX2PROF


# =============================================================================
# Device
# =============================================================================

def resolve_device(d: str) -> str:
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"
    return d


# =============================================================================
# Évaluation
# =============================================================================

def evaluate_multiclass(
    model: nn.Module, env: Env, device: str = "cpu"
) -> float:
    """Top-1 accuracy sur un environnement multi-classes."""
    model.eval()
    dev = torch.device(device)
    with torch.no_grad():
        logits = model(env.X.to(dev))  # (N, C)
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = env.y.cpu().numpy()
    return float((preds == y_true).mean())


def evaluate_group(
    model: nn.Module, envs: List[Env], device: str = "cpu"
) -> float:
    """Moyenne des accuracies sur plusieurs environnements."""
    if not envs:
        return 0.0
    return float(np.mean([evaluate_multiclass(model, e, device) for e in envs]))


def evaluate_by_gender(
    model: nn.Module, env: Env, device: str = "cpu"
) -> Dict[str, float]:
    """
    Accuracy séparée pour les exemples masculins et féminins.

    Utilise meta['gender_array'] stocké à la construction des Env.
    """
    model.eval()
    dev = torch.device(device)
    with torch.no_grad():
        logits = model(env.X.to(dev))
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = env.y.cpu().numpy()
    g = env.meta.get("gender_array", None)

    if g is None or len(g) != len(y_true):
        return {"acc_male": float("nan"), "acc_female": float("nan"), "gap": float("nan")}

    mask_m = g == 0
    mask_f = g == 1
    acc_m = float((preds[mask_m] == y_true[mask_m]).mean()) if mask_m.any() else float("nan")
    acc_f = float((preds[mask_f] == y_true[mask_f]).mean()) if mask_f.any() else float("nan")
    gap = abs(acc_f - acc_m) if not (np.isnan(acc_m) or np.isnan(acc_f)) else float("nan")
    return {"acc_male": acc_m, "acc_female": acc_f, "gap": gap}


def evaluate_and_log_step(
    tag: str,
    step: int,
    model: nn.Module,
    train_envs: List[Env],
    val_envs: List[Env],
    test_env: Env,
    device: str = "cpu",
    loss_val: Optional[float] = None,
):
    parts = [f"[{tag}] step {step}"]
    if loss_val is not None:
        parts.append(f"loss={loss_val:.4f}")

    tr_acc = evaluate_group(model, train_envs, device)
    parts.append(f"Train(ID): acc={tr_acc:.3f}")

    va_acc = evaluate_group(model, val_envs, device)
    parts.append(f"Val(ID): acc={va_acc:.3f}")

    te_acc = evaluate_multiclass(model, test_env, device)
    parts.append(f"Test(OOD): acc={te_acc:.3f}")

    # Gender gap sur le test OOD
    gender_stats = evaluate_by_gender(model, test_env, device)
    gap = gender_stats["gap"]
    if not np.isnan(gap):
        parts.append(f"GenderGap(OOD): {gap:.3f}")

    print(" | ".join(parts))


# =============================================================================
# Visualisation
# =============================================================================

def plot_results(
    erm_history: Dict,
    irm_history: Dict,
    outdir: str = "plots",
):
    """
    Trace les courbes d'accuracy Train/Val/Test pour ERM et IRM,
    ainsi que le gender gap sur le test OOD.
    """
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Bias in Bios — ERM vs IRM", fontsize=14)

    for tag, hist, color in [("ERM", erm_history, "tab:blue"), ("IRM", irm_history, "tab:orange")]:
        steps = hist["step"]
        axes[0].plot(steps, hist["train_acc"], label=f"{tag} Train", color=color, linestyle="-")
        axes[0].plot(steps, hist["val_acc"], label=f"{tag} Val", color=color, linestyle="--")
    axes[0].set_title("Accuracy In-Distribution")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for tag, hist, color in [("ERM", erm_history, "tab:blue"), ("IRM", irm_history, "tab:orange")]:
        axes[1].plot(hist["step"], hist["test_acc"], label=tag, color=color)
    axes[1].set_title("Accuracy Test OOD")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for tag, hist, color in [("ERM", erm_history, "tab:blue"), ("IRM", irm_history, "tab:orange")]:
        if "gender_gap" in hist and hist["gender_gap"]:
            axes[2].plot(hist["step"], hist["gender_gap"], label=tag, color=color)
    axes[2].set_title("Gender Gap (Test OOD) |acc_F - acc_M|")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("|acc_F - acc_M|")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(outdir, "erm_vs_irm_bios.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Plot sauvegardé : {out_path}")
    plt.close()


def plot_per_class_accuracy(
    model: nn.Module,
    env: Env,
    tag: str,
    device: str = "cpu",
    outdir: str = "plots",
):
    """
    Barplot de l'accuracy par profession pour un modèle donné.
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    dev = torch.device(device)
    with torch.no_grad():
        logits = model(env.X.to(dev))
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = env.y.cpu().numpy()

    accs = []
    names = []
    for k in range(N_CLASSES):
        mask = y_true == k
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == k).mean())
        names.append(IDX2PROF[k])

    order = np.argsort(accs)
    accs = [accs[i] for i in order]
    names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, accs, color="steelblue")
    ax.set_xlabel("Accuracy")
    ax.set_title(f"{tag} — Accuracy par profession (Test OOD)")
    ax.axvline(np.mean(accs), color="red", linestyle="--", label=f"Moyenne={np.mean(accs):.3f}")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, f"per_class_{tag.lower()}.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Plot par classe sauvegardé : {out_path}")
    plt.close()
