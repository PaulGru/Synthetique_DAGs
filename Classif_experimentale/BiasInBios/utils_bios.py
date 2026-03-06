"""
utils_bios.py — Utilitaires d'évaluation et de visualisation pour Bias in Bios.

Métriques implémentées :
  - Accuracy (top-1)
  - F1-score macro / micro
  - Precision / Recall par classe
  - Matrice de confusion
  - F1 stratifié par genre (hommes vs femmes)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn

from data_bios import Env, N_CLASSES, IDX2PROF, PROFESSIONS


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
# Prédictions brutes
# =============================================================================

def _predict(model: nn.Module, env: Env, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (y_pred, y_true) en numpy."""
    model.eval()
    dev = torch.device(device)
    with torch.no_grad():
        logits = model(env.X.to(dev))
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = env.y.cpu().numpy()
    return preds, y_true


# =============================================================================
# Évaluation standard
# =============================================================================

def evaluate_multiclass(model: nn.Module, env: Env, device: str = "cpu") -> float:
    """Top-1 accuracy."""
    preds, y_true = _predict(model, env, device)
    return float((preds == y_true).mean())


def evaluate_f1_macro(model: nn.Module, env: Env, device: str = "cpu") -> float:
    """F1-score macro (non pondéré, pénalise les classes minoritaires)."""
    preds, y_true = _predict(model, env, device)
    return float(f1_score(y_true, preds, average="macro", zero_division=0))


def evaluate_group(model: nn.Module, envs: List[Env], device: str = "cpu") -> float:
    """Moyenne des accuracies sur plusieurs environnements."""
    if not envs:
        return 0.0
    return float(np.mean([evaluate_multiclass(model, e, device) for e in envs]))


def evaluate_group_f1(model: nn.Module, envs: List[Env], device: str = "cpu") -> float:
    """Moyenne des F1 macro sur plusieurs environnements."""
    if not envs:
        return 0.0
    return float(np.mean([evaluate_f1_macro(model, e, device) for e in envs]))


# =============================================================================
# Rapport complet
# =============================================================================

def evaluate_full_report(
    model: nn.Module,
    env: Env,
    device: str = "cpu",
) -> Dict:
    """
    Rapport complet pour un environnement :
    - accuracy, f1_macro, f1_micro
    - per_class : {nom_prof: {precision, recall, f1, support}}
    """
    preds, y_true = _predict(model, env, device)

    # Labels présents dans cet env
    labels = sorted(set(y_true.tolist()) | set(preds.tolist()))

    f1_mac = float(f1_score(y_true, preds, average="macro", zero_division=0))
    f1_mic = float(f1_score(y_true, preds, average="micro", zero_division=0))
    acc = float((preds == y_true).mean())

    report = classification_report(
        y_true, preds,
        labels=list(range(N_CLASSES)),
        target_names=PROFESSIONS,
        zero_division=0,
        output_dict=True,
    )

    per_class = {
        PROFESSIONS[k]: {
            "precision": report[PROFESSIONS[k]]["precision"],
            "recall":    report[PROFESSIONS[k]]["recall"],
            "f1":        report[PROFESSIONS[k]]["f1-score"],
            "support":   int(report[PROFESSIONS[k]]["support"]),
        }
        for k in range(N_CLASSES)
        if PROFESSIONS[k] in report
    }

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_micro": f1_mic,
        "per_class": per_class,
    }


# =============================================================================
# Évaluation stratifiée par genre
# =============================================================================

def evaluate_by_gender(
    model: nn.Module, env: Env, device: str = "cpu"
) -> Dict[str, float]:
    """Accuracy séparée pour les exemples masculins et féminins."""
    preds, y_true = _predict(model, env, device)
    g = env.meta.get("gender_array", None)

    if g is None or len(g) != len(y_true):
        return {"acc_male": float("nan"), "acc_female": float("nan"),
                "gap_acc": float("nan"), "f1_male": float("nan"),
                "f1_female": float("nan"), "gap_f1": float("nan")}

    mask_m = g == 0
    mask_f = g == 1

    acc_m = float((preds[mask_m] == y_true[mask_m]).mean()) if mask_m.any() else float("nan")
    acc_f = float((preds[mask_f] == y_true[mask_f]).mean()) if mask_f.any() else float("nan")

    f1_m = float(f1_score(y_true[mask_m], preds[mask_m],
                           average="macro", zero_division=0)) if mask_m.any() else float("nan")
    f1_f = float(f1_score(y_true[mask_f], preds[mask_f],
                           average="macro", zero_division=0)) if mask_f.any() else float("nan")

    gap_acc = abs(acc_f - acc_m) if not any(np.isnan([acc_m, acc_f])) else float("nan")
    gap_f1  = abs(f1_f - f1_m)  if not any(np.isnan([f1_m, f1_f]))  else float("nan")

    return {
        "acc_male":   acc_m,
        "acc_female": acc_f,
        "gap_acc":    gap_acc,
        "f1_male":    f1_m,
        "f1_female":  f1_f,
        "gap_f1":     gap_f1,
    }


# =============================================================================
# Logging périodique
# =============================================================================

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

    tr_acc  = evaluate_group(model, train_envs, device)
    va_acc  = evaluate_group(model, val_envs, device)
    te_acc  = evaluate_multiclass(model, test_env, device)
    te_f1   = evaluate_f1_macro(model, test_env, device)
    gs      = evaluate_by_gender(model, test_env, device)

    parts.append(f"Train acc={tr_acc:.3f}")
    parts.append(f"Val acc={va_acc:.3f}")
    parts.append(f"Test acc={te_acc:.3f} | F1={te_f1:.3f}")
    if not np.isnan(gs["gap_f1"]):
        parts.append(f"GenderGap(f1)={gs['gap_f1']:.3f}")

    print(" | ".join(parts))


# =============================================================================
# Résumé final complet
# =============================================================================

def print_full_summary(tag: str, report: Dict, gender_stats: Dict):
    print(f"\n{'='*60}")
    print(f"  {tag} — Résumé complet (Test OOD)")
    print(f"{'='*60}")
    print(f"  Accuracy      : {report['accuracy']:.4f}")
    print(f"  F1 Macro      : {report['f1_macro']:.4f}")
    print(f"  F1 Micro      : {report['f1_micro']:.4f}")
    print(f"\n  Par genre :")
    print(f"    Hommes  — acc={gender_stats['acc_male']:.3f}  F1={gender_stats['f1_male']:.3f}")
    print(f"    Femmes  — acc={gender_stats['acc_female']:.3f}  F1={gender_stats['f1_female']:.3f}")
    print(f"    Gap acc : {gender_stats['gap_acc']:.3f} | Gap F1 : {gender_stats['gap_f1']:.3f}")
    print(f"\n  Par profession (F1 / precision / recall / support):")
    per_class = report["per_class"]
    # Tri par F1 croissant
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]["f1"])
    print(f"  {'Profession':<22} {'F1':>6} {'Prec':>7} {'Rec':>7} {'N':>6}")
    print(f"  {'-'*52}")
    for prof, m in sorted_classes:
        if m["support"] > 0:
            print(f"  {prof:<22} {m['f1']:>6.3f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['support']:>6}")
    print(f"{'='*60}")


# =============================================================================
# Visualisation
# =============================================================================

def plot_confusion_matrix(
    model: nn.Module,
    env: Env,
    tag: str,
    device: str = "cpu",
    outdir: str = "plots",
):
    """Matrice de confusion normalisée (par ligne = vraie classe)."""
    os.makedirs(outdir, exist_ok=True)
    preds, y_true = _predict(model, env, device)

    # Garder uniquement les classes qui ont du support
    classes_present = sorted(set(y_true.tolist()))
    labels = [PROFESSIONS[k] for k in classes_present]

    cm = confusion_matrix(y_true, preds, labels=classes_present, normalize="true")

    fig, ax = plt.subplots(figsize=(len(labels) * 1.0 + 1.5, len(labels) * 0.8 + 1.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai label")
    ax.set_title(f"{tag} — Matrice de confusion (Test OOD, normalisée)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    out_path = os.path.join(outdir, f"confusion_{tag.lower()}.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Matrice de confusion sauvegardée : {out_path}")
    plt.close()


def plot_f1_per_class_comparison(
    erm_model: Optional[nn.Module],
    irm_model: Optional[nn.Module],
    env: Env,
    device: str = "cpu",
    outdir: str = "plots",
):
    """Barplot côte-à-côte du F1 par profession pour ERM et IRM."""
    os.makedirs(outdir, exist_ok=True)

    f1_erm = {}
    f1_irm = {}

    if erm_model:
        rep = evaluate_full_report(erm_model, env, device)
        f1_erm = {k: v["f1"] for k, v in rep["per_class"].items()}
    if irm_model:
        rep = evaluate_full_report(irm_model, env, device)
        f1_irm = {k: v["f1"] for k, v in rep["per_class"].items()}

    # Professions avec support > 0
    all_profs = sorted(
        set(f1_erm.keys()) | set(f1_irm.keys()),
        key=lambda p: (f1_erm.get(p, 0) + f1_irm.get(p, 0)) / 2
    )

    x = np.arange(len(all_profs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(all_profs) * 1.2), 6))
    if erm_model:
        ax.bar(x - width/2, [f1_erm.get(p, 0) for p in all_profs],
               width, label="ERM", color="tab:blue", alpha=0.85)
    if irm_model:
        ax.bar(x + width/2, [f1_irm.get(p, 0) for p in all_profs],
               width, label="IRM", color="tab:orange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_profs, rotation=40, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-score par profession — ERM vs IRM (Test OOD)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(outdir, "f1_per_class_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ F1 par classe sauvegardé : {out_path}")
    plt.close()


def plot_results(
    erm_history: Dict,
    irm_history: Dict,
    outdir: str = "plots",
):
    """Courbes d'accuracy / F1 macro et gender gap au cours du temps."""
    os.makedirs(outdir, exist_ok=True)

    has_f1 = bool(erm_history.get("test_f1") or irm_history.get("test_f1"))
    n_plots = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle("Bias in Bios — ERM vs IRM", fontsize=14)

    for tag, hist, color in [("ERM", erm_history, "tab:blue"), ("IRM", irm_history, "tab:orange")]:
        if not hist["step"]:
            continue
        axes[0].plot(hist["step"], hist["train_acc"], label=f"{tag} Train", color=color, linestyle="-")
        axes[0].plot(hist["step"], hist["val_acc"],   label=f"{tag} Val",   color=color, linestyle="--")
    axes[0].set_title("Accuracy In-Distribution")
    axes[0].set_xlabel("Steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for tag, hist, color in [("ERM", erm_history, "tab:blue"), ("IRM", irm_history, "tab:orange")]:
        if not hist["step"]:
            continue
        axes[1].plot(hist["step"], hist["test_acc"], label=f"{tag} acc", color=color)
        if has_f1 and hist.get("test_f1"):
            axes[1].plot(hist["step"], hist["test_f1"], label=f"{tag} F1",  color=color, linestyle="--")
    axes[1].set_title("Test OOD")
    axes[1].set_xlabel("Steps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_f1:
        for tag, hist, color in [("ERM", erm_history, "tab:blue"), ("IRM", irm_history, "tab:orange")]:
            if not hist["step"] or not hist.get("gender_gap_f1"):
                continue
            axes[2].plot(hist["step"], hist["gender_gap_f1"], label=tag, color=color)
        axes[2].set_title("Gender Gap F1 (Test OOD) |F1_F - F1_M|")
        axes[2].set_xlabel("Steps")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(outdir, "erm_vs_irm_bios.png")
    plt.savefig(out_path, dpi=150)
    print(f"✅ Plot sauvegardé : {out_path}")
    plt.close()
