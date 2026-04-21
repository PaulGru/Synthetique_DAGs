#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

"""
visualize_moji_5models.py
=========================
Visualisation complète des modèles Moji.

  M1  ERM frozen      (DistilBERT gelé + tête ERM)
  M2  IRM frozen      (DistilBERT gelé + tête IRM)
  M3  ERM fine-tuned  (DistilBERT fine-tuné end-to-end)
  M5  IRM + FT        (backbone M3 gelé + tête IRM sur E1+E2)

Plots générés :
  01_summary_test_ood.png     — Bar chart global (6 métriques × 4 modèles)
  02_per_group_accuracy.png   — Accuracy par groupe (4 groupes × 4 modèles)
  03_fairness_gaps.png        — TPR / FPR par dialecte (AAE vs SAE)
  04_training_curves_m3.png   — Courbes d'entraînement M3 (par epoch)
  05_training_curves_m5.png   — Courbes convergence tête M5 (par step)
  06_heatmap.png              — Heatmap métriques × modèles

Usage :
    uv run visualize_moji_5models.py
    uv run visualize_moji_5models.py --out_dir plots_moji_models
"""


import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Palette & helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "M1": "#e05252",   # rouge  — ERM frozen
    "M2": "#4caf7d",   # vert   — IRM frozen
    "M3": "#5b8dd9",   # bleu   — ERM fine-tuned
    "M5": "#9b59b6",   # violet — IRM + FT
    "aae": "#5b8dd9",
    "sae": "#f5a623",
}

MODEL_LABELS = {
    "M1": "M1 — ERM frozen",
    "M2": "M2 — IRM frozen",
    "M3": "M3 — ERM fine-tuned",
    "M5": "M5 — IRM fine-tuned",
}

GROUP_KEYS   = ["(Y=0,A=0)", "(Y=0,A=1)", "(Y=1,A=0)", "(Y=1,A=1)"]
GROUP_LABELS = ["Neg SAE", "Neg AAE", "Pos SAE", "Pos AAE"]
GROUP_COLORS = ["#f0a500", "#5b8dd9", "#e05252", "#4caf7d"]


def _annotate(ax, bar, v, fmt=".3f", dy=0.012, fontsize=8):
    if not np.isnan(v):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{v:{fmt}}", ha="center", va="bottom", fontsize=fontsize,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────────────────────

def load_all(
    m12_path="plots_moji/results.json",
    m3_path="logs/finetuned_erm/results.json",
    m5_path="logs_irm_ft/results.json",
    m3_hist_path="logs/finetuned_erm/history.json",
    m5_hist_path="logs_irm_ft/history_irm_ft.json",
):
    with open(m12_path)  as f: d12 = json.load(f)
    with open(m3_path)   as f: d3  = json.load(f)
    with open(m5_path)   as f: d5  = json.load(f)

    # Harmonise : chaque modèle → {"val_ind": {...}, "test_ood": {...}}
    models = {
        "M1": {"val_ind": d12["erm"]["val_ind"],    "test_ood": d12["erm"]["test_ood"]},
        "M2": {"val_ind": d12["irm"]["val_ind"],    "test_ood": d12["irm"]["test_ood"]},
        "M3": {"val_ind": d3.get("val_ind", {}),    "test_ood": d3.get("test_ood", {})},
        "M5": {"val_ind": d5["irm_ft"]["val_ind"],  "test_ood": d5["irm_ft"]["test_ood"]},
    }

    histories = {}
    for key, path in [("M3", m3_hist_path), ("M5", m5_hist_path)]:
        if os.path.exists(path):
            with open(path) as f:
                histories[key] = json.load(f)

    return models, histories


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Résumé comparatif Test OOD
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(models: dict, out_dir: str):
    """
    01_summary_test_ood.png
    Grouped bar chart : 6 métriques principales × 5 modèles sur Test OOD.
    Pour FNR et EOD : plus bas = mieux (annotées d'une flèche ↓).
    Pour les autres : plus haut = mieux.
    """
    metrics = [
        ("accuracy",        "Accuracy",              True),
        ("macro_f1",        "Macro-F1",              True),
        ("worst_group_acc", "Worst-Group Acc",       True),
        ("avg_group_acc",   "Avg-Group Acc",         True),
        ("fnr_pos_aae",     "FNR (Y=1,A=1) ↓",      False),
        ("eod_tpr",         "EOD TPR ↓",             False),
    ]

    model_keys = list(models.keys())
    hatches    = ["", "//", "..", "xx", "--"]
    n_metrics  = len(metrics)
    n_models   = len(model_keys)
    x = np.arange(n_metrics)
    width = 0.14

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, mk in enumerate(model_keys):
        vals = [models[mk]["test_ood"].get(m, float("nan")) for m, _, _ in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      color=COLORS[mk], hatch=hatches[i],
                      edgecolor="white", linewidth=0.5, alpha=0.88)
        for bar, v in zip(bars, vals):
            _annotate(ax, bar, v, fmt=".3f", dy=0.005, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in metrics], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    legend_patches = [
        mpatches.Patch(color=COLORS[mk], hatch=h, label=MODEL_LABELS[mk], alpha=0.88)
        for mk, h in zip(model_keys, hatches)
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Annotations flèche "lower is better"
    for xi, (_, _, higher_better) in enumerate(metrics):
        if not higher_better:
            ax.annotate("↓ mieux", xy=(xi, 1.08), ha="center",
                        fontsize=8, color="#c0392b", fontweight="bold")

    fig.suptitle(
        "01 — Comparaison des 5 modèles — Test OOD (équilibré, 4×7500)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "01_summary_test_ood.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [01] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Per-group accuracy (4 groupes × 5 modèles)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_group(models: dict, out_dir: str):
    """
    02_per_group_accuracy.png
    Val InD (gauche) et Test OOD (droite) : accuracy par groupe × modèle.
    """
    model_keys = list(models.keys())
    hatches = ["", "//", "..", "xx", "--"]
    n = len(model_keys)
    width = 0.14
    x = np.arange(len(GROUP_KEYS))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, split_key, split_title in [
        (axes[0], "val_ind",  "Val InD (biaisé 80/20)"),
        (axes[1], "test_ood", "Test OOD (équilibré)"),
    ]:
        for i, mk in enumerate(model_keys):
            vals = [
                models[mk][split_key].get("acc_groups", {}).get(gk, float("nan"))
                for gk in GROUP_KEYS
            ]
            offset = (i - n / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          color=COLORS[mk], hatch=hatches[i],
                          label=MODEL_LABELS[mk],
                          edgecolor="white", linewidth=0.4, alpha=0.88)
            for bar, v in zip(bars, vals):
                _annotate(ax, bar, v, fmt=".2f", dy=0.006, fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_LABELS, fontsize=10)
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Group Accuracy", fontsize=10)
        ax.set_title(split_title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)
        ax.grid(axis="y", alpha=0.3)

        # Couleur de fond par groupe
        for gi, gc in enumerate(GROUP_COLORS):
            ax.axvspan(gi - 0.5, gi + 0.5, color=gc, alpha=0.04, zorder=0)

    fig.suptitle(
        "02 — Accuracy par groupe (Y×A) — 5 modèles comparés\n"
        "Un modèle équitable produit des barres de hauteur uniforme",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "02_per_group_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [02] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Fairness gaps : TPR & FPR par dialecte
# ─────────────────────────────────────────────────────────────────────────────

def plot_fairness_gaps(models: dict, out_dir: str):
    """
    03_fairness_gaps.png
    2×2 : (TPR | FPR) × (Val InD | Test OOD) — barres AAE/SAE par modèle.
    """
    model_keys = list(models.keys())
    hatches = ["", "//", "..", "xx", "--"]
    n = len(model_keys)
    width = 0.12
    x = np.array([0.0, 1.0])   # AAE vs SAE

    def _rates(r):
        fnr = r.get("fnr_pos_aae", float("nan"))
        tpr_aae = 1.0 - fnr if not np.isnan(fnr) else float("nan")
        ag = r.get("acc_groups", {})
        tpr_sae = ag.get("(Y=1,A=0)", float("nan"))
        fpr_aae = 1.0 - ag.get("(Y=0,A=1)", float("nan"))
        fpr_sae = r.get("fpr_neg_sae", float("nan"))
        return tpr_aae, tpr_sae, fpr_aae, fpr_sae

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, (split_key, split_title) in enumerate([
        ("val_ind",  "Val InD"),
        ("test_ood", "Test OOD"),
    ]):
        for row, (rate_idx, ylabel, higher_better) in enumerate([
            ((0, 1), "True Positive Rate (TPR)\nhigher = fewer missed positives", True),
            ((2, 3), "False Positive Rate (FPR)\nlower = fewer false alarms", False),
        ]):
            ax = axes[row][col]

            for i, mk in enumerate(model_keys):
                rates = _rates(models[mk][split_key])
                aae_val = rates[rate_idx[0]]
                sae_val = rates[rate_idx[1]]
                offset  = (i - n / 2 + 0.5) * width

                for xi, val, lbl in [(0, aae_val, "AAE"), (1, sae_val, "SAE")]:
                    bar = ax.bar(
                        xi + offset, val, width,
                        color=COLORS[mk], hatch=hatches[i], alpha=0.88,
                        edgecolor="white", linewidth=0.4,
                        label=MODEL_LABELS[mk] if xi == 0 else "_",
                    )
                    _annotate(ax, bar[0], val, fmt=".3f", dy=0.010, fontsize=6.5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["AAE", "SAE"], fontsize=11)
            ax.set_ylim(0, 1.28)
            ax.grid(axis="y", alpha=0.3)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(split_title, fontsize=11, fontweight="bold")

            # Annotation "mieux"
            lbl_better = "↑ mieux" if higher_better else "↓ mieux"
            ax.set_title(
                (split_title if row == 0 else "") +
                (f"\n{lbl_better}" if True else ""),
                fontsize=10, fontweight="bold",
            )

            # Légende une seule fois
            if col == 0 and row == 0:
                model_patches = [
                    mpatches.Patch(color=COLORS[mk], hatch=h, label=MODEL_LABELS[mk], alpha=0.85)
                    for mk, h in zip(model_keys, hatches)
                ]
                ax.legend(handles=model_patches, fontsize=7, loc="upper right", ncol=1)

    fig.suptitle(
        "03 — Fairness gaps TPR/FPR par dialecte (AAE vs SAE) — 5 modèles\n"
        "Un modèle équitable a des barres AAE ≈ SAE",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "03_fairness_gaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [03] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Courbes d'entraînement M3 (par epoch)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_m3(hist: dict, out_dir: str):
    """
    04_training_curves_m3.png
    Évolution par epoch : train loss, acc (train/val/test), worst-group.
    """
    epochs = hist["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], color=COLORS["M3"], lw=2, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (BCE)")
    ax.set_title("Train Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # Panel 2 — Accuracy
    ax = axes[1]
    ax.plot(epochs, hist["train_acc"], color=COLORS["M3"], lw=2, marker="o",
            label="Train acc")
    ax.plot(epochs, hist["val_acc"], color=COLORS["M3"], lw=2, marker="s",
            ls="--", label="Val InD acc")
    ax.plot(epochs, hist["test_acc"], color="#e74c3c", lw=2, marker="^",
            ls=":", label="Test OOD acc")
    best_epoch = epochs[int(np.argmax(hist["val_acc"]))]
    ax.axvline(best_epoch, color="#888", ls="--", lw=1.2, alpha=0.7,
               label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy — Train / Val InD / Test OOD", fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # Panel 3 — Worst-group
    ax = axes[2]
    ax.plot(epochs, hist["val_worst_group"], color="#9b59b6", lw=2, marker="o",
            label="Val InD — worst group")
    ax.plot(epochs, hist["test_worst_group"], color="#e74c3c", lw=2, marker="s",
            ls="--", label="Test OOD — worst group")
    ax.plot(epochs, hist["val_macro_f1"], color="#9b59b6", lw=1.5, marker="^",
            ls=":", alpha=0.7, label="Val InD — Macro-F1")
    ax.axvline(best_epoch, color="#888", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Worst-Group Accuracy & Macro-F1", fontweight="bold")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    fig.suptitle(
        "04 — Modèle 3 : courbes d'entraînement (fine-tune DistilBERT ERM)\n"
        "Le backbone fine-tuné mémorise progressivement le biais E1/E2 → déclin OOD après epoch 2",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "04_training_curves_m3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [04] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Convergence tête M5 (IRM+FT) par step
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_m5(hist_m5: dict, out_dir: str):
    """
    05_training_curves_m5.png
    Convergence de la tête de classification de M5 (IRM+FT).
    """
    hist  = hist_m5
    color = COLORS["M5"]
    steps = hist["step"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — Loss
    ax = axes[0]
    ax.plot(steps, hist["loss"], color=color, lw=1.5, alpha=0.9)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("M5 — Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2 — Train / Val / Test accuracy
    ax = axes[1]
    ax.plot(steps, hist["train_acc"], color=color, lw=1.5, label="Train")
    ax.plot(steps, hist["val_acc"],   color=color, lw=1.5, ls="--",
            alpha=0.8, label="Val InD")
    ax.plot(steps, hist["test_acc"],  color="#e74c3c", lw=1.5, ls=":",
            label="Test OOD")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("M5 — Accuracy", fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Poids (w_z et w_y = scalaire de la tête LogReg)
    ax = axes[2]
    ax.plot(steps, hist["w_z"], color=color, lw=1.5, label="|W| (norme)")
    ax.plot(steps, hist["w_y"], color="#555", lw=1.5, ls="--", alpha=0.7,
            label="biais")
    ax.set_xlabel("Step")
    ax.set_ylabel("Valeur")
    ax.set_title("M5 — Poids tête LogReg", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "05 — Convergence de la tête M5 (IRM+FT)\n"
        "Backbone M3 gelé — seule la tête LogReg est entraînée",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "05_training_curves_m5.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [05] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Heatmap métriques × modèles
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(models: dict, out_dir: str):
    """
    06_heatmap.png
    Heatmap normalisée (meilleur=vert, pire=rouge) sur Test OOD.
    Permet de voir d'un coup d'œil quel modèle excelle sur quelle métrique.
    """
    metrics_info = [
        ("accuracy",        "Accuracy",              True),
        ("macro_f1",        "Macro-F1",              True),
        ("worst_group_acc", "Worst-Group",           True),
        ("avg_group_acc",   "Avg-Group",             True),
        ("fnr_pos_aae",     "FNR (AAE pos) ↓",       False),
        ("fpr_neg_sae",     "FPR (SAE neg) ↓",       False),
        ("eod_tpr",         "EOD TPR ↓",             False),
        ("eod_fpr",         "EOD FPR ↓",             False),
    ]

    model_keys = list(models.keys())
    row_labels  = [lab for _, lab, _ in metrics_info]
    col_labels  = [MODEL_LABELS[mk].replace(" — ", "\n") for mk in model_keys]

    data = np.zeros((len(metrics_info), len(model_keys)))
    for j, mk in enumerate(model_keys):
        ood = models[mk]["test_ood"]
        for i, (metric, _, higher_better) in enumerate(metrics_info):
            if metric.startswith("acc_groups."):
                v = ood.get("acc_groups", {}).get(metric[11:], float("nan"))
            else:
                v = ood.get(metric, float("nan"))
            # Normalise : 1 = meilleur, 0 = pire
            data[i, j] = v if not np.isnan(v) else 0.0

    # Normalise ligne par ligne en tenant compte du sens
    data_norm = np.zeros_like(data)
    for i, (_, _, higher_better) in enumerate(metrics_info):
        row = data[i]
        mn, mx = row.min(), row.max()
        if mx > mn:
            normalized = (row - mn) / (mx - mn)
            data_norm[i] = normalized if higher_better else 1.0 - normalized
        else:
            data_norm[i] = 0.5

    n_frozen = sum(1 for mk in model_keys if mk in ("M1", "M2"))

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(data_norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Annotations valeurs brutes
    for i in range(len(metrics_info)):
        for j in range(len(model_keys)):
            v = data[i, j]
            brightness = data_norm[i, j]
            text_color = "black" if 0.25 < brightness < 0.75 else "white"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(metrics_info)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Séparateur entre frozen et fine-tuned : trait blanc épais
    ax.axvline(n_frozen - 0.5, color="white", lw=6)

    # Labels de groupe au-dessus de la heatmap (coordonnées data)
    ax.text((n_frozen - 1) / 2, -0.7, "BERT frozen",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#555")
    ax.text(n_frozen + (len(model_keys) - n_frozen - 1) / 2, -0.7, "BERT fine-tuned",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#555")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "06_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [06] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES TEXTE — résumé console
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(models: dict):
    metrics = [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro-F1"),
        ("worst_group_acc", "Worst-Group Acc"),
        ("avg_group_acc",   "Avg-Group Acc"),
        ("fnr_pos_aae",     "FNR (Y=1,A=1) AAE"),
        ("fpr_neg_sae",     "FPR (Y=0,A=0) SAE"),
        ("eod_tpr",         "EOD TPR"),
        ("eod_fpr",         "EOD FPR"),
    ]
    model_keys = list(models.keys())
    header = f"  {'Metric':<28}" + "".join(f" {mk:>10}" for mk in model_keys)
    sep = "  " + "-" * (28 + 11 * len(model_keys))

    for split_key, split_title in [("test_ood", "Test OOD"), ("val_ind", "Val InD")]:
        print(f"\n{'='*70}")
        print(f"  {split_title}")
        print("="*70)
        print(header)
        print(sep)
        for metric, label in metrics:
            row = f"  {label:<28}"
            for mk in model_keys:
                v = models[mk][split_key].get(metric, float("nan"))
                row += f" {v:>10.4f}"
            print(row)
        print(f"\n  {'Acc par groupe':<28}")
        for gk, gl in zip(GROUP_KEYS, GROUP_LABELS):
            row = f"  {'  '+gl:<28}"
            for mk in model_keys:
                v = models[mk][split_key].get("acc_groups", {}).get(gk, float("nan"))
                row += f" {v:>10.4f}"
            print(row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualisation 5 modèles Moji")
    parser.add_argument("--m12_results",
                        default=str(_Path(__file__).parent / "plots" / "moji" / "results.json"))
    parser.add_argument("--m3_results",
                        default=str(_Path(__file__).parent / "logs" / "finetuned_erm" / "results.json"))
    parser.add_argument("--m5_results",
                        default=str(_Path(__file__).parent / "logs_irm_ft" / "results.json"))
    parser.add_argument("--m3_history",
                        default=str(_Path(__file__).parent / "logs" / "finetuned_erm" / "history.json"))
    parser.add_argument("--m5_history",
                        default=str(_Path(__file__).parent / "logs_irm_ft" / "history_irm_ft.json"))
    parser.add_argument("--out_dir",
                        default=str(_Path(__file__).parent / "plots" / "5models"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Chargement des résultats…")
    models, histories = load_all(
        m12_path=args.m12_results,
        m3_path=args.m3_results,
        m5_path=args.m5_results,
        m3_hist_path=args.m3_history,
        m5_hist_path=args.m5_history,
    )

    print_summary(models)

    print(f"\nGénération des plots dans {args.out_dir}/…")
    plot_summary(models, args.out_dir)
    plot_per_group(models, args.out_dir)
    plot_fairness_gaps(models, args.out_dir)

    if "M3" in histories:
        plot_training_m3(histories["M3"], args.out_dir)
    else:
        print("  [04] SKIP — history M3 introuvable")

    if "M5" in histories:
        plot_training_m5(histories["M5"], args.out_dir)
    else:
        print("  [05] SKIP — history M5 introuvable")

    plot_heatmap(models, args.out_dir)

    print(f"\nDone! Tous les plots sauvegardés dans {args.out_dir}/")


if __name__ == "__main__":
    main()
