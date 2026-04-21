#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

"""
run_civilcomments_bias_envs.py
==============================
Compare ERM vs IRM (IRMv1) sur CivilComments avec des environnements
construits selon la **force de la corrélation spurieuse** identité→toxicité.

Intuition IRM : la corrélation "citer une identité → toxicité" varie entre
environnements. En séparant les identités par niveau de biais, IRM est forcé
d'ignorer cette corrélation pour généraliser uniformément.

Environnements d'entraînement (2, assignment exclusif) :
    Env "high_bias"   : exemples dont l'identité principale a taux tox ≥ seuil
                        (black 31 %, white 28 %, LGBTQ 27 %, muslim 22 %)
                        → corrélation spurieuse forte
    Env "low_bias"    : tous les autres exemples (faiblement ou non identifiés)
                        (christian 9 %, female 14 %, male 15 %, others 15 %, ∅)
                        → corrélation spurieuse faible ou nulle

Assignment : chaque exemple va dans UN seul environnement.
    Priorité  high_bias > low_bias > no_identity

Évaluation sur les 16 groupes habituels (8 identités × 2 labels).

Usage :
    uv run run_civilcomments_bias_envs.py --device auto
    uv run run_civilcomments_bias_envs.py --device auto --bias_threshold 0.20
    uv run run_civilcomments_bias_envs.py --device auto --max_per_env 5000
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_synth import Env
from models_training import train_erm, train_irm
from utils_irm import resolve_device

# =============================================================================
# Configuration des 16 groupes d'évaluation (identique à identity_envs)
# =============================================================================

IDENTITY_COLUMNS = [
    "male", "female", "christian", "muslim",
    "other_religions", "black", "white", "LGBTQ",
]

IDENTITY_LABELS = {
    "male":            "Homme",
    "female":          "Femme",
    "christian":       "Chrétien",
    "muslim":          "Musulman",
    "other_religions": "Autres relig.",
    "black":           "Noir",
    "white":           "Blanc",
    "LGBTQ":           "LGBTQ+",
}

GROUP_NAMES: List[str] = []
GROUP_SPECS: List[Tuple[str, int]] = []
for _id in IDENTITY_COLUMNS:
    GROUP_NAMES.append(f"{IDENTITY_LABELS[_id]} + Civil")
    GROUP_SPECS.append((_id, 0))
    GROUP_NAMES.append(f"{IDENTITY_LABELS[_id]} + Toxique")
    GROUP_SPECS.append((_id, 1))


# =============================================================================
# Téléchargement / chargement du CSV
# =============================================================================

_WILDS_URL = (
    "https://worksheets.codalab.org/rest/bundles/"
    "0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/"
)


def _find_csv(root_dir: str) -> str | None:
    for candidate in [
        os.path.join(root_dir, "all_data_with_identities.csv"),
        os.path.join(root_dir, "civilcomments_v1.0", "all_data_with_identities.csv"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _download_civilcomments_no_ssl(root_dir: str) -> str:
    import ssl
    import tarfile
    import urllib.request

    csv_path = _find_csv(root_dir)
    if csv_path is not None:
        return csv_path

    os.makedirs(root_dir, exist_ok=True)
    archive_path = os.path.join(root_dir, "civilcomments_archive.tar.gz")
    print(f"  Téléchargement depuis {_WILDS_URL}")
    print("  (SSL non vérifié — certificat codalab.org expiré)")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(_WILDS_URL, context=ctx) as resp:
        total = int(resp.getheader("Content-Length", 0))
        downloaded = 0
        with open(archive_path, "wb") as fout:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fout.write(chunk)
                downloaded += len(chunk)
                mb_done = downloaded / 1_048_576
                mb_total_str = f"{total / 1_048_576:.1f}" if total else "?"
                print(f"\r  {mb_done:.1f} / {mb_total_str} MB", end="", flush=True)
    print()

    print("  Extraction …")
    with tarfile.open(archive_path) as tf:
        tf.extractall(root_dir, filter="data")
    os.remove(archive_path)

    csv_path = _find_csv(root_dir)
    if csv_path is None:
        raise FileNotFoundError(f"CSV introuvable dans {root_dir} après extraction.")
    print(f"  Dataset prêt : {csv_path}")
    return csv_path


def load_dataframe(root_dir: str = "."):
    import pandas as pd
    csv_path = _download_civilcomments_no_ssl(root_dir)
    print(f"Chargement du CSV : {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    df["Y"] = (df["toxicity"] >= 0.5).astype(int)
    return df


# =============================================================================
# Embeddings DistilBERT (gelé, mean pooling)
# =============================================================================

def embed_texts(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """DistilBERT gelé → vecteur (N, 768) par texte."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False
    bert = bert.to(device)

    # Forcer str et remplacer les NaN/None par une chaîne vide
    texts = [t if isinstance(t, str) else ("" if t != t else str(t)) for t in texts]

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            out = bert(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device))
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].to(device).unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_emb.append(emb.cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(all_emb, axis=0).astype(np.float32)


def _make_env(
    embeddings: np.ndarray,
    Y: np.ndarray,
    meta: Optional[dict] = None,
) -> Env:
    X = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(Y.astype(np.float32))
    return Env(X=X, y=y, meta=meta or {})


# =============================================================================
# Construction des environnements par niveau de biais (Option A)
# =============================================================================

def compute_identity_tox_rates(
    df_train,
    id_cols_present: List[str],
) -> Dict[str, float]:
    """Calcule le taux de toxicité par identité sur le split train."""
    rates = {}
    for col in id_cols_present:
        mask = df_train[col] >= 0.5
        if mask.sum() > 0:
            rates[col] = float(df_train.loc[mask, "Y"].mean())
        else:
            rates[col] = 0.0
    return rates


def build_bias_envs(
    df_train,
    id_cols_present: List[str],
    embeddings: np.ndarray,
    global_indices: np.ndarray,
    bias_threshold: float = 0.20,
    max_per_env: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Env], Dict[str, int], Dict[str, List[str]]]:
    """
    Crée 3 environnements d'entraînement selon la force de corrélation spurieuse.

    Assignment exclusif (priorité high_bias > low_bias > no_identity) :
        high_bias    : exemples ayant ≥ 1 identité avec taux tox ≥ bias_threshold
        low_bias     : tous les autres exemples (faiblement identifiés OU sans identité)

    Paramètres
    ----------
    bias_threshold : seuil du taux de toxicité pour séparer high/low bias.
                     Défaut 0.20 → high = {black, white, LGBTQ, muslim}
                                   low  = {christian, female, male, other_religions}

    Retourne
    --------
    train_envs  : List[Env]  (2 envs dans l'ordre high, low)
    env_sizes   : dict  {nom_env: taille}
    env_members : dict  {nom_env: liste des colonnes d'identité}
    """
    rng = np.random.default_rng(seed)
    pos_map = {int(gi): pos for pos, gi in enumerate(global_indices)}

    tox_rates = compute_identity_tox_rates(df_train, id_cols_present)

    high_cols = [c for c in id_cols_present if tox_rates.get(c, 0) >= bias_threshold]
    low_cols  = [c for c in id_cols_present if tox_rates.get(c, 0) <  bias_threshold]

    env_members = {"high_bias": high_cols, "low_bias": low_cols}

    print(f"  Seuil de biais : {bias_threshold:.0%}")
    print(f"  high_bias  ({len(high_cols)} identités) : "
          + ", ".join(f"{c}={tox_rates[c]:.1%}" for c in high_cols))
    print(f"  low_bias   ({len(low_cols)} identités + pas d'identité) : "
          + ", ".join(f"{c}={tox_rates[c]:.1%}" for c in low_cols))

    # Masque high_bias : au moins une identité à fort biais
    if high_cols:
        has_high = df_train[high_cols].max(axis=1) >= 0.5
    else:
        has_high = df_train.index.map(lambda _: False)

    # Masque low_bias : tout le reste (faiblement identifié OU sans identité)
    has_low = ~has_high

    train_envs = []
    env_sizes: Dict[str, int] = {}

    for name, mask in [("high_bias", has_high), ("low_bias", has_low)]:
        df_idx = df_train.index[mask].to_numpy()
        valid = np.array([i for i in df_idx if i in pos_map])

        if len(valid) == 0:
            print(f"  [SKIP] env '{name}' vide")
            continue

        if max_per_env is not None and len(valid) > max_per_env:
            valid = rng.choice(valid, size=max_per_env, replace=False)

        pos   = np.array([pos_map[int(i)] for i in valid])
        emb   = embeddings[pos]
        Y_vals = df_train.loc[valid, "Y"].to_numpy(dtype=np.float32)

        env_sizes[name] = len(valid)
        train_envs.append(_make_env(emb, Y_vals,
                                    meta={"name": name,
                                          "tox_rate": float(Y_vals.mean()),
                                          "n": len(Y_vals)}))
        print(f"  {name:22s}: {len(valid):7,} exemples  | tox={Y_vals.mean():.1%}")

    return train_envs, env_sizes, env_members


# =============================================================================
# Validation (identique à identity_envs)
# =============================================================================

def build_val_env_combined(
    df_val,
    id_cols_present: List[str],
    embeddings: np.ndarray,
    global_indices: np.ndarray,
) -> Env:
    pos_map = {int(gi): pos for pos, gi in enumerate(global_indices)}
    has_id  = df_val[id_cols_present].max(axis=1) >= 0.5
    df_idx  = df_val[has_id].index.to_numpy()
    valid   = np.array([i for i in df_idx if i in pos_map])
    pos     = np.array([pos_map[int(i)] for i in valid])
    emb     = embeddings[pos]
    Y_vals  = df_val.loc[valid, "Y"].to_numpy(dtype=np.float32)
    print(f"  Val (WILDS val split) : {len(valid):,} exemples  | tox={Y_vals.mean():.1%}")
    return _make_env(emb, Y_vals, meta={"name": "val_wilds", "n": len(valid)})


# =============================================================================
# 16 groupes d'évaluation (identique à identity_envs)
# =============================================================================

def build_16_groups(
    df_test,
    embeddings: np.ndarray,
    global_indices: np.ndarray,
    group_specs: List[Tuple[str, int]],
    max_per_group: Optional[int] = None,
    seed: int = 42,
) -> List[Optional[Env]]:
    rng     = np.random.default_rng(seed)
    pos_map = {int(gi): pos for pos, gi in enumerate(global_indices)}
    envs: List[Optional[Env]] = []

    for identity_col, y_label in group_specs:
        if identity_col not in df_test.columns:
            envs.append(None)
            continue

        mask  = (df_test[identity_col] >= 0.5) & (df_test["Y"] == y_label)
        df_idx = df_test.index[mask].to_numpy()
        valid  = np.array([i for i in df_idx if i in pos_map])

        if len(valid) == 0:
            envs.append(None)
            continue

        if max_per_group is not None and len(valid) > max_per_group:
            valid = rng.choice(valid, size=max_per_group, replace=False)

        pos    = np.array([pos_map[int(i)] for i in valid])
        emb    = embeddings[pos]
        Y_vals = df_test.loc[valid, "Y"].to_numpy(dtype=np.float32)
        envs.append(_make_env(emb, Y_vals, meta={
            "identity": identity_col, "y_label": y_label, "n": len(Y_vals),
        }))

    return envs


# =============================================================================
# Évaluation sur les 16 groupes
# =============================================================================

def evaluate_16_groups(
    model: nn.Module,
    group_envs: List[Optional[Env]],
    group_names: List[str],
    group_specs: List[Tuple[str, int]],
    device: str = "cpu",
) -> List[Dict]:
    model.eval()
    results = []

    for i, (env, name, (identity_col, y_label)) in enumerate(
        zip(group_envs, group_names, group_specs)
    ):
        if env is None or len(env.y) == 0:
            results.append({
                "group": i + 1, "name": name, "n": 0,
                "accuracy": float("nan"), "fpr": float("nan"), "fnr": float("nan"),
            })
            continue

        with torch.no_grad():
            logits = model(env.X.to(device)).squeeze()
        y_pred = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
        y_true = env.y.cpu().numpy()

        acc = float((y_pred == y_true).mean())

        if y_label == 0:
            fpr = float((y_pred == 1).mean())
            fnr = float("nan")
        else:
            fpr = float("nan")
            fnr = float((y_pred == 0).mean())

        results.append({
            "group": i + 1, "name": name, "n": len(y_true),
            "accuracy": acc, "fpr": fpr, "fnr": fnr,
        })

    return results


# =============================================================================
# Visualisation
# =============================================================================

def plot_training_curves(hist_erm: dict, hist_irm: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(hist_erm["step"], hist_erm["loss"], label="ERM")
    ax.plot(hist_irm["step"], hist_irm["loss"], label="IRM")
    ax.set_title("Training Loss"); ax.set_xlabel("Step")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hist_erm["step"], hist_erm["train_acc"], label="ERM train")
    ax.plot(hist_irm["step"], hist_irm["train_acc"], label="IRM train")
    ax.plot(hist_erm["step"], hist_erm["val_acc"],   ls="--", label="ERM val")
    ax.plot(hist_irm["step"], hist_irm["val_acc"],   ls="--", label="IRM val")
    ax.set_title("Accuracy (Train & Val)"); ax.set_xlabel("Step")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(hist_erm["step"], hist_erm["test_acc"], label="ERM test (16 groupes)")
    ax.plot(hist_irm["step"], hist_irm["test_acc"], label="IRM test (16 groupes)")
    ax.set_title("Accuracy (Test global)"); ax.set_xlabel("Step")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Courbes → {path}")


def plot_accuracy_16_groups(
    results_erm: List[Dict],
    results_irm: List[Dict],
    out_dir: str,
):
    names = [r["name"] for r in results_erm]
    erm_v = [r["accuracy"] for r in results_erm]
    irm_v = [r["accuracy"] for r in results_irm]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width / 2, erm_v, width, label="ERM",
           color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, irm_v, width, label="IRM",
           color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy par groupe — ERM vs IRM\n(CivilComments, envs par niveau de biais)")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="50 %")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i in range(len(names)):
        ax.text(i, -0.09, str(i + 1), ha="center", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.text(-0.5, -0.12, "Groupe #", ha="left", fontsize=7,
            transform=ax.get_xaxis_transform())

    for k in range(len(IDENTITY_COLUMNS)):
        if k % 2 == 0:
            ax.axvspan(k * 2 - 0.5, k * 2 + 1.5, alpha=0.04, color="blue")

    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_16_groups.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Accuracy 16 groupes → {path}")


def plot_fpr_fnr_16_groups(
    results_erm: List[Dict],
    results_irm: List[Dict],
    out_dir: str,
):
    civil_names = [r["name"] for r in results_erm if not np.isnan(r["fpr"])]
    civil_erm   = [r["fpr"]  for r in results_erm if not np.isnan(r["fpr"])]
    civil_irm   = [r["fpr"]  for r in results_irm if not np.isnan(r["fpr"])]

    toxic_names = [r["name"] for r in results_erm if not np.isnan(r["fnr"])]
    toxic_erm   = [r["fnr"]  for r in results_erm if not np.isnan(r["fnr"])]
    toxic_irm   = [r["fnr"]  for r in results_irm if not np.isnan(r["fnr"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    width = 0.38

    for ax, names, erm_v, irm_v, ylabel, title in [
        (ax1, civil_names, civil_erm, civil_irm,
         "FPR", "Faux Positifs — Civils classés toxiques (Y=0)"),
        (ax2, toxic_names, toxic_erm, toxic_irm,
         "FNR", "Faux Négatifs — Toxiques manqués (Y=1)"),
    ]:
        if not names:
            ax.set_visible(False)
            continue
        x = np.arange(len(names))
        ax.bar(x - width / 2, erm_v, width, label="ERM",
               color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, irm_v, width, label="IRM",
               color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        top = max(max(erm_v, default=0), max(irm_v, default=0))
        ax.set_ylim(0, top * 1.35 + 0.02)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for xi, (e, r) in enumerate(zip(erm_v, irm_v)):
            ax.text(xi - width / 2, e + 0.005, f"{e:.2f}", ha="center", fontsize=6.5)
            ax.text(xi + width / 2, r + 0.005, f"{r:.2f}", ha="center", fontsize=6.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "fpr_fnr_16_groups.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  FPR/FNR 16 groupes → {path}")


def plot_scatter_erm_vs_irm(
    results_erm: List[Dict],
    results_irm: List[Dict],
    out_dir: str,
):
    erm_v, irm_v, names, is_toxic = [], [], [], []
    for re, ri in zip(results_erm, results_irm):
        if np.isnan(re["accuracy"]) or np.isnan(ri["accuracy"]):
            continue
        erm_v.append(re["accuracy"])
        irm_v.append(ri["accuracy"])
        names.append(re["name"])
        is_toxic.append("Toxique" in re["name"])

    colors = ["#e74c3c" if t else "#3498db" for t in is_toxic]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(erm_v, irm_v, c=colors, s=90, zorder=3,
               edgecolors="black", linewidth=0.5)

    lo = min(min(erm_v), min(irm_v)) - 0.04
    hi = max(max(erm_v), max(irm_v)) + 0.04
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Accuracy ERM", fontsize=11)
    ax.set_ylabel("Accuracy IRM", fontsize=11)
    ax.set_title("ERM vs IRM par groupe\n(au-dessus de la diag. : IRM améliore)")
    ax.grid(alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(color="#3498db", label="Civil (Y=0)"),
        mpatches.Patch(color="#e74c3c", label="Toxique (Y=1)"),
    ])
    for name, xe, xi in zip(names, erm_v, irm_v):
        if abs(xi - xe) > 0.05:
            ax.annotate(name, (xe, xi), fontsize=6, ha="left",
                        xytext=(4, 3), textcoords="offset points")

    plt.tight_layout()
    path = os.path.join(out_dir, "scatter_erm_vs_irm.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Scatter → {path}")


def plot_disparity_summary(
    results_erm: List[Dict],
    results_irm: List[Dict],
    out_dir: str,
):
    data = [(re["name"], re["accuracy"], ri["accuracy"])
            for re, ri in zip(results_erm, results_irm)
            if not np.isnan(re["accuracy"])]
    data.sort(key=lambda x: x[1])

    names = [d[0] for d in data]
    erm_v = [d[1] for d in data]
    irm_v = [d[2] for d in data]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(y - 0.2, erm_v, 0.38, label="ERM",
            color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.barh(y + 0.2, irm_v, 0.38, label="IRM",
            color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Disparité inter-groupes — ERM vs IRM\n(trié par Acc ERM croissante)")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0, 1.1)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "disparity_horizontal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Disparité → {path}")


def plot_env_bias_overview(
    env_members: Dict[str, List[str]],
    tox_rates: Dict[str, float],
    bias_threshold: float,
    out_dir: str,
):
    """Visualise la classification des identités en high/low bias."""
    all_cols  = [c for c in IDENTITY_COLUMNS if c in tox_rates]
    all_rates = [tox_rates[c] for c in all_cols]
    colors    = ["#e74c3c" if r >= bias_threshold else "#3498db" for r in all_rates]
    labels    = [IDENTITY_LABELS.get(c, c) for c in all_cols]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(all_cols)), all_rates, color=colors, edgecolor="black",
                  linewidth=0.6, alpha=0.85)
    ax.axhline(bias_threshold, color="black", ls="--", lw=1.2,
               label=f"Seuil {bias_threshold:.0%}")
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Taux de toxicité")
    ax.set_title("Classification des identités par niveau de biais\n"
                 "Rouge = high_bias (env 1)  |  Bleu = low_bias (env 2, inclut ∅ identité)")
    ax.set_ylim(0, max(all_rates) * 1.35)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (r, bar) in enumerate(zip(all_rates, bars)):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.005,
                f"{r:.1%}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "env_bias_overview.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Biais par identité → {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CivilComments – ERM vs IRM, environnements = niveaux de biais"
    )
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--device",         type=str,   default="auto")
    parser.add_argument("--root_dir",       type=str,   default=".")
    parser.add_argument("--max_per_env",    type=int,   default=0,
                        help="Cap par environnement (0 = pas de cap)")
    parser.add_argument("--max_per_group",  type=int,   default=0,
                        help="Cap par groupe d'évaluation (0 = pas de cap)")
    parser.add_argument("--bias_threshold", type=float, default=0.20,
                        help="Seuil taux de tox pour séparer high/low bias (défaut 0.20)")
    parser.add_argument("--bert_model",     type=str,   default="distilbert-base-uncased")
    parser.add_argument("--max_length",     type=int,   default=128)
    parser.add_argument("--embed_batch",    type=int,   default=64)

    # Entraînement
    parser.add_argument("--erm_steps",    type=int,   default=30_000)
    parser.add_argument("--erm_lr",       type=float, default=1e-4)
    parser.add_argument("--irm_steps",    type=int,   default=30_000)
    parser.add_argument("--irm_lr",       type=float, default=1e-3)
    parser.add_argument("--irm_lambda",   type=float, default=5000.0)
    parser.add_argument("--batch",        type=int,   default=256)
    parser.add_argument("--eval_every",   type=int,   default=200)

    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "plots" / "bias_envs"))

    args = parser.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 1 : Chargement du CSV
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement du dataset CivilComments")
    print("=" * 70)

    df = load_dataframe(root_dir=args.root_dir)
    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()
    df_test  = df[df["split"] == "test"].copy()

    id_cols_present = [c for c in IDENTITY_COLUMNS if c in df.columns]
    print(f"  Train : {len(df_train):,} exemples")
    print(f"  Val   : {len(df_val):,} exemples   ← split WILDS officiel")
    print(f"  Test  : {len(df_test):,} exemples")
    print(f"  Identités disponibles ({len(id_cols_present)}) : {id_cols_present}")

    tox_rates = compute_identity_tox_rates(df_train, id_cols_present)
    for col in id_cols_present:
        n_id = int((df_train[col] >= 0.5).sum())
        print(f"    {col:22s}: {n_id:7,} exemples  tox={tox_rates[col]:.1%}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2 : Embeddings DistilBERT
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Embeddings DistilBERT")
    print("=" * 70)

    max_per_env_arg   = args.max_per_env   if args.max_per_env   > 0 else None
    max_per_group_arg = args.max_per_group if args.max_per_group > 0 else None

    # Tous les exemples du train (identifiés ou non)
    train_indices = df_train.index.to_numpy()
    print(f"  Encodage train : {len(train_indices):,} textes …")
    emb_train = embed_texts(
        df_train.loc[train_indices, "comment_text"].tolist(),
        model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch)

    val_has_id  = df_val[id_cols_present].max(axis=1) >= 0.5
    val_indices = df_val[val_has_id].index.to_numpy()
    print(f"  Encodage val   : {len(val_indices):,} textes …")
    emb_val = embed_texts(
        df_val.loc[val_indices, "comment_text"].tolist(),
        model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch)

    test_has_id  = df_test[id_cols_present].max(axis=1) >= 0.5
    test_indices = df_test[test_has_id].index.to_numpy()
    print(f"  Encodage test  : {len(test_indices):,} textes …")
    emb_test = embed_texts(
        df_test.loc[test_indices, "comment_text"].tolist(),
        model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Environnements par niveau de biais
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Environnements d'entraînement (par niveau de biais)")
    print("=" * 70)

    train_envs, env_sizes, env_members = build_bias_envs(
        df_train=df_train,
        id_cols_present=id_cols_present,
        embeddings=emb_train,
        global_indices=train_indices,
        bias_threshold=args.bias_threshold,
        max_per_env=max_per_env_arg,
        seed=args.seed,
    )

    print(f"\n  Total envs d'entraînement : {len(train_envs)}")
    print("\n  Construction du set de validation (split WILDS val) :")
    env_val = build_val_env_combined(
        df_val=df_val,
        id_cols_present=id_cols_present,
        embeddings=emb_val,
        global_indices=val_indices,
    )
    val_envs_for_train = [env_val]

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4 : 16 groupes d'évaluation (split test)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Groupes d'évaluation (16 groupes, split test)")
    print("=" * 70)

    group_envs = build_16_groups(
        df_test=df_test,
        embeddings=emb_test,
        global_indices=test_indices,
        group_specs=GROUP_SPECS,
        max_per_group=max_per_group_arg,
        seed=args.seed,
    )

    for i, (genv, gname) in enumerate(zip(group_envs, GROUP_NAMES)):
        n = len(genv.y) if genv is not None else 0
        print(f"  Groupe {i+1:2d} : {gname:<30} {n:>5} exemples"
              + ("  [VIDE]" if genv is None else ""))

    valid_test_envs = [g for g in group_envs if g is not None]
    env_test_global = Env(
        X=torch.cat([g.X for g in valid_test_envs]),
        y=torch.cat([g.y for g in valid_test_envs]),
        meta={"name": "all_16_groups"},
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5a : Entraînement ERM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 5a : Entraînement ERM")
    print("=" * 70)

    model_erm, hist_erm = train_erm(
        envs=train_envs,
        steps=args.erm_steps,
        lr=args.erm_lr,
        batch=args.batch,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        val_envs=val_envs_for_train,
        test_env=env_test_global,
        dataset_name="civilcomments",
        n_classes=2,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5b : Entraînement IRM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 5b : Entraînement IRM")
    print("=" * 70)

    model_irm, hist_irm = train_irm(
        envs=train_envs,
        steps=args.irm_steps,
        lr=args.irm_lr,
        batch=args.batch,
        irm_lambda=args.irm_lambda,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        val_envs=val_envs_for_train,
        test_env=env_test_global,
        dataset_name="civilcomments",
        n_classes=2,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 6 : Évaluation sur les 16 groupes
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 6 : Évaluation sur les 16 groupes")
    print("=" * 70)

    results_erm = evaluate_16_groups(
        model_erm, group_envs, GROUP_NAMES, GROUP_SPECS, device=device
    )
    results_irm = evaluate_16_groups(
        model_irm, group_envs, GROUP_NAMES, GROUP_SPECS, device=device
    )

    # ─────────────────────────────────────────────────────────────────────
    # Résumé tableau
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉSULTATS PAR GROUPE")
    print("=" * 70)
    print(f"{'#':>2}  {'Groupe':<28} {'N':>5}  "
          f"{'ERM Acc':>8} {'IRM Acc':>8}  "
          f"{'ERM FPR/FNR':>12} {'IRM FPR/FNR':>12}")
    print("-" * 82)

    for re, ri in zip(results_erm, results_irm):
        e_acc = f"{re['accuracy']:.3f}" if not np.isnan(re["accuracy"]) else "   N/A"
        i_acc = f"{ri['accuracy']:.3f}" if not np.isnan(ri["accuracy"]) else "   N/A"
        if not np.isnan(re["fpr"]):
            e_rate = f"FPR {re['fpr']:.3f}"
            i_rate = f"FPR {ri['fpr']:.3f}"
        elif not np.isnan(re["fnr"]):
            e_rate = f"FNR {re['fnr']:.3f}"
            i_rate = f"FNR {ri['fnr']:.3f}"
        else:
            e_rate = i_rate = "        N/A"
        print(f"{re['group']:>2}  {re['name']:<28} {re['n']:>5}  "
              f"{e_acc:>8} {i_acc:>8}  {e_rate:>12} {i_rate:>12}")

    accs_erm = [r["accuracy"] for r in results_erm if not np.isnan(r["accuracy"])]
    accs_irm = [r["accuracy"] for r in results_irm if not np.isnan(r["accuracy"])]
    fprs_erm = [r["fpr"] for r in results_erm if not np.isnan(r["fpr"])]
    fprs_irm = [r["fpr"] for r in results_irm if not np.isnan(r["fpr"])]

    print("\n" + "-" * 82)
    print(f"  {'':30} {'ERM':>15} {'IRM':>15}")
    print(f"  {'Accuracy moyenne':30} {np.mean(accs_erm):>15.3f} {np.mean(accs_irm):>15.3f}")
    print(f"  {'Accuracy minimale (worst group)':30} {np.min(accs_erm):>15.3f} {np.min(accs_irm):>15.3f}")
    print(f"  {'Écart-type inter-groupes':30} {np.std(accs_erm):>15.3f} {np.std(accs_irm):>15.3f}")
    print(f"  {'FPR moyen (groupes civils)':30} {np.mean(fprs_erm):>15.3f} {np.mean(fprs_irm):>15.3f}")
    print(f"  {'FPR maximal (civils)':30} {np.max(fprs_erm):>15.3f} {np.max(fprs_irm):>15.3f}")
    print("\n  Écart-type bas  → performances équilibrées entre groupes")
    print("  Acc min élevée  → pas de groupe sacrifié (fairness)")

    # ─────────────────────────────────────────────────────────────────────
    # Sauvegarde
    # ─────────────────────────────────────────────────────────────────────
    output = {
        "bias_threshold": args.bias_threshold,
        "env_members": env_members,
        "tox_rates": tox_rates,
        "erm": results_erm,
        "irm": results_irm,
        "summary": {
            "erm": {
                "mean_acc": float(np.mean(accs_erm)),
                "min_acc":  float(np.min(accs_erm)),
                "std_acc":  float(np.std(accs_erm)),
                "mean_fpr": float(np.mean(fprs_erm)),
                "max_fpr":  float(np.max(fprs_erm)),
            },
            "irm": {
                "mean_acc": float(np.mean(accs_irm)),
                "min_acc":  float(np.min(accs_irm)),
                "std_acc":  float(np.std(accs_irm)),
                "mean_fpr": float(np.mean(fprs_irm)),
                "max_fpr":  float(np.max(fprs_irm)),
            },
        },
    }

    json_path = os.path.join(args.out_dir, "results_16_groups.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Résultats JSON → {json_path}")

    plot_training_curves(hist_erm, hist_irm, args.out_dir)
    plot_env_bias_overview(env_members, tox_rates, args.bias_threshold, args.out_dir)
    plot_accuracy_16_groups(results_erm, results_irm, args.out_dir)
    plot_fpr_fnr_16_groups(results_erm, results_irm, args.out_dir)
    plot_scatter_erm_vs_irm(results_erm, results_irm, args.out_dir)
    plot_disparity_summary(results_erm, results_irm, args.out_dir)

    print("\nTerminé !")


if __name__ == "__main__":
    main()
