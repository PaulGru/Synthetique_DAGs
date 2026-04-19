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
run_civilcomments_identity_envs.py
===================================
Compare ERM vs IRM (IRMv1) sur CivilComments en utilisant les 8 identités
démographiques comme environnements d'entraînement distincts.

Intuition IRM : la corrélation intra-groupe entre « mentionner une identité »
et la toxicité varie d'une identité à l'autre. En traitant chaque identité
comme un environnement séparé, IRM est forcé de trouver un prédicteur de
toxicité qui fonctionne uniformément sur tous les groupes — sans exploiter
aucune association spurieuse identité→toxicité.

Environnements d'entraînement (9 : 8 identités + "sans identité") :
    Env_k = { x ∈ train | colonne identité_k(x) ≥ 0.5 }
    Env_9 = { x ∈ train | aucune colonne identité(x) ≥ 0.5 }

Évaluation sur 16 groupes (8 identités × 2 labels) :
    Groupe 2k-1 : identité_k + Civil   (Y=0)
    Groupe 2k   : identité_k + Toxique (Y=1)
    k = 1..8  →  Homme, Femme, Chrétien, Musulman,
                 Autres religions, Noir, Blanc, LGBTQ+

Métriques par groupe :
    Accuracy | FPR (civils: fraction classés toxiques)
             | FNR (toxiques: fraction manquée)
    + Acc minimale et écart-type inter-groupes (disparité)

Usage :
    uv run run_civilcomments_identity_envs.py --device auto
    uv run run_civilcomments_identity_envs.py --device auto --max_per_env 2000
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
# Configuration des 16 groupes
# =============================================================================

# Ordre : LGBTQ en dernier → groupes 15 et 16
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

# Liste des 16 groupes dans l'ordre numéroté
GROUP_NAMES: List[str] = []
GROUP_SPECS: List[Tuple[str, int]] = []  # (identity_col, y_label)
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
    """Télécharge le CSV CivilComments si absent (SSL désactivé — cert codalab expiré)."""
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
    """Charge le CSV, ajoute la colonne binaire Y (toxicity ≥ 0.5)."""
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
# Construction des environnements d'entraînement (un par identité)
# =============================================================================

def build_identity_envs(
    df_train,
    id_cols_present: List[str],
    embeddings: np.ndarray,
    global_indices: np.ndarray,
    max_per_env: Optional[int] = None,
    include_no_identity: bool = False,
    seed: int = 42,
) -> Tuple[List[Env], Dict[str, int]]:
    """
    Crée un environnement d'entraînement par identité à partir du split train.

    Pour l'identité k :
        Env_k = { x | identité_k(x) ≥ 0.5 }

    La validation est gérée séparément depuis le split WILDS "val".
    Voir build_val_env_combined().

    Paramètres
    ----------
    df_train       : DataFrame du split train
    id_cols_present: colonnes d'identité présentes dans df_train
    embeddings     : (M, 768) pour les M exemples de global_indices
    global_indices : indices CSV correspondant aux M embeddings
    max_per_env    : cap par env (sous-échantillonnage aléatoire)
    include_no_identity : ajouter un env « sans identité »
    seed           : graine aléatoire

    Retourne
    --------
    train_envs : List[Env]
    env_sizes  : dict  {nom_env: taille}
    """
    rng = np.random.default_rng(seed)
    pos_map = {int(gi): pos for pos, gi in enumerate(global_indices)}

    train_envs = []
    env_sizes: Dict[str, int] = {}

    cols_to_build = list(id_cols_present)
    if include_no_identity:
        cols_to_build.append("_no_identity_")

    for col in cols_to_build:
        if col == "_no_identity_":
            available_id_cols = [c for c in IDENTITY_COLUMNS if c in df_train.columns]
            mask = df_train[available_id_cols].max(axis=1) < 0.5
            name = "no_identity"
        else:
            if col not in df_train.columns:
                print(f"  [SKIP] colonne '{col}' absente du CSV")
                continue
            mask = df_train[col] >= 0.5
            name = col

        df_idx = df_train.index[mask].to_numpy()
        valid = np.array([i for i in df_idx if i in pos_map])

        if len(valid) == 0:
            print(f"  [SKIP] env '{name}' vide après filtrage")
            continue

        if max_per_env is not None and len(valid) > max_per_env:
            valid = rng.choice(valid, size=max_per_env, replace=False)

        env_sizes[name] = len(valid)
        pos = np.array([pos_map[int(i)] for i in valid])
        emb = embeddings[pos]
        Y_vals = df_train.loc[valid, "Y"].to_numpy(dtype=np.float32)
        train_envs.append(_make_env(emb, Y_vals,
                                    meta={"name": name, "tox_rate": float(Y_vals.mean()),
                                          "n": len(Y_vals)}))

        print(f"  {name:22s}: {len(valid):5,} exemples  | tox={Y_vals.mean():.1%}")

    return train_envs, env_sizes


def build_val_env_combined(
    df_val,
    id_cols_present: List[str],
    embeddings: np.ndarray,
    global_indices: np.ndarray,
) -> Env:
    """
    Construit un unique Env de validation à partir du split WILDS "val".
    Combine tous les exemples ayant au moins une identité.
    """
    pos_map = {int(gi): pos for pos, gi in enumerate(global_indices)}

    has_id = df_val[id_cols_present].max(axis=1) >= 0.5
    df_idx = df_val[has_id].index.to_numpy()
    valid = np.array([i for i in df_idx if i in pos_map])

    pos = np.array([pos_map[int(i)] for i in valid])
    emb = embeddings[pos]
    Y_vals = df_val.loc[valid, "Y"].to_numpy(dtype=np.float32)
    print(f"  Val (WILDS val split) : {len(valid):,} exemples  | tox={Y_vals.mean():.1%}")
    return _make_env(emb, Y_vals, meta={"name": "val_wilds", "n": len(valid)})


# =============================================================================
# Construction des 16 groupes d'évaluation (split test)
# =============================================================================

def build_16_groups(
    df_test,
    embeddings: np.ndarray,
    global_indices: np.ndarray,
    group_specs: List[Tuple[str, int]],
    max_per_group: Optional[int] = None,
    seed: int = 42,
) -> List[Optional[Env]]:
    """
    Construit les 16 Env d'évaluation depuis le split test.

    Groupe (identity_col, y_label) = exemples du test où
        identity_col ≥ 0.5   ET   Y == y_label.

    Un exemple peut apparaître dans plusieurs groupes (multi-identité).
    Retourne une liste de 16 Env (ou None si le groupe est vide).
    """
    rng = np.random.default_rng(seed)
    pos_map = {int(gi): pos for pos, gi in enumerate(global_indices)}

    envs: List[Optional[Env]] = []
    for identity_col, y_label in group_specs:
        if identity_col not in df_test.columns:
            envs.append(None)
            continue

        mask = (df_test[identity_col] >= 0.5) & (df_test["Y"] == y_label)
        df_idx = df_test.index[mask].to_numpy()
        valid = np.array([i for i in df_idx if i in pos_map])

        if len(valid) == 0:
            envs.append(None)
            continue

        if max_per_group is not None and len(valid) > max_per_group:
            valid = rng.choice(valid, size=max_per_group, replace=False)

        pos = np.array([pos_map[int(i)] for i in valid])
        emb = embeddings[pos]
        Y_vals = df_test.loc[valid, "Y"].to_numpy(dtype=np.float32)
        envs.append(_make_env(emb, Y_vals, meta={
            "identity": identity_col,
            "y_label": y_label,
            "n": len(Y_vals),
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
    """
    Calcule pour chaque groupe :
        accuracy, FPR (civils Y=0), FNR (toxiques Y=1).

    FPR = P(ŷ=1 | y=0) : fraction de civils incorrectement signalés
    FNR = P(ŷ=0 | y=1) : fraction de toxiques manqués (= 1 − recall)
    """
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
            # Groupe Civil : FPR = fraction classée toxique
            fpr = float((y_pred == 1).mean())
            fnr = float("nan")
        else:
            # Groupe Toxique : FNR = fraction manquée
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
    """Loss + accuracy (train/val/test) au fil des steps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(hist_erm["step"], hist_erm["loss"], label="ERM")
    ax.plot(hist_irm["step"], hist_irm["loss"], label="IRM")
    ax.set_title("Training Loss"); ax.set_xlabel("Step")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(hist_erm["step"], hist_erm["train_acc"], label="ERM train")
    ax.plot(hist_irm["step"], hist_irm["train_acc"], label="IRM train")
    ax.plot(hist_erm["step"], hist_erm["val_acc"], ls="--", label="ERM val InD")
    ax.plot(hist_irm["step"], hist_irm["val_acc"], ls="--", label="IRM val InD")
    ax.set_title("Accuracy (Train & Val InD)"); ax.set_xlabel("Step")
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
    """Grouped bar chart : accuracy ERM vs IRM sur les 16 groupes."""
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
    ax.set_title("Accuracy par groupe — ERM vs IRM\n(CivilComments, 16 groupes)")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="50 %")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Numéros de groupe sous les barres
    for i in range(len(names)):
        ax.text(i, -0.09, str(i + 1), ha="center", fontsize=7,
                transform=ax.get_xaxis_transform())
    ax.text(-0.5, -0.12, "Groupe #", ha="left", fontsize=7,
            transform=ax.get_xaxis_transform())

    # Alternance de fond par identité (blocs de 2)
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
    """
    Deux panneaux côte à côte :
    Gauche  → FPR pour les 8 groupes Civils   (Y=0)
    Droite  → FNR pour les 8 groupes Toxiques (Y=1)
    """
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
            ax.text(xi - width / 2, e + 0.005, f"{e:.2f}",
                    ha="center", fontsize=6.5)
            ax.text(xi + width / 2, r + 0.005, f"{r:.2f}",
                    ha="center", fontsize=6.5)

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
    """
    Scatter plot : accuracy ERM (x) vs IRM (y) par groupe.
    Points au-dessus de la diagonale y=x → IRM améliore ce groupe.
    Points civils en bleu, toxiques en rouge.
    """
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
    ax.set_title("ERM vs IRM par groupe\n(au-dessus de la diagonale : IRM améliore)")
    ax.grid(alpha=0.3)

    ax.legend(handles=[
        mpatches.Patch(color="#3498db", label="Civil (Y=0)"),
        mpatches.Patch(color="#e74c3c", label="Toxique (Y=1)"),
    ])

    # Annoter les groupes avec un écart significatif
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
    """
    Barres horizontales : accuracy par groupe, triées.
    Montre la disparité inter-groupes ERM vs IRM.
    """
    data = [(re["name"], re["accuracy"], ri["accuracy"])
            for re, ri in zip(results_erm, results_irm)
            if not np.isnan(re["accuracy"])]
    # Tri par accuracy ERM croissante
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CivilComments – ERM vs IRM, environnements = identités"
    )
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--root_dir",     type=str,   default=".",
                        help="Répertoire contenant all_data_with_identities.csv")
    parser.add_argument("--max_per_env",  type=int,   default=0,
                        help="Taille max par environnement d'entraînement (0 = pas de cap, utilise tout le train)")
    parser.add_argument("--max_per_group", type=int,  default=0,
                        help="Taille max par groupe d'évaluation (0 = pas de cap, utilise tout le test)")
    parser.add_argument("--bert_model",   type=str,   default="distilbert-base-uncased")
    parser.add_argument("--max_length",   type=int,   default=128)
    parser.add_argument("--embed_batch",  type=int,   default=64)

    # Entraînement
    parser.add_argument("--erm_steps",    type=int,   default=30_000)
    parser.add_argument("--erm_lr",       type=float, default=1e-4)
    parser.add_argument("--irm_steps",    type=int,   default=30_000)
    parser.add_argument("--irm_lr",       type=float, default=1e-3)
    parser.add_argument("--irm_lambda",   type=float, default=5000.0)
    parser.add_argument("--batch",        type=int,   default=256)
    parser.add_argument("--eval_every",   type=int,   default=200)

    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "plots" / "identity_envs"))

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
    df_val   = df[df["split"] == "val"].copy()    # split de validation WILDS officiel
    df_test  = df[df["split"] == "test"].copy()

    id_cols_present = [c for c in IDENTITY_COLUMNS if c in df.columns]
    print(f"  Train : {len(df_train):,} exemples")
    print(f"  Val   : {len(df_val):,} exemples   ← split WILDS officiel")
    print(f"  Test  : {len(df_test):,} exemples")
    print(f"  Identités disponibles ({len(id_cols_present)}) : {id_cols_present}")

    for col in id_cols_present:
        n_id   = int((df_train[col] >= 0.5).sum())
        tox_id = float(df_train.loc[df_train[col] >= 0.5, "Y"].mean())
        print(f"    {col:22s}: {n_id:7,} exemples  tox={tox_id:.1%}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2 : Embeddings DistilBERT
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Embeddings DistilBERT")
    print("=" * 70)

    rng0 = np.random.default_rng(args.seed)
    max_per_env_arg   = args.max_per_env   if args.max_per_env   > 0 else None
    max_per_group_arg = args.max_per_group if args.max_per_group > 0 else None

    # --- Train : exemples avec au moins une identité + ceux sans aucune identité ---
    train_has_id = df_train[id_cols_present].max(axis=1) >= 0.5
    train_id_idx = df_train[train_has_id].index.to_numpy()
    no_id_idx    = df_train[~train_has_id].index.to_numpy()
    if max_per_env_arg is not None:
        no_id_idx = rng0.choice(no_id_idx, size=min(max_per_env_arg, len(no_id_idx)), replace=False)
    train_indices = np.unique(np.concatenate([train_id_idx, no_id_idx]))
    print(f"  Dont {len(train_id_idx):,} avec identité + {len(no_id_idx):,} sans identité")

    print(f"  Encodage train : {len(train_indices):,} textes …")
    emb_train = embed_texts(
        df_train.loc[train_indices, "comment_text"].tolist(),
        model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch)

    # --- Val : split WILDS officiel, exemples avec au moins une identité ---
    val_has_id = df_val[id_cols_present].max(axis=1) >= 0.5
    val_indices = df_val[val_has_id].index.to_numpy()
    print(f"  Encodage val   : {len(val_indices):,} textes …")
    emb_val = embed_texts(
        df_val.loc[val_indices, "comment_text"].tolist(),
        model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch)

    # --- Test : exemples avec au moins une identité ---
    test_has_id = df_test[id_cols_present].max(axis=1) >= 0.5
    test_indices = df_test[test_has_id].index.to_numpy()
    print(f"  Encodage test  : {len(test_indices):,} textes …")
    emb_test = embed_texts(
        df_test.loc[test_indices, "comment_text"].tolist(),
        model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Environnements d'entraînement (un par identité)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Environnements d'entraînement")
    print("=" * 70)

    train_envs, env_sizes = build_identity_envs(
        df_train=df_train,
        id_cols_present=id_cols_present,
        embeddings=emb_train,
        global_indices=train_indices,
        max_per_env=max_per_env_arg,
        include_no_identity=True,
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

    # Env de test global (pour la courbe d'entraînement seulement)
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

    # Métriques agrégées de disparité
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
    # Sauvegarde JSON + plots
    # ─────────────────────────────────────────────────────────────────────
    output = {
        "erm": results_erm,
        "irm": results_irm,
        "summary": {
            "erm": {
                "mean_acc":  float(np.mean(accs_erm)),
                "min_acc":   float(np.min(accs_erm)),
                "std_acc":   float(np.std(accs_erm)),
                "mean_fpr":  float(np.mean(fprs_erm)),
                "max_fpr":   float(np.max(fprs_erm)),
            },
            "irm": {
                "mean_acc":  float(np.mean(accs_irm)),
                "min_acc":   float(np.min(accs_irm)),
                "std_acc":   float(np.std(accs_irm)),
                "mean_fpr":  float(np.mean(fprs_irm)),
                "max_fpr":   float(np.max(fprs_irm)),
            },
        },
    }

    json_path = os.path.join(args.out_dir, "results_16_groups.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Résultats JSON → {json_path}")

    plot_training_curves(hist_erm, hist_irm, args.out_dir)
    plot_accuracy_16_groups(results_erm, results_irm, args.out_dir)
    plot_fpr_fnr_16_groups(results_erm, results_irm, args.out_dir)
    plot_scatter_erm_vs_irm(results_erm, results_irm, args.out_dir)
    plot_disparity_summary(results_erm, results_irm, args.out_dir)

    print("\nTerminé !")


if __name__ == "__main__":
    main()
