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
run_civilcomments_erm_vs_irm.py
===============================
Compare ERM vs IRM (IRMv1) sur la tâche de classification de toxicité
du dataset CivilComments (version WILDS : shlomihod/civil-comments-wilds).

Objectif : montrer qu'IRM réduit le False Positive Rate sur le groupe
**Civils avec Identité (Y=0, A=1)** — le groupe le plus pénalisé par
un modèle qui utilise la présence d'identités démographiques comme
raccourci fallacieux.

Pipeline :
    1. Chargement + variable d'identité A
    2. Construction de 2 environnements biaisés (E1=80/20, E2=60/40)
    3. Val InD (5 % de chaque env), Test OOD (contre-intuitif), Val OOD
    4. Entraînement ERM et IRM sur embeddings DistilBERT gelé
    5. Évaluation : Accuracy, Loss, FPR(Y=0,A=1)

Usage :
    uv run run_civilcomments_erm_vs_irm.py
    uv run run_civilcomments_erm_vs_irm.py --device auto --irm_lambda 500
"""


import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_synth import Env
from models_training import train_erm, train_irm
from utils_irm import resolve_device

# =============================================================================
# 1. Chargement du dataset et création des variables Y et A
# =============================================================================

# Noms des colonnes d'identité tels qu'exposés par la librairie WILDS
IDENTITY_COLUMNS = [
    "male", "female", "LGBTQ",
    "christian", "muslim", "other_religions",
    "black", "white",
]


_WILDS_URL = (
    "https://worksheets.codalab.org/rest/bundles/"
    "0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/"
)


def _find_csv(root_dir: str) -> str | None:
    """Cherche all_data_with_identities.csv dans root_dir et ses sous-dossiers immédiats."""
    candidates = [
        os.path.join(root_dir, "all_data_with_identities.csv"),
        os.path.join(root_dir, "civilcomments_v1.0", "all_data_with_identities.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _download_civilcomments_no_ssl(root_dir: str) -> str:
    """
    Télécharge et extrait l'archive CivilComments en désactivant la
    vérification SSL (contournement du certificat expiré sur codalab.org).

    Returns : chemin absolu vers all_data_with_identities.csv
    """
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
        block = 1 << 20  # 1 MB
        downloaded = 0
        with open(archive_path, "wb") as fout:
            while True:
                chunk = resp.read(block)
                if not chunk:
                    break
                fout.write(chunk)
                downloaded += len(chunk)
                mb_done = downloaded / 1_048_576
                mb_total_str = f"{total / 1_048_576:.1f}" if total else "?"
                print(f"\r  {mb_done:.1f} / {mb_total_str} MB", end="", flush=True)
    print()

    print("  Extraction de l'archive …")
    with tarfile.open(archive_path) as tf:
        tf.extractall(root_dir, filter="data")
    os.remove(archive_path)

    csv_path = _find_csv(root_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"all_data_with_identities.csv introuvable dans {root_dir} "
            "après extraction. Vérifiez le contenu de l'archive."
        )
    print(f"  Dataset CivilComments prêt : {csv_path}")
    return csv_path


def load_civilcomments(root_dir: str = ".") -> dict:
    """
    Charge le dataset CivilComments depuis le CSV officiel WILDS et
    construit les variables binaires Y (toxicité) et A (identité).

    Télécharge automatiquement le CSV au premier appel.

    Parameters
    ----------
    root_dir : str
        Répertoire où chercher / stocker all_data_with_identities.csv.

    Returns
    -------
    splits : dict
        "train_pool" : train + val concaténés  → pool d'entraînement
        "test"       : split test original      → évaluation OOD
        Chaque namespace expose :
            .comment_text  : List[str]
            .Y             : np.ndarray  int64 (0/1)
            .A             : np.ndarray  int64 (0/1)
    """
    import pandas as pd
    from types import SimpleNamespace

    csv_path = _download_civilcomments_no_ssl(root_dir)

    print(f"Chargement du CSV : {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    # Y : toxicity >= 0.5
    df["Y"] = (df["toxicity"] >= 0.5).astype(int)

    # A : utilise la colonne pré-calculée identity_any si disponible,
    #     sinon recalcule à partir des colonnes individuelles présentes
    id_cols_present = [c for c in IDENTITY_COLUMNS if c in df.columns]
    if "identity_any" in df.columns:
        df["A"] = (df["identity_any"] >= 0.5).astype(int)
    else:
        df["A"] = (df[id_cols_present].max(axis=1) >= 0.5).astype(int)

    # --- Tableau formel des groupes par split ---
    splits_available = sorted(df["split"].unique())
    groups = [(0, 0, "civil  sans identité"),
              (0, 1, "civil  avec identité"),
              (1, 0, "toxic  sans identité"),
              (1, 1, "toxic  avec identité")]

    header = f"  {'Groupe':<26}" + "".join(f"{s:>10}" for s in splits_available) + f"{'TOTAL':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for y, a, label in groups:
        counts = [int(((df[df["split"] == s]["Y"] == y) & (df[df["split"] == s]["A"] == a)).sum())
                  for s in splits_available]
        total = sum(counts)
        row = f"  Y={y},A={a} ({label})" + "".join(f"{c:>10,}" for c in counts) + f"{total:>10,}"
        print(row)
    print("  " + "-" * (len(header) - 2))
    totals = [int((df["split"] == s).sum()) for s in splits_available]
    print(f"  {'TOTAL':<26}" + "".join(f"{t:>10,}" for t in totals) + f"{sum(totals):>10,}")

    # --- Construction des namespaces ---
    result = {}

    # train_pool = train + val (pas de leakage avec test)
    pool_df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    result["train_pool"] = SimpleNamespace(
        comment_text=pool_df["comment_text"].tolist(),
        Y=pool_df["Y"].to_numpy(dtype=np.int64),
        A=pool_df["A"].to_numpy(dtype=np.int64),
    )

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    result["test"] = SimpleNamespace(
        comment_text=test_df["comment_text"].tolist(),
        Y=test_df["Y"].to_numpy(dtype=np.int64),
        A=test_df["A"].to_numpy(dtype=np.int64),
    )

    print(f"\n  Pool d'entraînement (train+val) : {len(result['train_pool'].Y):,} exemples")
    print(f"  Test OOD                         : {len(result['test'].Y):,} exemples")
    return result


# =============================================================================
# 2. Construction des environnements d'entraînement par sous-échantillonnage
# =============================================================================

def _get_group_indices(split, y_val: int, a_val: int) -> np.ndarray:
    """Renvoie les indices du split où Y==y_val et A==a_val."""
    Y = np.asarray(split.Y)
    A = np.asarray(split.A)
    return np.where((Y == y_val) & (A == a_val))[0]


def _subsample_env(
    split,
    p_toxic_given_identity: float,
    p_civil_given_no_identity: float,
    rng: np.random.Generator,
    max_per_group: int | None = None,
) -> np.ndarray:
    """
    Construit un sous-ensemble biaisé en piochant dans les 4 groupes (Y,A).

    Pour A=1 : p_toxic_given_identity    => fraction de Y=1
    Pour A=0 : p_civil_given_no_identity => fraction de Y=0

    Renvoie les indices sélectionnés dans le split.
    """
    idx_y0_a0 = _get_group_indices(split, 0, 0)
    idx_y1_a0 = _get_group_indices(split, 1, 0)
    idx_y0_a1 = _get_group_indices(split, 0, 1)
    idx_y1_a1 = _get_group_indices(split, 1, 1)

    # --- A=1 : p_toxic_given_identity % toxiques, reste civils ---
    n_y1_a1 = len(idx_y1_a1)
    n_y0_a1 = len(idx_y0_a1)
    # Prendre le max permis par le groupe le plus contraint
    # n_toxic_a1 / n_total_a1 = p_toxic_given_identity
    # => n_civil_a1 = n_toxic_a1 * (1-p) / p
    n_toxic_a1 = n_y1_a1
    n_civil_a1 = int(n_toxic_a1 * (1 - p_toxic_given_identity) / p_toxic_given_identity)
    if n_civil_a1 > n_y0_a1:
        n_civil_a1 = n_y0_a1
        n_toxic_a1 = int(n_civil_a1 * p_toxic_given_identity / (1 - p_toxic_given_identity))

    # --- A=0 : p_civil_given_no_identity % civils, reste toxiques ---
    n_y0_a0 = len(idx_y0_a0)
    n_y1_a0 = len(idx_y1_a0)
    n_civil_a0 = n_y0_a0
    n_toxic_a0 = int(n_civil_a0 * (1 - p_civil_given_no_identity) / p_civil_given_no_identity)
    if n_toxic_a0 > n_y1_a0:
        n_toxic_a0 = n_y1_a0
        n_civil_a0 = int(n_toxic_a0 * p_civil_given_no_identity / (1 - p_civil_given_no_identity))

    if max_per_group is not None:
        n_toxic_a1 = min(n_toxic_a1, max_per_group)
        n_civil_a1 = min(n_civil_a1, max_per_group)
        n_civil_a0 = min(n_civil_a0, max_per_group)
        n_toxic_a0 = min(n_toxic_a0, max_per_group)

    sel = np.concatenate([
        rng.choice(idx_y1_a1, size=n_toxic_a1, replace=False),
        rng.choice(idx_y0_a1, size=n_civil_a1, replace=False),
        rng.choice(idx_y0_a0, size=n_civil_a0, replace=False),
        rng.choice(idx_y1_a0, size=n_toxic_a0, replace=False),
    ])
    rng.shuffle(sel)
    return sel


def build_train_envs(
    ds,
    max_per_group: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit les indices des deux environnements d'entraînement.

    E1 – Biais fort  : A=1 → 80 % toxique  / A=0 → 80 % civil
    E2 – Biais modéré: A=1 → 60 % toxique  / A=0 → 60 % civil

    Returns : (indices_e1, indices_e2)  dans ds["train_pool"]
    """
    rng = np.random.default_rng(seed)
    train_split = ds["train_pool"]  # SimpleNamespace

    idx_e1 = _subsample_env(
        train_split,
        p_toxic_given_identity=0.90,
        p_civil_given_no_identity=0.90,
        rng=rng,
        max_per_group=max_per_group,
    )
    idx_e2 = _subsample_env(
        train_split,
        p_toxic_given_identity=0.70,
        p_civil_given_no_identity=0.70,
        rng=rng,
        max_per_group=max_per_group,
    )
    return idx_e1, idx_e2


# =============================================================================
# 3. Embeddings DistilBERT (gelé)
# =============================================================================

def embed_texts(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Calcule les embeddings DistilBERT (mean pooling) pour une liste de textes.
    Le modèle est gelé — pas de fine-tuning.

    Returns : np.ndarray de shape (N, 768).
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        hidden = out.last_hidden_state  # (B, seq, 768)
        mask = attention_mask.unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_emb.append(emb.cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  Embedded {i + len(batch)}/{len(texts)}")

    return np.concatenate(all_emb, axis=0).astype(np.float32)


# =============================================================================
# 4. Construction des sets de validation et de test
# =============================================================================

def _split_val_from_indices(
    indices: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sépare val_frac % des indices pour la validation."""
    n_val = max(1, int(len(indices) * val_frac))
    perm = rng.permutation(len(indices))
    val_idx = indices[perm[:n_val]]
    train_idx = indices[perm[n_val:]]
    return train_idx, val_idx


def build_test_ood(
    ds,
    max_per_group: int = 3000,
    seed: int = 42,
) -> np.ndarray:
    """
    Construit le Test OOD à partir du split test original.

    Test OOD = uniquement les deux groupes contre-intuitifs :
        - Civils avec Identité (Y=0, A=1)
        - Toxiques sans Identité (Y=1, A=0)
    Répartition 50/50.

    Returns : test_ood_indices  dans ds["test"]
    """
    rng = np.random.default_rng(seed)
    test_split = ds["test"]  # SimpleNamespace

    idx_y0_a1 = _get_group_indices(test_split, 0, 1)  # Civil + Identité
    idx_y1_a0 = _get_group_indices(test_split, 1, 0)  # Toxic + Pas d'identité

    # 50/50
    n = min(len(idx_y0_a1), len(idx_y1_a0), max_per_group)
    sel_y0_a1 = rng.choice(idx_y0_a1, size=n, replace=False)
    sel_y1_a0 = rng.choice(idx_y1_a0, size=n, replace=False)

    all_ood = np.concatenate([sel_y0_a1, sel_y1_a0])
    rng.shuffle(all_ood)
    return all_ood


# =============================================================================
# 5. Assemblage des Env à partir d'embeddings
# =============================================================================

def make_env(
    embeddings: np.ndarray,
    labels: np.ndarray,
    identities: np.ndarray | None = None,
    meta: dict | None = None,
) -> Env:
    """Crée un objet Env(X, y) à partir d'arrays NumPy."""
    X = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).float()
    if meta is None:
        meta = {}
    if identities is not None:
        meta["A"] = torch.from_numpy(identities).long()
    return Env(X=X, y=y, meta=meta)


# =============================================================================
# 6. Métriques spécifiques
# =============================================================================

def compute_predictions(model: nn.Module, env: Env, device: str = "cpu") -> np.ndarray:
    """Renvoie les prédictions binaires (0/1) pour un Env."""
    model.eval()
    with torch.no_grad():
        logits = model(env.X.to(device))
    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return (probs >= 0.5).astype(np.float32)


def compute_fpr_group(
    model: nn.Module,
    env: Env,
    y_target: int,
    a_target: int,
    device: str = "cpu",
) -> float:
    """
    False Positive Rate pour le sous-groupe (Y=y_target, A=a_target).
    FPR = #(pred=1 & true=0) / #(true=0)  pour le sous-groupe.
    """
    y_true = env.y.cpu().numpy().reshape(-1)
    A = env.meta["A"].cpu().numpy().reshape(-1)
    y_pred = compute_predictions(model, env, device=device)

    mask = (y_true == y_target) & (A == a_target)
    if mask.sum() == 0:
        return float("nan")

    # FPR : parmi les vrais négatifs du sous-groupe, combien sont prédits positifs
    # On s'intéresse au cas y_target=0 (civils), a_target=1 (avec identité)
    # FP = pred=1 alors que true=0
    fp = ((y_pred == 1) & mask).sum()
    return float(fp / mask.sum())


def compute_fnr_group(
    model: nn.Module,
    env: Env,
    y_target: int,
    a_target: int,
    device: str = "cpu",
) -> float:
    """
    False Negative Rate pour le sous-groupe (Y=y_target, A=a_target).
    FNR = #(pred=0 & true=1) / #(true=1)  pour le sous-groupe.
    """
    y_true = env.y.cpu().numpy().reshape(-1)
    A = env.meta["A"].cpu().numpy().reshape(-1)
    y_pred = compute_predictions(model, env, device=device)

    mask = (y_true == y_target) & (A == a_target)
    if mask.sum() == 0:
        return float("nan")

    fn = ((y_pred == 0) & mask).sum()
    return float(fn / mask.sum())


def full_evaluation(
    model: nn.Module,
    env: Env,
    device: str = "cpu",
    label: str = "",
) -> Dict:
    """Évalue accuracy, loss, FPR(Y=0, A=1) et FNR(Y=1, A=0)."""
    model.eval()
    with torch.no_grad():
        logits = model(env.X.to(device)).squeeze()
    y = env.y.to(device).float()
    loss = nn.BCEWithLogitsLoss()(logits, y).item()

    y_pred = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
    y_np = env.y.cpu().numpy().reshape(-1)
    acc = (y_pred == y_np).mean()

    fpr_y0_a1 = compute_fpr_group(model, env, y_target=0, a_target=1, device=device)
    fnr_y1_a0 = compute_fnr_group(model, env, y_target=1, a_target=0, device=device)

    res = {
        "accuracy": float(acc),
        "loss": float(loss),
        "fpr_civil_identity": float(fpr_y0_a1),
        "fnr_toxic_no_identity": float(fnr_y1_a0),
    }
    if label:
        print(
            f"  [{label}] Acc={acc:.4f}  Loss={loss:.4f}  "
            f"FPR(Y=0,A=1)={fpr_y0_a1:.4f}  FNR(Y=1,A=0)={fnr_y1_a0:.4f}"
        )
    return res


# =============================================================================
# 7. Visualisation
# =============================================================================

def _ema_smooth(values: list, alpha: float = 0.05) -> list:
    """Lissage par moyenne exponentielle mobile (EMA).

    alpha proche de 0 → très lisse ; alpha proche de 1 → quasi-brut.
    """
    if not values:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_training_curves(
    hist_erm: dict,
    hist_irm: dict,
    out_dir: str,
    smooth_alpha: float = 0.05,
):
    """Trace les courbes d'entraînement ERM vs IRM.

    La loss est affichée en deux couches :
      - courbe brute (mini-batch) en transparence pour visualiser la variance
      - courbe EMA lissée en premier plan pour la tendance
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # --- Loss ---
    ax = axes[0]
    for hist, color, name in [
        (hist_erm, "C0", "ERM"),
        (hist_irm, "C1", "IRM"),
    ]:
        steps = hist["step"]
        raw   = hist["loss"]
        smooth = _ema_smooth(raw, alpha=smooth_alpha)
        ax.plot(steps, raw,    color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smooth, color=color, alpha=0.9,  linewidth=1.8, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (EMA lissée)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Train Acc ---
    ax = axes[1]
    ax.plot(hist_erm["step"], hist_erm["train_acc"], label="ERM train", alpha=0.8)
    ax.plot(hist_irm["step"], hist_irm["train_acc"], label="IRM train", alpha=0.8)
    ax.plot(hist_erm["step"], hist_erm["val_acc"], label="ERM val InD", ls="--", alpha=0.8)
    ax.plot(hist_irm["step"], hist_irm["val_acc"], label="IRM val InD", ls="--", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (Train & Val InD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Test OOD Acc ---
    ax = axes[2]
    ax.plot(hist_erm["step"], hist_erm["test_acc"], label="ERM test OOD", alpha=0.8)
    ax.plot(hist_irm["step"], hist_irm["test_acc"], label="IRM test OOD", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (Test OOD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Projections des poids ---
    ax = axes[3]
    has_weights = (
        any(v != 0.0 for v in hist_erm.get("w_z", []))
        or any(v != 0.0 for v in hist_irm.get("w_z", []))
    )
    if has_weights:
        ax.plot(hist_erm["step"], _ema_smooth(hist_erm["w_z"], smooth_alpha),
                color="C0", ls="-",  linewidth=1.8, label="ERM — dir causale (Y)")
        ax.plot(hist_erm["step"], _ema_smooth(hist_erm["w_y"], smooth_alpha),
                color="C0", ls="--", linewidth=1.8, label="ERM — dir spurieuse (A)")
        ax.plot(hist_irm["step"], _ema_smooth(hist_irm["w_z"], smooth_alpha),
                color="C1", ls="-",  linewidth=1.8, label="IRM — dir causale (Y)")
        ax.plot(hist_irm["step"], _ema_smooth(hist_irm["w_y"], smooth_alpha),
                color="C1", ls="--", linewidth=1.8, label="IRM — dir spurieuse (A)")
        ax.set_ylabel("|cos(w, direction)|")
    else:
        ax.text(0.5, 0.5, "Directions non disponibles",
                ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.set_xlabel("Step")
    ax.set_title("Projections des poids\n(directions causale vs spurieuse)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Courbes sauvegardées dans {out_dir}/training_curves.png")


def plot_fpr_comparison(results: dict, out_dir: str):
    """Bar chart comparant FPR(Y=0,A=1) et FNR(Y=1,A=0) entre ERM et IRM."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    metrics = [
        ("fpr_civil_identity",   "FPR – Civils avec Identité\n(Y=0, A=1)",   "FPR"),
        ("fnr_toxic_no_identity", "FNR – Toxiques sans Identité\n(Y=1, A=0)", "FNR"),
    ]
    colors = ["#e74c3c", "#2ecc71"]
    methods = ["ERM", "IRM"]

    for ax, (key, title, ylabel) in zip(axes, metrics):
        values = [
            results["erm"]["test_ood"][key],
            results["irm"]["test_ood"][key],
        ]
        bars = ax.bar(methods, values, color=colors, width=0.5, edgecolor="black")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.3 + 0.05)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Taux d'erreur sur les groupes contre-intuitifs – Test OOD", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fpr_comparison.png"), dpi=150)
    plt.close()
    print(f"  FPR/FNR plot sauvegardé dans {out_dir}/fpr_comparison.png")


def plot_accuracy_comparison(results: dict, out_dir: str):
    """Bar chart comparant les accuracies sur les différents sets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sets = ["val_ind", "test_ood"]
    labels = ["Val InD", "Test OOD"]
    x = np.arange(len(sets))
    width = 0.3

    erm_accs = [results["erm"][s]["accuracy"] for s in sets]
    irm_accs = [results["irm"][s]["accuracy"] for s in sets]

    ax.bar(x - width / 2, erm_accs, width, label="ERM", color="#e74c3c", edgecolor="black")
    ax.bar(x + width / 2, irm_accs, width, label="IRM", color="#2ecc71", edgecolor="black")

    for i, (e, r) in enumerate(zip(erm_accs, irm_accs)):
        ax.text(i - width / 2, e + 0.01, f"{e:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparaison ERM vs IRM – CivilComments")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"  Accuracy plot sauvegardé dans {out_dir}/accuracy_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CivilComments – ERM vs IRM (IRMv1)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_per_group", type=int, default=None,
                        help="Taille max par groupe (Y,A) dans chaque env (défaut: pas de limite)")
    parser.add_argument("--bert_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--embed_batch", type=int, default=64)

    # Entraînement
    parser.add_argument("--erm_steps", type=int, default=100_000)
    parser.add_argument("--erm_lr", type=float, default=1e-4)
    parser.add_argument("--irm_steps", type=int, default=100_000)
    parser.add_argument("--irm_lr", type=float, default=1e-4)
    parser.add_argument("--irm_lambda", type=float, default=1200.0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval_every", type=int, default=100)

    # Sortie
    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "plots" / "base"))
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Répertoire racine WILDS (données dans root_dir/civilcomments_v1.0/)")

    args = parser.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 1 : Chargement du dataset + variables Y, A
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement du dataset CivilComments")
    print("=" * 70)
    ds = load_civilcomments(root_dir=args.root_dir)  # dict of SimpleNamespace

    cap_str = str(args.max_per_group) if args.max_per_group is not None else "aucune"
    print(f"  Limite max_per_group : {cap_str}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2 : Environnements d'entraînement
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Construction des environnements biaisés")
    print("=" * 70)

    idx_e1, idx_e2 = build_train_envs(ds, max_per_group=args.max_per_group, seed=args.seed)
    print(f"  E1 (biais fort)  : {len(idx_e1):,} exemples")
    print(f"  E2 (biais modéré): {len(idx_e2):,} exemples")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Val InD (5 % de chaque env) + Test/Val OOD
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Splits de validation et de test")
    print("=" * 70)

    rng = np.random.default_rng(args.seed + 1)
    idx_e1_train, idx_e1_val = _split_val_from_indices(idx_e1, 0.05, rng)
    idx_e2_train, idx_e2_val = _split_val_from_indices(idx_e2, 0.05, rng)

    idx_val_ind = np.concatenate([idx_e1_val, idx_e2_val])
    print(f"  Val InD : {len(idx_val_ind):,} exemples")

    idx_test_ood = build_test_ood(
        ds, max_per_group=args.max_per_group, seed=args.seed
    )
    print(f"  Test OOD: {len(idx_test_ood):,} exemples")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3b : Extraction des embeddings DistilBERT
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3b : Embeddings DistilBERT")
    print("=" * 70)

    # Rassembler tous les textes nécessaires d'un coup pour ne pas
    # encoder le même texte deux fois.
    train_split = ds["train_pool"]
    test_split = ds["test"]

    # -- Textes train (E1_train + E2_train + val_ind, tous dans ds["train_pool"])
    all_train_indices = np.unique(
        np.concatenate([idx_e1_train, idx_e2_train, idx_val_ind])
    )
    # Mapping : indice original → position dans all_train_indices
    train_pos = {int(orig): pos for pos, orig in enumerate(all_train_indices)}

    texts_train = [train_split.comment_text[int(i)] for i in all_train_indices]
    print(f"  Encodage de {len(texts_train):,} textes (train) …")
    emb_train_all = embed_texts(
        texts_train,
        model_name=args.bert_model,
        max_length=args.max_length,
        device=device,
        batch_size=args.embed_batch,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Directions dans l'espace des embeddings
    # ─────────────────────────────────────────────────────────────────────
    # dir_sem  : direction causale    — diff de moyennes Y=1 vs Y=0
    # dir_conf : direction spurieuse  — diff de moyennes A=1 vs A=0
    # Ces vecteurs sont injectés dans les metas des envs d'entraînement ;
    # models_training.py les utilise pour projeter w à chaque step
    # et logguer w_z (proj. causale) et w_y (proj. spurieuse).
    Y_all_tr = np.asarray(train_split.Y)[all_train_indices]
    A_all_tr = np.asarray(train_split.A)[all_train_indices]

    dir_sem = emb_train_all[Y_all_tr == 1].mean(0) - emb_train_all[Y_all_tr == 0].mean(0)
    dir_sem = (dir_sem / (np.linalg.norm(dir_sem) + 1e-9)).astype(np.float32)

    dir_conf = emb_train_all[A_all_tr == 1].mean(0) - emb_train_all[A_all_tr == 0].mean(0)
    dir_conf = (dir_conf / (np.linalg.norm(dir_conf) + 1e-9)).astype(np.float32)

    cos_dc = float(np.dot(dir_sem, dir_conf))
    print(f"  dir_sem ‖={np.linalg.norm(dir_sem):.3f}  "
          f"dir_conf ‖={np.linalg.norm(dir_conf):.3f}  "
          f"cos(sem,conf)={cos_dc:.3f}")

    # -- Textes test (tous dans ds["test"])
    all_test_indices = np.unique(idx_test_ood)
    test_pos = {int(orig): pos for pos, orig in enumerate(all_test_indices)}

    texts_test = [test_split.comment_text[int(i)] for i in all_test_indices]
    print(f"  Encodage de {len(texts_test):,} textes (test) …")
    emb_test_all = embed_texts(
        texts_test,
        model_name=args.bert_model,
        max_length=args.max_length,
        device=device,
        batch_size=args.embed_batch,
    )

    # ─────────────────────────────────────────────────────────────────————
    # Helper : créer un Env à partir d'indices dans un split
    # ─────────────────────────────────────────────────────────────────————
    def _make_env_from_split(split, indices, emb_all, pos_map, meta=None):
        positions = np.array([pos_map[int(i)] for i in indices])
        emb = emb_all[positions]
        Y = np.asarray(split.Y)[indices].astype(np.float32)
        A = np.asarray(split.A)[indices].astype(np.int64)
        return make_env(emb, Y, identities=A, meta=meta)

    # Construction des Env
    env_e1 = _make_env_from_split(
        train_split, idx_e1_train, emb_train_all, train_pos,
        meta={"name": "E1_strong_bias", "dir_sem": dir_sem, "dir_conf": dir_conf},
    )
    env_e2 = _make_env_from_split(
        train_split, idx_e2_train, emb_train_all, train_pos,
        meta={"name": "E2_moderate_bias", "dir_sem": dir_sem, "dir_conf": dir_conf},
    )
    env_val_ind = _make_env_from_split(
        train_split, idx_val_ind, emb_train_all, train_pos,
        meta={"name": "val_ind"},
    )
    env_test_ood = _make_env_from_split(
        test_split, idx_test_ood, emb_test_all, test_pos,
        meta={"name": "test_ood"},
    )

    train_envs = [env_e1, env_e2]
    val_envs_for_log = [env_val_ind]

    print(f"\n  Env E1   : X {env_e1.X.shape}  Y=1: {env_e1.y.mean():.2%}")
    print(f"  Env E2   : X {env_e2.X.shape}  Y=1: {env_e2.y.mean():.2%}")
    print(f"  Val InD  : X {env_val_ind.X.shape}")
    print(f"  Test OOD : X {env_test_ood.X.shape}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4 : Entraînement ERM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4a : Entraînement ERM")
    print("=" * 70)

    model_erm, hist_erm = train_erm(
        envs=train_envs,
        steps=args.erm_steps,
        lr=args.erm_lr,
        batch=args.batch,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        val_envs=val_envs_for_log,
        test_env=env_test_ood,
        dataset_name="civilcomments",
        n_classes=2,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4b : Entraînement IRM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4b : Entraînement IRM")
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
        val_envs=val_envs_for_log,
        test_env=env_test_ood,
        dataset_name="civilcomments",
        n_classes=2,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5 : Évaluation finale
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Évaluation finale")
    print("=" * 70)

    results = {"erm": {}, "irm": {}}

    print("\n--- ERM ---")
    results["erm"]["val_ind"] = full_evaluation(model_erm, env_val_ind, device, "Val InD")
    results["erm"]["test_ood"] = full_evaluation(model_erm, env_test_ood, device, "Test OOD")

    print("\n--- IRM ---")
    results["irm"]["val_ind"] = full_evaluation(model_irm, env_val_ind, device, "Val InD")
    results["irm"]["test_ood"] = full_evaluation(model_irm, env_test_ood, device, "Test OOD")

    # ─────────────────────────────────────────────────────────────────────
    # Résumé
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"{'Métrique':<30} {'ERM':>10} {'IRM':>10}")
    print("-" * 52)
    print(f"{'Acc Test OOD':<30} "
          f"{results['erm']['test_ood']['accuracy']:>10.4f} "
          f"{results['irm']['test_ood']['accuracy']:>10.4f}")
    print(f"{'Acc Val InD':<30} "
          f"{results['erm']['val_ind']['accuracy']:>10.4f} "
          f"{results['irm']['val_ind']['accuracy']:>10.4f}")
    print(f"{'FPR(Y=0,A=1) Test OOD':<30} "
          f"{results['erm']['test_ood']['fpr_civil_identity']:>10.4f} "
          f"{results['irm']['test_ood']['fpr_civil_identity']:>10.4f}")
    print(f"{'FNR(Y=1,A=0) Test OOD':<30} "
          f"{results['erm']['test_ood']['fnr_toxic_no_identity']:>10.4f} "
          f"{results['irm']['test_ood']['fnr_toxic_no_identity']:>10.4f}")
    print(f"{'Loss Test OOD':<30} "
          f"{results['erm']['test_ood']['loss']:>10.4f} "
          f"{results['irm']['test_ood']['loss']:>10.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # Sauvegarde des résultats et plots
    # ─────────────────────────────────────────────────────────────────────
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats JSON sauvegardés dans {args.out_dir}/results.json")

    plot_training_curves(hist_erm, hist_irm, args.out_dir)
    plot_fpr_comparison(results, args.out_dir)
    plot_accuracy_comparison(results, args.out_dir)

    print("\nTerminé !")


if __name__ == "__main__":
    main()
