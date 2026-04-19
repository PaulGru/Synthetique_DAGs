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
run_moji_erm_vs_irm.py
======================
Compare ERM vs IRM (IRMv1) sur la tâche de classification de sentiment
du dataset LabHC/moji (~1.6M tweets Twitter).

Variable de confounding A : dialecte linguistique
    sa=1 → SAE (Standard American English) — A=0
    sa=0 → AAE (African American English)  — A=1

Biais spurieux : les tweets AAE (A=1) sont sur-représentés dans les
exemples négatifs (Y=0). Un modèle ERM peut apprendre le raccourci
A=1 → Y=0 et pénaliser les tweets AAE positifs.

Objectif : montrer qu'IRM réduit le taux d'erreur sur le groupe
**Positifs AAE (Y=1, A=1)** — FNR capturé via le Test OOD.

Pipeline :
    1. Chargement depuis HuggingFace (LabHC/moji)
    2. Construction de 2 environnements biaisés (biais fort / biais modéré)
    3. Val InD (5 % de chaque env), Test OOD (groupes contre-intuitifs)
    4. Embeddings DistilBERT gelé (mean pooling)
    5. Entraînement ERM et IRM
    6. Évaluation : Accuracy, FNR(Y=1,A=1), FPR(Y=0,A=0)

Usage :
    uv run run_moji_erm_vs_irm.py
    uv run run_moji_erm_vs_irm.py --device auto --irm_lambda 100
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

from data_synth import Env
from models_training import train_erm, train_irm
from utils_irm import resolve_device


# =============================================================================
# 1. Chargement du dataset LabHC/moji
# =============================================================================

def load_moji() -> dict:
    """
    Charge le dataset LabHC/moji depuis HuggingFace et concatène
    les trois splits (train + dev + test) en un seul pool global.

    Colonnes HF :
        text  : str  — texte du tweet
        label : int  — 0=négatif, 1=positif
        sa    : int  — 1=SAE (Standard American), 0=AAE

    On pose :
        Y = label  (0=négatif, 1=positif)
        A = 1 - sa (A=1 = AAE, A=0 = SAE)

    Returns
    -------
    {"all": SimpleNamespace(comment_text, Y, A)}
        Le namespace "all" contient l'union de tous les splits HF.
        Tous les indices utilisés dans les fonctions aval (build_train_envs,
        build_test_from_pool, etc.) font référence à ce pool global.
    """
    from datasets import load_dataset

    print("Chargement du dataset LabHC/moji depuis HuggingFace (3 splits)…")
    ds_hf = load_dataset("LabHC/moji")

    all_texts: list = []
    all_Y_parts: list = []
    all_A_parts: list = []

    for hf_name in ("train", "dev", "test"):
        hf_split = ds_hf[hf_name]
        texts = [t if isinstance(t, str) else "" for t in hf_split["text"]]
        Y = np.array(hf_split["label"], dtype=np.int64)
        A = (1 - np.array(hf_split["sa"], dtype=np.int64))  # A=1 = AAE

        all_texts.extend(texts)
        all_Y_parts.append(Y)
        all_A_parts.append(A)

        p_aae = Y[A == 1].mean() if (A == 1).any() else float("nan")
        p_sae = Y[A == 0].mean() if (A == 0).any() else float("nan")
        print(
            f"  [{hf_name}] N={len(Y):,}  "
            f"P(pos|AAE)={p_aae:.2%}  P(pos|SAE)={p_sae:.2%}"
        )

    Y_all = np.concatenate(all_Y_parts)
    A_all = np.concatenate(all_A_parts)
    ns = SimpleNamespace(comment_text=all_texts, Y=Y_all, A=A_all)

    print(f"\n  [POOL TOTAL] N={len(Y_all):,}")
    for y_v in (0, 1):
        for a_v in (0, 1):
            n = int(((Y_all == y_v) & (A_all == a_v)).sum())
            lbl = f"{'Neg' if y_v == 0 else 'Pos'} {'AAE' if a_v == 1 else 'SAE'}"
            print(f"    {lbl}: {n:>9,}")

    # Expose aussi les splits individuels (rétrocompatibilité avec
    # train_moji_finetuned_erm.py et eval_moji_dfr_irm_ft.py)
    result: dict = {"all": ns}
    offset = 0
    for hf_name, Y_p, A_p in zip(
        ("train", "dev", "test"), all_Y_parts, all_A_parts
    ):
        n = len(Y_p)
        texts_p = all_texts[offset : offset + n]
        result[hf_name] = SimpleNamespace(comment_text=texts_p, Y=Y_p, A=A_p)
        offset += n
    return result


# =============================================================================
# 2. Construction des environnements d'entraînement par sous-échantillonnage
# =============================================================================

def _get_group_indices(split, y_val: int, a_val: int) -> np.ndarray:
    """Renvoie les indices du split où Y==y_val et A==a_val."""
    Y = np.asarray(split.Y)
    A = np.asarray(split.A)
    return np.where((Y == y_val) & (A == a_val))[0]


def build_train_envs(
    ds: dict,
    excluded: np.ndarray | None = None,
    p_e1: float = 0.80,
    p_e2: float = 0.70,
    seed: int = 42,
    include_anticorr_env: bool = False,
    sae_ratio: float = 2.0,
    # legacy alias
    max_per_group: int | None = None,
) -> List[np.ndarray]:
    """
    Construit E1 et E2 (et optionnellement E3) depuis le pool global (ds["all"])
    en excluant les exemples réservés pour le test et la validation.

    Conception des environnements
    ─────────────────────────────
    Les deux corrélations spurieuses sont symétriques :
        P(neg|AAE) = P(pos|SAE) = p_e1  pour E1
        P(neg|AAE) = P(pos|SAE) = p_e2  pour E2

    Ratio dialectes (sae_ratio)
    ───────────────────────────
    Reflète le déséquilibre naturel du dataset où SAE >> AAE.
        sae_ratio = 2.0 (défaut) → N_SAE = 2 × N_AAE
            AAE : 1/3 de l'environnement
            SAE : 2/3 de l'environnement
        sae_ratio = 1.0 → équilibre parfait AAE = SAE (comportement historique)
    Les données AAE sont conservées intégralement (groupe limitant) ;
    SAE est suréchantillonné en proportion.

    Pipeline
    ────────
    1. Pool par groupe = tous les indices hors `excluded`, partitionné en 50/50
       entre E1 et E2 (disjonction stricte).
    2. Anchor sur neg_aae (groupe limitant) → dériver pos_aae.
    3. total_sae = sae_ratio × total_aae → dériver neg_sae et pos_sae.
    4. Fallback sur le groupe le plus rare si l'une des tailles dépasse son pool.

    include_anticorr_env
    ────────────────────
    Si True, construit un E3 anti-corrélé (P(neg|AAE)=30 %, P(pos|SAE)=30 %)
    à partir des exemples NON consommés par E1 et E2 (aucun doublon, aucune
    donnée test/val utilisée).

    Returns
    ───────
    list : [idx_e1, idx_e2]  ou  [idx_e1, idx_e2, idx_e3]
        Tous les indices font référence au pool global ds["all"].
    """
    if excluded is None:
        excluded = np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    all_split = ds["all"]
    excluded_set = set(int(i) for i in excluded)

    # ── Étape 1 : pool disponible par groupe, partitionné 50/50 entre E1 et E2 ──
    pool: dict = {}
    for y_val in (0, 1):
        for a_val in (0, 1):
            all_idx = _get_group_indices(all_split, y_val, a_val)
            avail = all_idx[~np.isin(all_idx, excluded)]
            perm = rng.permutation(len(avail))
            cut = len(avail) // 2
            pool[(y_val, a_val, 1)] = avail[perm[:cut]]
            pool[(y_val, a_val, 2)] = avail[perm[cut:]]

    # ── Étape 2 : échantillonnage avec tracking ──
    def _sample_env_balanced(env_id: int, p_neg_aae: float, p_pos_sae: float):
        idx_neg_aae = pool[(0, 1, env_id)]
        idx_pos_aae = pool[(1, 1, env_id)]
        idx_neg_sae = pool[(0, 0, env_id)]
        idx_pos_sae = pool[(1, 0, env_id)]

        # --- AAE : anchor sur neg_aae (groupe limitant), dériver pos_aae ---
        n_neg_aae = len(idx_neg_aae)
        n_pos_aae = int(n_neg_aae * (1 - p_neg_aae) / p_neg_aae)
        if n_pos_aae > len(idx_pos_aae):
            n_pos_aae = len(idx_pos_aae)
            n_neg_aae = int(n_pos_aae * p_neg_aae / (1 - p_neg_aae))
        total_aae = n_neg_aae + n_pos_aae

        # --- SAE : total_sae = sae_ratio × total_aae ---
        # Reflète le déséquilibre naturel SAE >> AAE du dataset.
        total_sae_target = int(round(total_aae * sae_ratio))
        n_neg_sae = int(round(total_sae_target * (1 - p_pos_sae)))
        n_pos_sae = total_sae_target - n_neg_sae

        # Fallbacks si un groupe SAE est épuisé
        if n_pos_sae > len(idx_pos_sae):
            n_pos_sae = len(idx_pos_sae)
            n_neg_sae = int(n_pos_sae * (1 - p_pos_sae) / p_pos_sae)
        if n_neg_sae > len(idx_neg_sae):
            n_neg_sae = len(idx_neg_sae)
            n_pos_sae = int(n_neg_sae * p_pos_sae / (1 - p_pos_sae))

        selected = {
            (0, 1): rng.choice(idx_neg_aae, size=n_neg_aae, replace=False),
            (1, 1): rng.choice(idx_pos_aae, size=n_pos_aae, replace=False),
            (1, 0): rng.choice(idx_pos_sae, size=n_pos_sae, replace=False),
            (0, 0): rng.choice(idx_neg_sae, size=n_neg_sae, replace=False),
        }
        combined = np.concatenate(list(selected.values()))
        rng.shuffle(combined)
        return combined, selected

    idx_e1, sel_e1 = _sample_env_balanced(1, p_neg_aae=p_e1, p_pos_sae=p_e1)
    idx_e2, sel_e2 = _sample_env_balanced(2, p_neg_aae=p_e2, p_pos_sae=p_e2)

    if not include_anticorr_env:
        return [idx_e1, idx_e2]

    # ── Étape 3 : exemples non consommés par E1 ou E2 → E3 anti-corrélé ──
    unused: dict = {}
    for y_val in (0, 1):
        for a_val in (0, 1):
            key = (y_val, a_val)
            used_set = set(sel_e1[key].tolist()) | set(sel_e2[key].tolist())
            all_pool = np.concatenate([pool[(y_val, a_val, 1)], pool[(y_val, a_val, 2)]])
            unused[key] = all_pool[~np.isin(all_pool, list(used_set))]

    # E3 : P(neg|AAE)=30 %, P(pos|SAE)=30 %  (corrélation inversée)
    p_neg_aae_e3, p_pos_sae_e3 = 0.30, 0.30
    n_neg_a1_e3 = len(unused[(0, 1)])
    n_pos_a1_e3 = int(n_neg_a1_e3 * (1 - p_neg_aae_e3) / p_neg_aae_e3)
    if n_pos_a1_e3 > len(unused[(1, 1)]):
        n_pos_a1_e3 = len(unused[(1, 1)])
        n_neg_a1_e3 = int(n_pos_a1_e3 * p_neg_aae_e3 / (1 - p_neg_aae_e3))

    n_neg_a0_e3 = len(unused[(0, 0)])
    n_pos_a0_e3 = int(n_neg_a0_e3 * p_pos_sae_e3 / (1 - p_pos_sae_e3))
    if n_pos_a0_e3 > len(unused[(1, 0)]):
        n_pos_a0_e3 = len(unused[(1, 0)])
        n_neg_a0_e3 = int(n_pos_a0_e3 * (1 - p_pos_sae_e3) / p_pos_sae_e3)

    idx_e3 = np.concatenate([
        rng.choice(unused[(0, 1)], size=n_neg_a1_e3, replace=False),
        rng.choice(unused[(1, 1)], size=n_pos_a1_e3, replace=False),
        rng.choice(unused[(1, 0)], size=n_pos_a0_e3, replace=False),
        rng.choice(unused[(0, 0)], size=n_neg_a0_e3, replace=False),
    ])
    rng.shuffle(idx_e3)
    return [idx_e1, idx_e2, idx_e3]


# =============================================================================
# 3. Embeddings DistilBERT (gelé, mean pooling)
# =============================================================================

def embed_texts(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Calcule les embeddings DistilBERT (mean pooling) pour une liste de textes.
    Le modèle est gelé — pas de fine-tuning.

    Accélération automatique :
    - batch_size=256 par défaut (augmenter si la VRAM le permet, ex. 512)
    - fp16 (autocast) activé automatiquement sur CUDA → ~2× plus rapide

    Returns : np.ndarray de shape (N, 768).
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    use_fp16 = str(device).startswith("cuda")
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
            if use_fp16:
                with torch.autocast(device_type="cuda"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        hidden = out.last_hidden_state.float()   # (B, seq, 768) — cast fp32 après autocast
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


def build_test_ood(ds: dict, n_per_group: int = 2500, seed: int = 42) -> np.ndarray:
    """
    Carve le Test OOD équilibré depuis le pool global (ds["all"]).

    Prélève n_per_group exemples de chacun des 4 groupes (Y×A), soit
    4 × n_per_group exemples au total.  Ces indices sont réservés EN PREMIER
    et exclus de tout usage d'entraînement ou de validation.

    n_per_group : défaut 2500 → 10 000 total.
    Returns : indices dans ds["all"] (tableau 1-D).
    """
    rng = np.random.default_rng(seed)
    all_split = ds["all"]

    min_available = min(
        len(_get_group_indices(all_split, y_val, a_val))
        for y_val in (0, 1) for a_val in (0, 1)
    )
    n = min(n_per_group, min_available)
    if n < n_per_group:
        print(
            f"  [Test OOD] n_per_group capé à {n:,} "
            f"(groupe le plus rare: {min_available:,} exemples)"
        )

    sel = []
    for y_val in (0, 1):
        for a_val in (0, 1):
            idx = _get_group_indices(all_split, y_val, a_val)
            sel.append(rng.choice(idx, size=n, replace=False))
    result = np.concatenate(sel)
    rng.shuffle(result)
    print(f"  [Test OOD] {len(result):,} exemples  (4 × {n:,}, équilibré)")
    return result


def build_val_biased_from_pool(
    ds: dict,
    excluded: np.ndarray,
    p_neg_aae: float = 0.75,
    p_pos_sae: float = 0.75,
    max_anchor: int = 5000,
    seed: int = 43,
) -> np.ndarray:
    """
    Construit un ensemble de validation biaisé depuis le pool global (ds["all"])
    en excluant tous les indices `excluded` (test OOD + anti-corr déjà réservés).

    Distribution : intermédiaire entre E1 (80 %) et E2 (70 %) → 75/25 par défaut.
    Logique minority-anchored identique à celle des envs d'entraînement.
    La val ne sert qu'au monitoring du training ; pas besoin de très grande taille.

    Returns : indices dans ds["all"].
    """
    rng = np.random.default_rng(seed)
    all_split = ds["all"]

    def _avail(y_val, a_val):
        idx = _get_group_indices(all_split, y_val, a_val)
        return idx[~np.isin(idx, excluded)]

    idx_neg_aae = _avail(0, 1)
    idx_pos_aae = _avail(1, 1)
    idx_neg_sae = _avail(0, 0)
    idx_pos_sae = _avail(1, 0)

    # AAE : anchor sur neg_aae, capé à max_anchor pour ne pas priver le train
    n_neg_aae = min(len(idx_neg_aae), max_anchor)
    n_pos_aae = int(n_neg_aae * (1 - p_neg_aae) / p_neg_aae)
    if n_pos_aae > len(idx_pos_aae):
        n_pos_aae = len(idx_pos_aae)
        n_neg_aae = int(n_pos_aae * p_neg_aae / (1 - p_neg_aae))
    total_aae = n_neg_aae + n_pos_aae

    # SAE : total_sae = total_aae (équilibre dialectes)
    n_neg_sae = int(round(total_aae * (1 - p_pos_sae)))
    n_pos_sae = total_aae - n_neg_sae
    if n_pos_sae > len(idx_pos_sae):
        n_pos_sae = len(idx_pos_sae)
        n_neg_sae = int(n_pos_sae * (1 - p_pos_sae) / p_pos_sae)
    if n_neg_sae > len(idx_neg_sae):
        n_neg_sae = len(idx_neg_sae)
        n_pos_sae = int(n_neg_sae * p_pos_sae / (1 - p_pos_sae))

    sel = np.concatenate([
        rng.choice(idx_neg_aae, size=n_neg_aae, replace=False),
        rng.choice(idx_pos_aae, size=n_pos_aae, replace=False),
        rng.choice(idx_neg_sae, size=n_neg_sae, replace=False),
        rng.choice(idx_pos_sae, size=n_pos_sae, replace=False),
    ])
    rng.shuffle(sel)
    total_aae_out = n_neg_aae + n_pos_aae
    total_sae_out = n_neg_sae + n_pos_sae
    print(
        f"  [Val biaisée] {len(sel):,} exemples  "
        f"P(neg|AAE)={n_neg_aae/(total_aae_out or 1):.0%}  "
        f"P(pos|SAE)={n_pos_sae/(total_sae_out or 1):.0%}"
    )
    return sel


def build_val_ind(ds: dict, seed: int = 43) -> np.ndarray:
    """Alias de compatibilité — appelle build_val_biased_from_pool sans excluded."""
    return build_val_biased_from_pool(ds, excluded=np.array([], dtype=np.int64), seed=seed)


# =============================================================================
# 5. Assemblage des Env
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
# 6. Affichage de la composition d'un split
# =============================================================================

def print_split_stats(
    name: str,
    indices: np.ndarray,
    Y_all: np.ndarray,
    A_all: np.ndarray,
) -> None:
    """
    Affiche un tableau de répartition Y×A pour un split donné.

      │              │  Négatif (Y=0)   │  Positif (Y=1)   │   Total  │
      │ AAE  (A=1)   │  neg_aae  (xx%)  │  pos_aae  (xx%)  │  N_aae  │
      │ SAE  (A=0)   │  neg_sae  (xx%)  │  pos_sae  (xx%)  │  N_sae  │
      │ Total        │  N_neg    (xx%)  │  N_pos    (xx%)  │    N   │
    """
    Y = Y_all[indices]
    A = A_all[indices]
    n_total = len(indices)

    neg_aae = int(((Y == 0) & (A == 1)).sum())
    pos_aae = int(((Y == 1) & (A == 1)).sum())
    neg_sae = int(((Y == 0) & (A == 0)).sum())
    pos_sae = int(((Y == 1) & (A == 0)).sum())
    total_aae = neg_aae + pos_aae
    total_sae = neg_sae + pos_sae
    total_neg = neg_aae + neg_sae
    total_pos = pos_aae + pos_sae

    def _pct(n: int, d: int) -> str:
        return f"{n/d:5.1%}" if d > 0 else "  —  "

    W = 70
    print(f"\n  ┬{'─'*W}┐")
    print(f"  │  {name:<{W-2}}│")
    print(f"  ├{'─'*22}┬{'─'*22}┬{'─'*22}┬{'─'*4}┤")
    print(f"  │{'':22}│  {'Négatif (Y=0)':>18}  │  {'Positif (Y=1)':>18}  │ {'Total':>6} │")
    print(f"  ├{'─'*22}┬{'─'*22}┬{'─'*22}┬{'─'*8}┤")
    print(
        f"  │ {'AAE  (A=1)':<20} │"
        f"  {neg_aae:>9,}  {_pct(neg_aae, total_aae)} │"
        f"  {pos_aae:>9,}  {_pct(pos_aae, total_aae)} │"
        f" {total_aae:>7,} │"
    )
    print(
        f"  │ {'SAE  (A=0)':<20} │"
        f"  {neg_sae:>9,}  {_pct(neg_sae, total_sae)} │"
        f"  {pos_sae:>9,}  {_pct(pos_sae, total_sae)} │"
        f" {total_sae:>7,} │"
    )
    print(f"  ├{'─'*22}┼{'─'*22}┼{'─'*22}┼{'─'*8}┤")
    print(
        f"  │ {'Total':<20} │"
        f"  {total_neg:>9,}  {_pct(total_neg, n_total)} │"
        f"  {total_pos:>9,}  {_pct(total_pos, n_total)} │"
        f" {n_total:>7,} │"
    )
    print(f"  └{'─'*22}┴{'─'*22}┴{'─'*22}┴{'─'*8}╛")


# =============================================================================
# 7. Métriques
# =============================================================================

def compute_predictions(model: nn.Module, env: Env, device: str = "cpu") -> np.ndarray:
    """Renvoie les prédictions binaires (0/1) pour un Env."""
    model.eval()
    with torch.no_grad():
        logits = model(env.X.to(device))
    probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return (probs >= 0.5).astype(np.float32)


def compute_error_rate_group(
    model: nn.Module,
    env: Env,
    y_target: int,
    a_target: int,
    device: str = "cpu",
) -> float:
    """
    Taux d'erreur sur le sous-groupe (Y=y_target, A=a_target).
    error_rate = #(pred != y_target) / #(true=y_target)  pour le sous-groupe.

    Pour (Y=1, A=1) : FNR — positifs AAE prédits comme négatifs.
    Pour (Y=0, A=0) : FPR — négatifs SAE prédits comme positifs.
    """
    y_true = env.y.cpu().numpy().reshape(-1)
    A = env.meta["A"].cpu().numpy().reshape(-1)
    y_pred = compute_predictions(model, env, device=device)

    mask = (y_true == y_target) & (A == a_target)
    if mask.sum() == 0:
        return float("nan")

    errors = (y_pred[mask] != y_target)
    return float(errors.mean())


def full_evaluation(
    model: nn.Module,
    env: Env,
    device: str = "cpu",
    label: str = "",
) -> Dict:
    """
    Évalue l'ensemble des métriques sur un Env.

    Métriques globales :
        accuracy          : accuracy globale
        macro_f1          : macro-F1 (moyenne des F1 par classe)
        loss              : BCE loss

    Métriques par groupe (Y, A) :
        fnr_pos_aae       : FNR (Y=1,A=1)  — positifs AAE prédits négatifs
        fpr_neg_sae       : FPR (Y=0,A=0)  — négatifs SAE prédits positifs
        acc_groups        : accuracy des 4 groupes {(Y,A)}
        worst_group_acc   : min des 4 précisions de groupe
        avg_group_acc     : moyenne des 4 précisions de groupe

    Métriques de fairness :
        eod_tpr           : Equal Opportunity Difference  |TPR_A=0 − TPR_A=1|
        eod_fpr           : Equalized Odds (côté FPR)      |FPR_A=0 − FPR_A=1|
    """
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        logits = model(env.X.to(device)).squeeze()
    y = env.y.to(device).float()
    loss = nn.BCEWithLogitsLoss()(logits, y).item()

    y_pred = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
    y_np = env.y.cpu().numpy().reshape(-1)
    A_np = env.meta["A"].cpu().numpy().reshape(-1)

    acc = float((y_pred == y_np).mean())
    macro_f1 = float(f1_score(y_np, y_pred, average="macro", zero_division=0))

    # ── Métriques par groupe ──
    def _tpr(y_true, y_hat, a_mask):
        """TPR = P(ŷ=1|Y=1) pour le sous-groupe a_mask."""
        pos = (y_true == 1) & a_mask
        if pos.sum() == 0:
            return float("nan")
        return float(y_hat[pos].mean())

    def _fpr(y_true, y_hat, a_mask):
        """FPR = P(ŷ=1|Y=0) pour le sous-groupe a_mask."""
        neg = (y_true == 0) & a_mask
        if neg.sum() == 0:
            return float("nan")
        return float(y_hat[neg].mean())

    def _acc_group(y_true, y_hat, mask):
        if mask.sum() == 0:
            return float("nan")
        return float((y_hat[mask] == y_true[mask]).mean())

    mask_aae = A_np == 1
    mask_sae = A_np == 0

    tpr_aae = _tpr(y_np, y_pred, mask_aae)   # TPR pour AAE
    tpr_sae = _tpr(y_np, y_pred, mask_sae)   # TPR pour SAE
    fpr_aae = _fpr(y_np, y_pred, mask_aae)   # FPR pour AAE
    fpr_sae = _fpr(y_np, y_pred, mask_sae)   # FPR pour SAE

    fnr_pos_aae = 1.0 - tpr_aae if not np.isnan(tpr_aae) else float("nan")
    fpr_neg_sae = fpr_sae

    eod_tpr = abs(tpr_sae - tpr_aae) if not (np.isnan(tpr_sae) or np.isnan(tpr_aae)) else float("nan")
    eod_fpr = abs(fpr_sae - fpr_aae) if not (np.isnan(fpr_sae) or np.isnan(fpr_aae)) else float("nan")

    acc_groups = {
        "(Y=0,A=0)": _acc_group(y_np, y_pred, (y_np == 0) & mask_sae),
        "(Y=0,A=1)": _acc_group(y_np, y_pred, (y_np == 0) & mask_aae),
        "(Y=1,A=0)": _acc_group(y_np, y_pred, (y_np == 1) & mask_sae),
        "(Y=1,A=1)": _acc_group(y_np, y_pred, (y_np == 1) & mask_aae),
    }
    valid_accs = [v for v in acc_groups.values() if not np.isnan(v)]
    worst_group_acc = float(min(valid_accs)) if valid_accs else float("nan")
    avg_group_acc   = float(np.mean(valid_accs)) if valid_accs else float("nan")

    res = {
        "accuracy":       acc,
        "macro_f1":       macro_f1,
        "loss":           float(loss),
        "fnr_pos_aae":    fnr_pos_aae,
        "fpr_neg_sae":    fpr_neg_sae,
        "eod_tpr":        eod_tpr,    # Equal Opportunity Difference sur TPR
        "eod_fpr":        eod_fpr,    # Equalized Odds sur FPR
        "worst_group_acc": worst_group_acc,
        "avg_group_acc":   avg_group_acc,
        "acc_groups":      acc_groups,
    }
    if label:
        print(
            f"  [{label}] Acc={acc:.4f}  MacroF1={macro_f1:.4f}  Loss={loss:.4f}\n"
            f"           FNR(AAE)={fnr_pos_aae:.4f}  FPR(SAE)={fpr_neg_sae:.4f}\n"
            f"           EOD_TPR={eod_tpr:.4f}  EOD_FPR={eod_fpr:.4f}\n"
            f"           Worst-Group={worst_group_acc:.4f}  Avg-Group={avg_group_acc:.4f}"
        )
    return res


# =============================================================================
# 7. Visualisation
# =============================================================================

# Palette centralisée — cohérente entre tous les plots
_C = {
    "erm":       "#e05252",   # rouge saumon
    "irm":       "#4caf7d",   # vert sauge
    "aae":       "#5b8dd9",   # bleu moyen  (groupe AAE)
    "sae":       "#f5a623",   # orange doux (groupe SAE)
    "val_ind":   "#9b59b6",   # violet
    "test_ood":  "#16a085",   # vert canard
    "ideal":     "#bbb",      # gris pour la ligne "idéale"
}


def _annotate_bar(ax, bar, v, fmt=".3f", dy=0.012, fontsize=9):
    """Annote une barre avec sa valeur si non-NaN."""
    if not np.isnan(v):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{v:{fmt}}", ha="center", va="bottom", fontsize=fontsize,
        )


# =============================================================================
# PLOT 1 — Training dynamics
# =============================================================================

def plot_training_curves(hist_erm: dict, hist_irm: dict, out_dir: str):
    """
    01_training_curves.png
    ──────────────────────
    Three panels showing how ERM and IRM evolve during training:

    Left  — Training loss. Both curves should decrease; the IRM penalty
             (λ × invariance term) often causes a plateau or slight increase
             before the model finds an invariant solution.

    Center — Accuracy on training envs (solid) and Val InD (dashed).
             Val InD uses the same biased distribution as training (80/20),
             so we expect both models to reach high accuracy there — this is
             NOT a fairness signal, just a convergence check.

    Right  — Accuracy on Test OOD (natural distribution, unbiased).
             A model that relied on the spurious AAE↔negative correlation
             will *drop* here compared to Val InD. IRM should maintain or
             recover accuracy on OOD compared to ERM.
    """
    ERM_COLOR = _C["erm"]
    IRM_COLOR = _C["irm"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — Loss
    ax = axes[0]
    ax.plot(hist_erm["step"], hist_erm["loss"], color=ERM_COLOR, label="ERM", alpha=0.85, lw=1.5)
    ax.plot(hist_irm["step"], hist_irm["loss"], color=IRM_COLOR, label="IRM", alpha=0.85, lw=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 — Accuracy InD
    ax = axes[1]
    ax.plot(hist_erm["step"], hist_erm["train_acc"], color=ERM_COLOR, lw=1.5,
            label="ERM — train", alpha=0.85)
    ax.plot(hist_irm["step"], hist_irm["train_acc"], color=IRM_COLOR, lw=1.5,
            label="IRM — train", alpha=0.85)
    ax.plot(hist_erm["step"], hist_erm["val_acc"], color=ERM_COLOR, lw=1.5, ls="--",
            label="ERM — Val InD", alpha=0.7)
    ax.plot(hist_irm["step"], hist_irm["val_acc"], color=IRM_COLOR, lw=1.5, ls="--",
            label="IRM — Val InD", alpha=0.7)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy — Train & Val InD", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Accuracy OOD
    ax = axes[2]
    ax.plot(hist_erm["step"], hist_erm["test_acc"], color=ERM_COLOR, lw=1.5,
            label="ERM — Test OOD", alpha=0.85)
    ax.plot(hist_irm["step"], hist_irm["test_acc"], color=IRM_COLOR, lw=1.5,
            label="IRM — Test OOD", alpha=0.85)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy — Test OOD (natural distribution)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "01  —  Training dynamics: ERM vs IRM",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [01] Training curves  →  {out_dir}/01_training_curves.png")


# =============================================================================
# PLOT 2 — Per-group Macro-F1
# =============================================================================

def plot_per_group_f1(results: dict, env_sizes: dict, out_dir: str):
    """
    02_per_group_f1.png
    ───────────────────
    Macro-F1 broken down by the four (Y × A) subgroups, shown side-by-side
    for Val InD (left) and Test OOD (right).

    Why Macro-F1 instead of accuracy?
      The dataset is highly imbalanced (AAE ≈ 5 % of examples).  Accuracy is
      dominated by the majority group (SAE).  Macro-F1 averages precision and
      recall equally across both classes within each subgroup, giving a fairer
      picture of how well each group is served.

    How to read this plot:
      • Higher is always better for every bar.
      • A *fair* model produces bars of similar height across all four groups.
      • ERM typically excels on majority groups (SAE) and underperforms on AAE.
      • IRM aims to narrow those gaps, often at a small cost on majority groups.

    The N= annotations show how many examples are in each group — small groups
    have noisier estimates.

    Groups:
      Neg SAE (Y=0, A=0) — negative sentiment, Standard American English
      Neg AAE (Y=0, A=1) — negative sentiment, African American English
      Pos SAE (Y=1, A=0) — positive sentiment, Standard American English
      Pos AAE (Y=1, A=1) — positive sentiment, African American English
    """
    from sklearn.metrics import f1_score as sk_f1

    group_keys   = ["(Y=0,A=0)", "(Y=0,A=1)", "(Y=1,A=0)", "(Y=1,A=1)"]
    group_labels = [
        "Neg SAE",
        "Neg AAE",
        "Pos SAE",
        "Pos AAE",
    ]
    # Couleurs distinctes par groupe (même charte partout)
    group_colors = ["#f0a500", "#5b8dd9", "#e05252", "#4caf7d"]

    splits = [
        ("val_ind",  "Val InD\n(same distribution as training)"),
        ("test_ood", "Test OOD\n(natural distribution)"),
    ]

    # Macro-F1 par groupe : on le calcule directement depuis acc_groups
    # (on n'a pas les prédictions brutes ici, on utilise l'accuracy de groupe
    #  comme proxy — pour un vrai F1 par groupe il faudrait stocker preds)
    # On utilise le macro_f1 global + acc_groups pour l'affichage relatif.
    # NOTE : l'acc_groups est l'accuracy binaire du groupe, pas le F1.
    # On l'affiche en étiquetant clairement "Group Accuracy".

    x = np.arange(len(group_keys))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (split_key, split_label) in zip(axes, splits):
        erm_vals = [results["erm"][split_key]["acc_groups"][k] for k in group_keys]
        irm_vals = [results["irm"][split_key]["acc_groups"][k] for k in group_keys]

        for gi, (gk, gc) in enumerate(zip(group_keys, group_colors)):
            b_erm = ax.bar(
                gi - width / 2, erm_vals[gi], width,
                color=gc, edgecolor="black", alpha=0.6,
                hatch="", label="ERM" if gi == 0 else "_nolegend_",
            )
            b_irm = ax.bar(
                gi + width / 2, irm_vals[gi], width,
                color=gc, edgecolor="black", alpha=1.0,
                hatch="///", label="IRM" if gi == 0 else "_nolegend_",
            )
            _annotate_bar(ax, b_erm[0], erm_vals[gi], fmt=".2f", dy=0.010, fontsize=8)
            _annotate_bar(ax, b_irm[0], irm_vals[gi], fmt=".2f", dy=0.010, fontsize=8)

            # Taille de groupe
            n = env_sizes.get(split_key, {}).get(gk, None)
            if n is not None:
                ax.text(gi, -0.07, f"N={n:,}", ha="center", fontsize=7, color="#555",
                        transform=ax.get_xaxis_transform())

        # Ligne macro-F1 globale pour référence
        f1_erm = results["erm"][split_key]["macro_f1"]
        f1_irm = results["irm"][split_key]["macro_f1"]
        ax.axhline(f1_erm, color=_C["erm"], ls="--", lw=1.2, alpha=0.7,
                   label=f"ERM overall Macro-F1 = {f1_erm:.3f}")
        ax.axhline(f1_irm, color=_C["irm"], ls="--", lw=1.2, alpha=0.7,
                   label=f"IRM overall Macro-F1 = {f1_irm:.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_title(split_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Group Accuracy  (higher = better)", fontsize=9)
        ax.set_ylim(0, 1.20)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        # Légende couleur = groupe
        from matplotlib.patches import Patch
        group_legend = [Patch(fc=gc, label=gl.replace("\n", " "))
                        for gc, gl in zip(group_colors, group_labels)]
        ax.legend(handles=group_legend + [
            plt.Line2D([0], [0], color=_C["erm"], ls="--", lw=1.4,
                       label=f"ERM Macro-F1 = {f1_erm:.3f}"),
            plt.Line2D([0], [0], color=_C["irm"], ls="--", lw=1.4,
                       label=f"IRM Macro-F1 = {f1_irm:.3f}"),
        ], fontsize=7.5, loc="upper right")

        # Légende hachures
        ax.bar([], [], color="grey", alpha=0.6, label="ERM (plain)")
        ax.bar([], [], color="grey", alpha=1.0, hatch="///", label="IRM (hatched)")

    fig.suptitle(
        "02  —  Per-group accuracy & overall Macro-F1\n"
        "Plain bars = ERM  ·  Hatched bars = IRM  ·  "
        "Dashed lines = overall Macro-F1  ·  A fair model has uniform bar heights",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_per_group_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [02] Per-group accuracy  →  {out_dir}/02_per_group_accuracy.png")


# =============================================================================
# PLOT 3 — Fairness gaps (TPR / FPR by dialect group)
# =============================================================================

def plot_fairness_gaps(results: dict, out_dir: str):
    """
    03_fairness_gaps.png
    ────────────────────
    A 2 × 2 grid examining classification rates broken down by dialect group
    (AAE vs SAE), on Val InD (left column) and Test OOD (right column).

    Row 1 — True Positive Rate (TPR = sensitivity = recall on positive class)
      TPR = fraction of *actual positives* correctly predicted as positive.
      • Higher is better.
      • A FAIR model has equal TPR for both dialect groups (Equal Opportunity).
      • ERM's spurious correlation (AAE → negative) inflates TPR_SAE while
        depressing TPR_AAE, creating a large gap.
      • The EOD-TPR annotation measures this gap: |TPR_SAE − TPR_AAE|.
        → Closer to 0 means fairer.

    Row 2 — False Positive Rate (FPR)
      FPR = fraction of *actual negatives* incorrectly predicted as positive.
      • Lower is better.
      • A FAIR model has equal FPR for both dialect groups (Equalized Odds).
      • ERM produces very high FPR_SAE: it over-predicts positives for SAE
        because it learned that "SAE → positive" from the spurious correlation.
      • The EOD-FPR annotation measures |FPR_SAE − FPR_AAE|.
        → Closer to 0 means fairer.

    Colour coding:
      Blue bars  = AAE group (A=1, African American English)
      Orange bars = SAE group (A=0, Standard American English)
      Plain bars  = ERM  ·  Hatched bars = IRM
    """
    def _rates(r):
        return {
            "tpr_aae": 1.0 - r["fnr_pos_aae"],
            "tpr_sae": r["acc_groups"]["(Y=1,A=0)"],
            "fpr_aae": 1.0 - r["acc_groups"]["(Y=0,A=1)"],
            "fpr_sae": r["fpr_neg_sae"],
        }

    splits  = [("val_ind", "Val InD\n(same distribution as training)"),
               ("test_ood", "Test OOD\n(natural distribution)")]
    rows = [
        ("tpr_aae", "tpr_sae", "eod_tpr",
         "True Positive Rate (TPR)\nhigher = fewer missed positives  ·  goal: equal across groups"),
        ("fpr_aae", "fpr_sae", "eod_fpr",
         "False Positive Rate (FPR)\nlower = fewer false alarms  ·  goal: equal across groups"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_pos = np.array([0.0, 1.0])   # AAE=0, SAE=1
    w = 0.30

    for col, (split_key, split_label) in enumerate(splits):
        erm_r = _rates(results["erm"][split_key])
        irm_r = _rates(results["irm"][split_key])

        for row, (k_aae, k_sae, eod_key, ylabel) in enumerate(rows):
            ax = axes[row][col]

            erm_vals = [erm_r[k_aae], erm_r[k_sae]]
            irm_vals = [irm_r[k_aae], irm_r[k_sae]]
            eod_erm  = results["erm"][split_key][eod_key]
            eod_irm  = results["irm"][split_key][eod_key]
            colors   = [_C["aae"], _C["sae"]]

            for gi in range(2):
                b_erm = ax.bar(
                    gi - w / 2, erm_vals[gi], w,
                    color=colors[gi], edgecolor="black", alpha=0.55,
                )
                b_irm = ax.bar(
                    gi + w / 2, irm_vals[gi], w,
                    color=colors[gi], edgecolor="black", alpha=1.0, hatch="///",
                )
                _annotate_bar(ax, b_erm[0], erm_vals[gi], fmt=".3f", dy=0.012, fontsize=9)
                _annotate_bar(ax, b_irm[0], irm_vals[gi], fmt=".3f", dy=0.012, fontsize=9)

            # Gap annotation
            arrow_col = "#c0392b" if eod_erm > eod_irm else "#27ae60"
            ax.annotate(
                f"Gap  ERM = {eod_erm:.3f}   →   IRM = {eod_irm:.3f}",
                xy=(0.5, 0.04), xycoords="axes fraction", ha="center",
                fontsize=9, color=arrow_col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=arrow_col, lw=1),
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels(["AAE", "SAE"], fontsize=10)
            ax.set_ylim(0, 1.25)
            ax.grid(axis="y", alpha=0.3)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title(split_label, fontsize=11, fontweight="bold", pad=8)

            # Légende dans la première colonne uniquement
            if col == 0 and row == 0:
                from matplotlib.patches import Patch
                leg_elems = [
                    Patch(fc=_C["aae"], label="AAE group"),
                    Patch(fc=_C["sae"], label="SAE group"),
                    Patch(fc="grey", alpha=0.55, label="ERM (plain)"),
                    Patch(fc="grey", alpha=1.0,  hatch="///", label="IRM (hatched)"),
                ]
                ax.legend(handles=leg_elems, fontsize=8, loc="upper right")

    fig.suptitle(
        "03  —  Fairness gaps by dialect group (AAE vs SAE)\n"
        "Plain bars = ERM  ·  Hatched bars = IRM  ·  "
        "Gap = |rate_AAE − rate_SAE|  ·  Smaller gap → fairer model",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "03_fairness_gaps.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [03] Fairness gaps  →  {out_dir}/03_fairness_gaps.png")


# =============================================================================
# 4. Construction du Test Anti-Corrélé
# =============================================================================

def build_test_ood_anticorr(
    ds: dict,
    excluded: np.ndarray,
    n_anchor: int = 5000,
    p_pos_aae: float = 0.70,
    p_neg_sae: float = 0.70,
    seed: int = 44,
) -> np.ndarray:
    """
    Construit un Test Anti-Corrélé depuis le pool global (ds["all"]).
    Les indices `excluded` (test OOD déjà réservé) sont retirés.

    Pendant l'entraînement : P(Y=0|AAE) = 80/70 % (AAE → négatif).
    Ici on INVERSE :
        P(Y=1|AAE) = p_pos_aae  (défaut 70 % → AAE majoritairement positif)
        P(Y=0|SAE) = p_neg_sae  (défaut 70 % → SAE majoritairement négatif)

    Un modèle ERM shortcut « AAE→négatif » commettra beaucoup d'erreurs ici.
    IRM correctement entraîné devrait mieux résister.

    Returns : indices dans ds["all"].
    """
    rng = np.random.default_rng(seed)
    all_split = ds["all"]

    def _avail(y_val, a_val):
        idx = _get_group_indices(all_split, y_val, a_val)
        return idx[~np.isin(idx, excluded)]

    idx_pos_aae = _avail(1, 1)  # Pos AAE — ressource rare
    idx_neg_aae = _avail(0, 1)
    idx_neg_sae = _avail(0, 0)
    idx_pos_sae = _avail(1, 0)

    # AAE : anchor sur Pos AAE (rarest), dériver Neg AAE
    n_pos_aae = min(n_anchor, len(idx_pos_aae))
    n_neg_aae = int(n_pos_aae * (1 - p_pos_aae) / p_pos_aae)
    if n_neg_aae > len(idx_neg_aae):
        n_neg_aae = len(idx_neg_aae)
        n_pos_aae = int(n_neg_aae * p_pos_aae / (1 - p_pos_aae))

    # SAE : même échelle qu'AAE, anchor sur Pos SAE (minoritaire ici)
    n_pos_sae = min(n_pos_aae, len(idx_pos_sae))
    n_neg_sae = int(n_pos_sae * p_neg_sae / (1 - p_neg_sae))
    if n_neg_sae > len(idx_neg_sae):
        n_neg_sae = len(idx_neg_sae)
        n_pos_sae = int(n_neg_sae * (1 - p_neg_sae) / p_neg_sae)

    sel = np.concatenate([
        rng.choice(idx_pos_aae, size=n_pos_aae, replace=False),
        rng.choice(idx_neg_aae, size=n_neg_aae, replace=False),
        rng.choice(idx_neg_sae, size=n_neg_sae, replace=False),
        rng.choice(idx_pos_sae, size=n_pos_sae, replace=False),
    ])
    rng.shuffle(sel)

    actual_p_pos_aae = n_pos_aae / (n_pos_aae + n_neg_aae)
    actual_p_neg_sae = n_neg_sae / (n_neg_sae + n_pos_sae)
    print(
        f"  [Test Anti-Corr] {len(sel):,} exemples  "
        f"P(pos|AAE)={actual_p_pos_aae:.1%} (cible {p_pos_aae:.0%})  "
        f"P(neg|SAE)={actual_p_neg_sae:.1%} (cible {p_neg_sae:.0%})"
    )
    return sel


# =============================================================================
# PLOT 4 — OOD équilibré vs Test Anti-Corrélé
# =============================================================================

def plot_anticorr_test(results: dict, env_sizes: dict, out_dir: str):
    """
    04_anticorr_test.png
    ────────────────────
    Compare les performances par groupe sur le Test OOD équilibré (25/25/25/25)
    et sur le Test Anti-Corrélé (AAE→positif 70%, SAE→négatif 70%).

    Le test anti-corrélé est le vrai stress test pour les modèles biaisés :
    un modèle ERM qui a appris « AAE→négatif » prédit agressivement négatif
    pour le groupe Pos AAE, et « SAE→positif » pour Neg SAE — les deux groupes
    dominants dans ce test.  IRM, s'il a appris un signal invariant, devrait
    conserver des performances raisonnables.

    Lignes pointillées = worst-group accuracy (signal de robustesse globale).
    """
    group_keys   = ["(Y=0,A=0)", "(Y=0,A=1)", "(Y=1,A=0)", "(Y=1,A=1)"]
    group_labels = ["Neg SAE", "Neg AAE", "Pos SAE", "Pos AAE"]
    group_colors = ["#f0a500", "#5b8dd9", "#e05252", "#4caf7d"]

    splits = [
        ("test_ood",      "Test OOD équilibré\n(25% chaque groupe, sans biais)"),
        ("test_anticorr", "Test Anti-Corrélé\n(AAE→positif 70%, SAE→négatif 70%)"),
    ]

    x = np.arange(len(group_keys))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (split_key, split_label) in zip(axes, splits):
        erm_vals = [results["erm"][split_key]["acc_groups"][k] for k in group_keys]
        irm_vals = [results["irm"][split_key]["acc_groups"][k] for k in group_keys]

        for gi, (gk, gc) in enumerate(zip(group_keys, group_colors)):
            b_erm = ax.bar(
                gi - width / 2, erm_vals[gi], width,
                color=gc, edgecolor="black", alpha=0.6,
            )
            b_irm = ax.bar(
                gi + width / 2, irm_vals[gi], width,
                color=gc, edgecolor="black", alpha=1.0, hatch="///",
            )
            _annotate_bar(ax, b_erm[0], erm_vals[gi], fmt=".2f", dy=0.010, fontsize=8)
            _annotate_bar(ax, b_irm[0], irm_vals[gi], fmt=".2f", dy=0.010, fontsize=8)

            n = env_sizes.get(split_key, {}).get(gk, None)
            if n is not None:
                ax.text(gi, -0.07, f"N={n:,}", ha="center", fontsize=7, color="#555",
                        transform=ax.get_xaxis_transform())

        wga_erm = results["erm"][split_key]["worst_group_acc"]
        wga_irm = results["irm"][split_key]["worst_group_acc"]
        ax.axhline(wga_erm, color=_C["erm"], ls=":", lw=1.8, alpha=0.85,
                   label=f"ERM worst-group = {wga_erm:.3f}")
        ax.axhline(wga_irm, color=_C["irm"], ls=":", lw=1.8, alpha=0.85,
                   label=f"IRM worst-group = {wga_irm:.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_title(split_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Group Accuracy  (higher = better)", fontsize=9)
        ax.set_ylim(0, 1.20)
        ax.grid(axis="y", alpha=0.3)

        from matplotlib.patches import Patch
        group_legend = [Patch(fc=gc, label=gl) for gc, gl in zip(group_colors, group_labels)]
        ax.legend(handles=group_legend + [
            plt.Line2D([0], [0], color="grey", alpha=0.6, lw=8, label="ERM (plain)"),
            plt.Line2D([0], [0], color="grey", alpha=1.0, lw=8, label="IRM (hatched)"),
            plt.Line2D([0], [0], color=_C["erm"], ls=":", lw=1.8,
                       label=f"ERM worst-group = {wga_erm:.3f}"),
            plt.Line2D([0], [0], color=_C["irm"], ls=":", lw=1.8,
                       label=f"IRM worst-group = {wga_irm:.3f}"),
        ], fontsize=7.5, loc="upper right")

    fig.suptitle(
        "04  —  Test OOD équilibré vs Test Anti-Corrélé\n"
        "Plain = ERM  ·  Hatched = IRM  ·  Pointillés = worst-group accuracy  ·  "
        "Le test anti-corrélé inverse le biais → stress test réel pour ERM",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_anticorr_test.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [04] Anti-corr test  →  {out_dir}/04_anticorr_test.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Moji – ERM vs IRM (IRMv1) — Sentiment × Dialecte"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--p_e1", type=float, default=0.80,
        help="Force de corrélation de E1 : P(neg|AAE) = P(pos|SAE) = p_e1 (défaut 0.80)",
    )
    parser.add_argument(
        "--p_e2", type=float, default=0.70,
        help="Force de corrélation de E2 : P(neg|AAE) = P(pos|SAE) = p_e2 (défaut 0.70)",
    )
    parser.add_argument(
        "--sae_ratio", type=float, default=2.0,
        help="Ratio N_SAE / N_AAE dans chaque environnement d'entraînement (défaut 2.0 : "
             "SAE = 2/3, AAE = 1/3 ; 1.0 = équilibre parfait).",
    )

    # DistilBERT
    parser.add_argument("--bert_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--embed_batch", type=int, default=1024,
        help="Embedding batch size. Increase to 512/1024 if GPU VRAM allows.",
    )

    # Modèle / entraînement
    parser.add_argument("--logreg_bn", action="store_true", default=False,
                        help="Ajoute un BatchNorm1d avant la couche linéaire (logreg uniquement). "
                             "Normalise les embeddings → convergence plus rapide et accuracy souvent meilleure.")
    parser.add_argument(
        "--balanced_sampling", action="store_true", default=False,
        help="Équilibre les 4 groupes (Y×A) dans chaque mini-batch par sur-échantillonnage "
             "(WeightedRandomSampler). Compense le déséquilibre Pos AAE / Neg SAE sans "
             "modifier la composition globale des environnements.",
    )
    parser.add_argument(
        "--anticorr_env", action="store_true", default=False,
        help="Ajoute un 3ème environnement d'entraînement anti-corrélé (P(Y=0|AAE)=30%%) "
             "construit à partir des exemples du split train non utilisés par E1/E2. "
             "Fournit à IRM le signal contrastif nécessaire pour identifier le biais dialectal.",
    )
    parser.add_argument("--erm_steps", type=int, default=100_000)
    parser.add_argument("--erm_lr", type=float, default=1e-5)
    parser.add_argument("--irm_steps", type=int, default=100_000)
    parser.add_argument("--irm_lr", type=float, default=1e-5)
    parser.add_argument("--irm_lambda", type=float, default=50.0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=100)


    parser.add_argument(
        "--n_per_group_test", type=int, default=2500,
        help="Nombre d'exemples par groupe (Y×A) dans le Test OOD équilibré (4 groupes × N = total)",
    )

    # Sortie
    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "plots" / "moji"))

    args = parser.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 1 : Chargement
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement du dataset LabHC/moji")
    print("=" * 70)
    ds = load_moji()

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2 : Réservation du test (en premier — no-leak garanti)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Réservation des sets de test (depuis le pool global)")
    print("=" * 70)

    # ── 2a. Test OOD équilibré ── (4 × n_per_group, depuis ds["all"])
    idx_test_ood = build_test_ood(ds, n_per_group=args.n_per_group_test, seed=args.seed)

    # ── 2b. Test Anti-Corrélé ── (depuis ds["all"] hors test OOD)
    idx_test_anticorr = build_test_ood_anticorr(
        ds, excluded=idx_test_ood, seed=args.seed + 1
    )

    # Union des index réservés pour test (utilisée pour exclure de train/val)
    idx_test_all = np.unique(np.concatenate([idx_test_ood, idx_test_anticorr]))
    print(f"  Total réservé pour test : {len(idx_test_all):,} indices")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Val biaisée + Environnements d'entraînement
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Val + Environnements d'entraînement biaisés")
    print("=" * 70)

    # ── 3a. Validation biaisée (intermédiaire 75/25) ──
    idx_val_ind = build_val_biased_from_pool(
        ds, excluded=idx_test_all, seed=args.seed + 2
    )
    idx_reserved = np.unique(np.concatenate([idx_test_all, idx_val_ind]))

    # ── 3b. Environnements E1 / E2 (et optionnellement E3) ──
    env_train_indices = build_train_envs(
        ds,
        excluded=idx_reserved,
        p_e1=args.p_e1,
        p_e2=args.p_e2,
        seed=args.seed,
        include_anticorr_env=args.anticorr_env,
        sae_ratio=args.sae_ratio,
    )
    idx_e1, idx_e2 = env_train_indices[0], env_train_indices[1]
    idx_e3 = env_train_indices[2] if len(env_train_indices) > 2 else None

    all_split = ds["all"]
    Y_all = np.asarray(all_split.Y)
    A_all = np.asarray(all_split.A)

    sae_ratio_lbl = f"SAE/AAE={args.sae_ratio:.1f}×"
    e1_lbl = f"E1 — biais fort    (P(neg|AAE)={args.p_e1:.0%}, P(pos|SAE)={args.p_e1:.0%}, {sae_ratio_lbl})"
    e2_lbl = f"E2 — biais modéré (P(neg|AAE)={args.p_e2:.0%}, P(pos|SAE)={args.p_e2:.0%}, {sae_ratio_lbl})"
    print_split_stats(e1_lbl, idx_e1, Y_all, A_all)
    print_split_stats(e2_lbl, idx_e2, Y_all, A_all)
    if idx_e3 is not None:
        print_split_stats("E3 — anti-corrélé (P(neg|AAE)=30%, P(pos|SAE)=30%)", idx_e3, Y_all, A_all)
    print_split_stats("Val InD — biaisée 75/25", idx_val_ind, Y_all, A_all)
    print_split_stats("Test OOD — équilibré (4 groupes × N)", idx_test_ood, Y_all, A_all)
    print_split_stats("Test Anti-Corrélé", idx_test_anticorr, Y_all, A_all)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3b : Embeddings DistilBERT (pool global — source unique)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3b : Embeddings DistilBERT (gelé, mean pooling)")
    print("=" * 70)

    # Tous les indices font référence à ds["all"]
    def _make_env_from_split(split, indices, emb_all, pos_map, meta=None):
        positions = np.array([pos_map[int(i)] for i in indices])
        emb = emb_all[positions]
        Y = np.asarray(split.Y)[indices].astype(np.float32)
        A = np.asarray(split.A)[indices].astype(np.int64)
        return make_env(emb, Y, identities=A, meta=meta)

    # --- Textes train (E1+E2+E3 sans doublon) ---
    all_idx_list = [idx_e1, idx_e2] + ([idx_e3] if idx_e3 is not None else [])
    all_train_indices = np.unique(np.concatenate(all_idx_list))
    train_pos = {int(orig): pos for pos, orig in enumerate(all_train_indices)}
    texts_train = [all_split.comment_text[int(i)] for i in all_train_indices]

    print(f"  Encodage de {len(texts_train):,} textes (train E1+E2{'+ E3' if idx_e3 is not None else ''}) …")
    emb_train_all = embed_texts(
        texts_train, model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch,
    )

    # --- Textes val ---
    val_pos = {int(orig): pos for pos, orig in enumerate(idx_val_ind)}
    texts_val = [all_split.comment_text[int(i)] for i in idx_val_ind]
    print(f"  Encodage de {len(texts_val):,} textes (val) …")
    emb_val = embed_texts(
        texts_val, model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch,
    )

    # --- Textes test (OOD + anti-corr, dédoublonnés) ---
    all_test_indices = np.unique(np.concatenate([idx_test_ood, idx_test_anticorr]))
    test_pos = {int(orig): pos for pos, orig in enumerate(all_test_indices)}
    texts_test = [all_split.comment_text[int(i)] for i in all_test_indices]
    print(f"  Encodage de {len(texts_test):,} textes (test OOD + anti-corr) …")
    emb_test_all = embed_texts(
        texts_test, model_name=args.bert_model, max_length=args.max_length,
        device=device, batch_size=args.embed_batch,
    )

    # --- Assemblage des Env (tous depuis ds["all"]) ---
    env_e1 = _make_env_from_split(all_split, idx_e1, emb_train_all, train_pos,
                                   meta={"name": "E1_bias"})
    env_e2 = _make_env_from_split(all_split, idx_e2, emb_train_all, train_pos,
                                   meta={"name": "E2_bias"})
    train_envs = [env_e1, env_e2]
    if idx_e3 is not None:
        env_e3 = _make_env_from_split(all_split, idx_e3, emb_train_all, train_pos,
                                       meta={"name": "E3_anticorr"})
        train_envs.append(env_e3)
    env_val_ind = _make_env_from_split(all_split, idx_val_ind, emb_val, val_pos,
                                        meta={"name": "val_ind"})
    env_test_ood = _make_env_from_split(all_split, idx_test_ood, emb_test_all, test_pos,
                                         meta={"name": "test_ood"})
    env_test_anticorr = _make_env_from_split(all_split, idx_test_anticorr, emb_test_all, test_pos,
                                              meta={"name": "test_anticorr"})

    val_envs_for_log = [env_val_ind]

    print(f"\n  Env E1  ({args.p_e1:.0%}/{1-args.p_e1:.0%}) : X {env_e1.X.shape}  Y=1: {env_e1.y.mean():.2%}")
    print(f"  Env E2  ({args.p_e2:.0%}/{1-args.p_e2:.0%}) : X {env_e2.X.shape}  Y=1: {env_e2.y.mean():.2%}")
    if idx_e3 is not None:
        print(f"  Env E3  (anti-corr 30%) : X {env_e3.X.shape}  Y=1: {env_e3.y.mean():.2%}")
    print(f"  Val biaisée 75%%          : X {env_val_ind.X.shape}  Y=1: {env_val_ind.y.mean():.2%}")
    print(f"  Test OOD                  : X {env_test_ood.X.shape}  Y=1: {env_test_ood.y.mean():.2%}")
    print(f"  Test Anti-C               : X {env_test_anticorr.X.shape}  Y=1: {env_test_anticorr.y.mean():.2%}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4a : ERM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4a : ERM Training")
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
        logreg_bn=args.logreg_bn,
        balanced_sampling=args.balanced_sampling,
        dataset_name="moji",
        n_classes=2,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4b : IRM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4b : IRM Training")
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
        logreg_bn=args.logreg_bn,
        balanced_sampling=args.balanced_sampling,
        dataset_name="moji",
        n_classes=2,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5 : Évaluation finale
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5 : Final Evaluation")
    print("=" * 70)

    results = {"erm": {}, "irm": {}}

    print("\n--- ERM ---")
    results["erm"]["val_ind"]        = full_evaluation(model_erm, env_val_ind,       device, "Val InD")
    results["erm"]["test_ood"]       = full_evaluation(model_erm, env_test_ood,      device, "Test OOD")
    results["erm"]["test_anticorr"]  = full_evaluation(model_erm, env_test_anticorr, device, "Test Anti-Corr")

    print("\n--- IRM ---")
    results["irm"]["val_ind"]        = full_evaluation(model_irm, env_val_ind,       device, "Val InD")
    results["irm"]["test_ood"]       = full_evaluation(model_irm, env_test_ood,      device, "Test OOD")
    results["irm"]["test_anticorr"]  = full_evaluation(model_irm, env_test_anticorr, device, "Test Anti-Corr")

    # ─────────────────────────────────────────────────────────────────────
    # Résumé
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<35} {'ERM':>10} {'IRM':>10}")
    print("  " + "-" * 57)

    print("\n  ── Test OOD — Natural HF Distribution ──")
    for metric, label_str in [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro-F1"),
        ("loss",            "Loss"),
        ("fnr_pos_aae",     "FNR  (Y=1, A=1)  AAE positives"),
        ("fpr_neg_sae",     "FPR  (Y=0, A=0)  SAE negatives"),
        ("eod_tpr",         "EOD TPR  |TPR_SAE − TPR_AAE|"),
        ("eod_fpr",         "EOD FPR  |FPR_SAE − FPR_AAE|"),
        ("worst_group_acc", "Worst-Group Accuracy"),
        ("avg_group_acc",   "Avg-Group Accuracy"),
    ]:
        e = results["erm"]["test_ood"][metric]
        r = results["irm"]["test_ood"][metric]
        print(f"  {label_str:<35} {e:>10.4f} {r:>10.4f}")
    print(f"  {'Group breakdown (accuracy)':<35}")
    for gk, glabel in [
        ("(Y=0,A=0)", "    Neg SAE  (Y=0, A=0)"),
        ("(Y=0,A=1)", "    Neg AAE  (Y=0, A=1)"),
        ("(Y=1,A=0)", "    Pos SAE  (Y=1, A=0)"),
        ("(Y=1,A=1)", "    Pos AAE  (Y=1, A=1)"),
    ]:
        e = results["erm"]["test_ood"]["acc_groups"][gk]
        r = results["irm"]["test_ood"]["acc_groups"][gk]
        print(f"  {glabel:<35} {e:>10.4f} {r:>10.4f}")

    print("\n  ── Test Anti-Corrélé — AAE→positif 70%, SAE→négatif 70% ──")
    for metric, label_str in [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro-F1"),
        ("fnr_pos_aae",     "FNR  (Y=1, A=1)  AAE positives"),
        ("fpr_neg_sae",     "FPR  (Y=0, A=0)  SAE negatives"),
        ("eod_tpr",         "EOD TPR  |TPR_SAE − TPR_AAE|"),
        ("worst_group_acc", "Worst-Group Accuracy"),
        ("avg_group_acc",   "Avg-Group Accuracy"),
    ]:
        e = results["erm"]["test_anticorr"][metric]
        r = results["irm"]["test_anticorr"][metric]
        print(f"  {label_str:<35} {e:>10.4f} {r:>10.4f}")
    print(f"  {'Group breakdown (accuracy)':<35}")
    for gk, glabel in [
        ("(Y=0,A=0)", "    Neg SAE  (Y=0, A=0)"),
        ("(Y=0,A=1)", "    Neg AAE  (Y=0, A=1)"),
        ("(Y=1,A=0)", "    Pos SAE  (Y=1, A=0)"),
        ("(Y=1,A=1)", "    Pos AAE  (Y=1, A=1)"),
    ]:
        e = results["erm"]["test_anticorr"]["acc_groups"][gk]
        r = results["irm"]["test_anticorr"]["acc_groups"][gk]
        print(f"  {glabel:<35} {e:>10.4f} {r:>10.4f}")

    print("\n  ── Val InD — 80/20 Biased Distribution ──")
    for metric, label_str in [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro-F1"),
        ("eod_tpr",         "EOD TPR  |TPR_SAE − TPR_AAE|"),
        ("worst_group_acc", "Worst-Group Accuracy"),
        ("avg_group_acc",   "Avg-Group Accuracy"),
    ]:
        e = results["erm"]["val_ind"][metric]
        r = results["irm"]["val_ind"][metric]
        print(f"  {label_str:<35} {e:>10.4f} {r:>10.4f}")
    print(f"  {'Group breakdown (accuracy)':<35}")
    for gk, glabel in [
        ("(Y=0,A=0)", "    Neg SAE  (Y=0, A=0)"),
        ("(Y=0,A=1)", "    Neg AAE  (Y=0, A=1)"),
        ("(Y=1,A=0)", "    Pos SAE  (Y=1, A=0)"),
        ("(Y=1,A=1)", "    Pos AAE  (Y=1, A=1)"),
    ]:
        e = results["erm"]["val_ind"]["acc_groups"][gk]
        r = results["irm"]["val_ind"]["acc_groups"][gk]
        print(f"  {glabel:<35} {e:>10.4f} {r:>10.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # Sauvegarde
    # ─────────────────────────────────────────────────────────────────────
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out_dir}/results.json")

    plot_training_curves(hist_erm, hist_irm, args.out_dir)

    # Compute group sizes for annotation in plots
    group_keys = ["(Y=0,A=0)", "(Y=0,A=1)", "(Y=1,A=0)", "(Y=1,A=1)"]
    env_sizes = {}
    for split_key, idx in [
        ("val_ind",       idx_val_ind),
        ("test_ood",      idx_test_ood),
        ("test_anticorr", idx_test_anticorr),
    ]:
        Y_s = np.asarray(all_split.Y)[idx]
        A_s = np.asarray(all_split.A)[idx]
        env_sizes[split_key] = {
            "(Y=0,A=0)": int(((Y_s == 0) & (A_s == 0)).sum()),
            "(Y=0,A=1)": int(((Y_s == 0) & (A_s == 1)).sum()),
            "(Y=1,A=0)": int(((Y_s == 1) & (A_s == 0)).sum()),
            "(Y=1,A=1)": int(((Y_s == 1) & (A_s == 1)).sum()),
        }

    plot_per_group_f1(results, env_sizes, args.out_dir)
    plot_fairness_gaps(results, args.out_dir)
    plot_anticorr_test(results, env_sizes, args.out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
