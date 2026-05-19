#!/usr/bin/env python3
"""
run_nli_snli_injection.py
=========================
Compare ERM vs IRM (IRMv1) sur MultiNLI avec injection de SNLI par
environnement.

Chaque environnement d'entraînement correspond à un genre MNLI enrichi d'une
fraction croissante d'exemples SNLI tirés aléatoirement du train SNLI :

    env 0 : MNLI fiction     + snli_ratios[0]  (défaut  5 %)
    env 1 : MNLI government  + snli_ratios[1]  (défaut 10 %)
    env 2 : MNLI slate       + snli_ratios[2]  (défaut 15 %)
    env 3 : MNLI telephone   + snli_ratios[3]  (défaut 20 %)
    env 4 : MNLI travel      + snli_ratios[4]  (défaut 25 %)

Le ratio est défini comme fraction de SNLI dans l'env total :
    n_snli = round(ratio * n_mnli_genre / (1 - ratio))

L'hétérogénéité inter-environnements (ratios SNLI croissants) constitue le
signal d'invariance exploité par IRM : un classifieur invariant ne doit pas
s'appuyer sur des caractéristiques propres à SNLI (légendes d'images) dont la
proportion varie systématiquement d'un environnement à l'autre.

Évaluation :
    - MNLI validation_matched    (ID, par genre)
    - MNLI validation_mismatched (OOD domaine)
    - ANLI R1 / R2 / R3          (OOD adversarial)
    - HANS                       (OOD biais heuristiques NLI, éval. binaire)
    - SNLI-Hard                  (OOD artefacts d'annotation)

Les embeddings BERT sont mis en cache pour ne pas re-calculer entre runs.
Le cache de l'expérience run_multinli_erm_vs_irm.py est réutilisé pour
MNLI et SNLI (même clé de cache).

Usage :
    uv run real/multinli/run_nli_snli_injection.py
    uv run real/multinli/run_nli_snli_injection.py --snli_ratios 0.05 0.10 0.15 0.20 0.25
    uv run real/multinli/run_nli_snli_injection.py --device auto --irm_lambda 500
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
for _p in [str(_ROOT), str(_ROOT / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import hashlib
import json
import os
import urllib.request
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_synth import Env
from models_training import train_erm, train_irm, compute_accuracy
from utils_irm import resolve_device, _predict_logits

# =============================================================================
# 0. Cache d'embeddings
# =============================================================================

# Répertoire de cache partagé avec run_multinli_erm_vs_irm.py pour réutiliser
# les embeddings MNLI et SNLI déjà calculés.
_DEFAULT_CACHE_DIR = str(_Path(__file__).parent / ".embed_cache")


def _cache_key(dataset: str, model_name: str, max_length: int, pooling: str) -> str:
    """Clé déterministe pour (dataset, modèle, params). Compatible avec l'ancien cache."""
    tag = f"{dataset}_{model_name}_{max_length}_{pooling}"
    h = hashlib.md5(tag.encode()).hexdigest()[:10]
    safe = tag.replace("/", "_")
    return f"{safe}_{h}"


def _load_cache(cache_dir: str, key: str):
    path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.isfile(path):
        print(f"  ✓ Cache trouvé : {path}")
        return dict(np.load(path, allow_pickle=True))
    return None


def _save_cache(cache_dir: str, key: str, **arrays):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.npz")
    np.savez(path, **arrays)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  ✓ Cache sauvegardé : {path} ({size_mb:.1f} MB)")


# =============================================================================
# 1. Chargement des datasets
# =============================================================================

TRAIN_GENRES    = ["fiction", "government", "slate", "telephone", "travel"]
ANLI_ROUNDS     = ["r1", "r2", "r3"]
HANS_HEURISTICS = ["lexical_overlap", "subsequence", "constituent"]


def load_multinli() -> dict:
    from datasets import load_dataset
    print("Chargement de MultiNLI depuis Hugging Face …")
    ds = load_dataset("nyu-mll/multi_nli")
    print(f"  train                : {len(ds['train']):,} exemples")
    print(f"  validation_matched   : {len(ds['validation_matched']):,} exemples")
    print(f"  validation_mismatched: {len(ds['validation_mismatched']):,} exemples")
    return ds


def load_snli() -> dict:
    from datasets import load_dataset
    print("Chargement de SNLI depuis Hugging Face …")
    ds = load_dataset("stanfordnlp/snli")
    print(f"  train : {len(ds['train']):,} exemples")
    print(f"  test  : {len(ds['test']):,} exemples")
    return ds


def load_anli() -> dict:
    from datasets import load_dataset
    print("Chargement de ANLI depuis Hugging Face …")
    ds = load_dataset("facebook/anli")
    for r in ANLI_ROUNDS:
        print(f"  test_{r} : {len(ds[f'test_{r}']):,} exemples")
    return ds


def load_hans() -> dict:
    """Charge HANS depuis le dépôt officiel McCoy et al. (GitHub TSV)."""
    URL = (
        "https://raw.githubusercontent.com/tommccoy1/hans/"
        "master/heuristics_evaluation_set.txt"
    )
    print(f"  Téléchargement HANS depuis GitHub …")
    with urllib.request.urlopen(URL) as resp:
        content = resp.read().decode("utf-8")

    lines  = content.strip().split("\n")
    header = lines[0].split("\t")
    col    = {name: idx for idx, name in enumerate(header)}

    label_map = {"entailment": 0, "non-entailment": 1}
    premises, hypotheses, labels, heuristics = [], [], [], []
    for line in lines[1:]:
        parts = line.split("\t")
        lbl = label_map.get(parts[col["gold_label"]], -1)
        if lbl == -1:
            continue
        premises.append(parts[col["sentence1"]])
        hypotheses.append(parts[col["sentence2"]])
        labels.append(lbl)
        heuristics.append(parts[col["heuristic"]])

    print(f"  HANS : {len(labels):,} exemples")
    return {
        "premise": premises, "hypothesis": hypotheses,
        "label": labels, "heuristic": heuristics,
    }


def load_snli_hard(
    cache_dir: str | None = None,
    snli_ds: dict | None = None,
) -> dict:
    """
    Charge SNLI-Hard (Gururangan et al., 2018).

    Stratégie en deux temps :
    1. Téléchargement direct du fichier depuis Stanford NLP.
       URL : https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl
    2. Si le téléchargement échoue, reconstruction à partir du test SNLI via
       un classifieur hypothèse-seule (TF-IDF + LogisticReg entraîné sur le
       train SNLI).  Les exemples 'hard' sont ceux prédits incorrectement par
       ce classifieur — méthodologie exacte de Gururangan et al. (2018).
       snli_ds doit être fourni pour activer ce fallback.

    Avec cache_dir, le résultat est sauvegardé en JSONL pour les runs suivants.
    """
    DIRECT_URL = (
        "https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl"
    )
    CACHE_FNAME = "snli_1.0_test_hard.jsonl"

    # ── 1. Cache local ──────────────────────────────────────────────────
    if cache_dir is not None:
        local_path = os.path.join(cache_dir, CACHE_FNAME)
        if os.path.isfile(local_path):
            print(f"  ✓ SNLI-Hard cache trouvé : {local_path}")
            with open(local_path, "r", encoding="utf-8") as f:
                raw_lines = [json.loads(l) for l in f if l.strip()]
            return _parse_snli_hard_lines(raw_lines)

    # ── 2. Téléchargement direct ─────────────────────────────────────────
    raw_lines = None
    try:
        print(f"  Téléchargement direct SNLI-Hard depuis Stanford NLP …")
        with urllib.request.urlopen(DIRECT_URL, timeout=30) as resp:
            if resp.status == 200:
                content = resp.read().decode("utf-8")
                raw_lines = [json.loads(l) for l in content.splitlines() if l.strip()]
                print(f"  ✓ Téléchargement réussi ({len(raw_lines):,} lignes)")
    except Exception as exc:
        print(f"  ✗ Téléchargement direct échoué ({exc})")

    # ── 3. Fallback : construction hypothèse-seule ────────────────────────
    if raw_lines is None:
        if snli_ds is None:
            raise RuntimeError(
                "Impossible de charger SNLI-Hard : téléchargement échoué et "
                "snli_ds non fourni pour le fallback hypothèse-seule."
            )
        print("  → Fallback : reconstruction SNLI-Hard par classifieur hypothèse-seule …")
        result = _build_snli_hard_hypothesis_only(snli_ds)
        if cache_dir is not None:
            _cache_snli_hard(cache_dir, CACHE_FNAME, result)
        return result

    # ── 4. Sauvegarde en cache ────────────────────────────────────────────
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, CACHE_FNAME)
        with open(local_path, "w", encoding="utf-8") as f:
            for item in raw_lines:
                f.write(json.dumps(item) + "\n")
        print(f"  ✓ SNLI-Hard sauvegardé : {local_path}")

    return _parse_snli_hard_lines(raw_lines)


def _parse_snli_hard_lines(raw_lines: list) -> dict:
    """Convertit les lignes JSONL SNLI-Hard en dict {premise, hypothesis, label}."""
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    premises, hypotheses, labels = [], [], []
    for item in raw_lines:
        # Le fichier officiel utilise 'gold_label' + 'sentence1'/'sentence2'
        lbl = label_map.get(item.get("gold_label", ""), -1)
        if lbl == -1:
            continue
        premises.append(item["sentence1"])
        hypotheses.append(item["sentence2"])
        labels.append(lbl)
    print(f"  SNLI-Hard : {len(labels):,} exemples")
    return {"premise": premises, "hypothesis": hypotheses, "label": labels}


def _build_snli_hard_hypothesis_only(snli_ds: dict, seed: int = 42) -> dict:
    """
    Reconstruit SNLI-Hard par la méthodologie de Gururangan et al. (2018) :
    entraîne un classifieur TF-IDF + LogisticReg sur les hypothèses SNLI train,
    puis garde les exemples du test SNLI prédits *incorrectement*.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Données train (hypothèse seule)
    tr_hyps   = [h for h, l in zip(snli_ds["train"]["hypothesis"],
                                    snli_ds["train"]["label"]) if l != -1]
    tr_labels = [l for l in snli_ds["train"]["label"] if l != -1]

    # Données test
    te_prem  = [p for p, l in zip(snli_ds["test"]["premise"],
                                   snli_ds["test"]["label"]) if l != -1]
    te_hyps  = [h for h, l in zip(snli_ds["test"]["hypothesis"],
                                   snli_ds["test"]["label"]) if l != -1]
    te_labels = [l for l in snli_ds["test"]["label"] if l != -1]

    print(f"    TF-IDF sur {len(tr_hyps):,} hypothèses SNLI train …")
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(tr_hyps)
    X_te = vec.transform(te_hyps)

    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0, n_jobs=-1)
    clf.fit(X_tr, tr_labels)

    preds = clf.predict(X_te)
    hard_mask = preds != np.array(te_labels)
    n_hard = int(hard_mask.sum())
    print(f"    Hypothèse-seule acc={float((~hard_mask).mean()):.3f} → "
          f"{n_hard:,}/{len(te_labels):,} exemples hard ({n_hard/len(te_labels):.1%})")

    return {
        "premise":    [te_prem[i]   for i in range(len(te_prem))   if hard_mask[i]],
        "hypothesis": [te_hyps[i]   for i in range(len(te_hyps))   if hard_mask[i]],
        "label":      [te_labels[i] for i in range(len(te_labels)) if hard_mask[i]],
    }


def _cache_snli_hard(cache_dir: str, fname: str, data: dict):
    """Sauvegarde SNLI-Hard reconstruit en JSONL (format compatible avec le fichier officiel)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, fname)
    label_names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    with open(path, "w", encoding="utf-8") as f:
        for p, h, l in zip(data["premise"], data["hypothesis"], data["label"]):
            f.write(json.dumps({"sentence1": p, "sentence2": h,
                                "gold_label": label_names[l]}) + "\n")
    print(f"  ✓ SNLI-Hard (reconstruit) sauvegardé : {path}")


# =============================================================================
# 2. Embeddings BERT gelé
# =============================================================================

def embed_texts(
    premises: List[str],
    hypotheses: List[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    loaded_model=None,
    loaded_tokenizer=None,
) -> np.ndarray:
    """
    Embeddings BERT gelé pour des paires (premise, hypothesis).

    Le tokenizer reçoit les deux phrases séparément pour générer les
    token_type_ids corrects (segment A / segment B).
    pooling='cls' est recommandé pour les tâches sentence-pair.
    """
    from transformers import AutoTokenizer, AutoModel

    assert len(premises) == len(hypotheses)

    if loaded_tokenizer is not None and loaded_model is not None:
        tokenizer = loaded_tokenizer
        model     = loaded_model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    use_autocast = "cuda" in str(device)

    all_emb = []
    n = len(premises)
    for i in range(0, n, batch_size):
        batch_p = premises[i : i + batch_size]
        batch_h = hypotheses[i : i + batch_size]
        enc = tokenizer(
            batch_p, batch_h,
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        fwd_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            fwd_kwargs["token_type_ids"] = token_type_ids

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_autocast):
                out = model(**fwd_kwargs)

        hidden = out.last_hidden_state
        if pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_emb.append(emb.cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  Embedded {i + len(batch_p):,}/{n:,}")

    return np.concatenate(all_emb, axis=0).astype(np.float32)


# =============================================================================
# 3. Construction des environnements avec injection SNLI
# =============================================================================

def build_envs_with_snli_injection(
    mnli_ds: dict,
    snli_ds: dict,
    snli_ratios: List[float],
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    seed: int = 42,
    loaded_model=None,
    loaded_tokenizer=None,
    cache_dir: str | None = None,
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit les environnements d'entraînement : chaque env = genre MNLI +
    fraction `snli_ratios[i]` d'exemples SNLI tirés aléatoirement.

    Le ratio est défini comme fraction de SNLI dans l'env total :
        n_snli = round(ratio * n_mnli / (1 - ratio))

    Returns
    -------
    train_envs     : List[Env]  — len(TRAIN_GENRES) envs avec métadonnées
                                  (genre, snli_ratio, n_mnli, n_snli)
    val_envs       : List[Env]  — validation_matched, 1 env par genre (ID)
    mismatched_env : Env        — validation_mismatched (OOD domaine)
    """
    assert len(snli_ratios) == len(TRAIN_GENRES), (
        f"--snli_ratios doit contenir exactement {len(TRAIN_GENRES)} valeurs "
        f"(une par genre : {TRAIN_GENRES}), reçu {len(snli_ratios)}"
    )
    rng = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────────────────
    # 3a. Collecte des paires MNLI (train + val_matched + val_mismatched)
    # ──────────────────────────────────────────────────────────────────────
    mnli_premises, mnli_hypotheses = [], []
    mnli_labels, mnli_splits, mnli_genres = [], [], []

    for split_name, hf_key in [
        ("train",          "train"),
        ("val_matched",    "validation_matched"),
        ("val_mismatched", "validation_mismatched"),
    ]:
        split_ds = mnli_ds[hf_key]
        for idx, label in enumerate(split_ds["label"]):
            if label == -1:
                continue
            mnli_premises.append(split_ds["premise"][idx])
            mnli_hypotheses.append(split_ds["hypothesis"][idx])
            mnli_labels.append(label)
            mnli_splits.append(split_name)
            mnli_genres.append(split_ds["genre"][idx])

    mnli_labels = np.array(mnli_labels, dtype=np.int64)
    mnli_splits = np.array(mnli_splits)
    mnli_genres = np.array(mnli_genres)

    # ──────────────────────────────────────────────────────────────────────
    # 3b. Collecte des paires SNLI train
    # ──────────────────────────────────────────────────────────────────────
    snli_premises_all, snli_hypotheses_all, snli_labels_all = [], [], []
    for idx, label in enumerate(snli_ds["train"]["label"]):
        if label == -1:
            continue
        snli_premises_all.append(snli_ds["train"]["premise"][idx])
        snli_hypotheses_all.append(snli_ds["train"]["hypothesis"][idx])
        snli_labels_all.append(label)
    snli_labels_all = np.array(snli_labels_all, dtype=np.int64)
    n_snli_total = len(snli_labels_all)

    # ──────────────────────────────────────────────────────────────────────
    # 3c. Calcul du volume SNLI à injecter par genre (sans remise par genre,
    #     avec remise entre genres pour ne pas épuiser le pool)
    # ──────────────────────────────────────────────────────────────────────
    print("\nVolume SNLI injecté par genre :")
    snli_injection_idx: Dict[str, np.ndarray] = {}
    for genre, ratio in zip(TRAIN_GENRES, snli_ratios):
        n_mnli_genre = int(((mnli_splits == "train") & (mnli_genres == genre)).sum())
        n_snli_needed = int(round(ratio * n_mnli_genre / max(1.0 - ratio, 1e-9)))
        n_snli_needed = min(n_snli_needed, n_snli_total)
        idx_s = rng.choice(n_snli_total, size=n_snli_needed, replace=False)
        snli_injection_idx[genre] = idx_s
        cc = np.bincount(snli_labels_all[idx_s], minlength=3)
        print(f"  {genre:12s} + {n_snli_needed:,} SNLI ({ratio:.0%})  "
              f"(E={cc[0]}, N={cc[1]}, C={cc[2]})")

    # ──────────────────────────────────────────────────────────────────────
    # 3d. Embeddings MNLI (avec cache partagé)
    # ──────────────────────────────────────────────────────────────────────
    mnli_embeddings = None
    use_cache = cache_dir is not None and loaded_model is None

    if use_cache:
        key = _cache_key("mnli_full", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            mnli_embeddings = cached["embeddings"]

    if mnli_embeddings is None:
        print(f"\nEmbedding de {len(mnli_premises):,} paires MNLI "
              f"(train + val_matched + val_mismatched) …")
        mnli_embeddings = embed_texts(
            mnli_premises, mnli_hypotheses,
            model_name=bert_model, max_length=max_length,
            device=device, batch_size=batch_size, pooling=pooling,
            loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=mnli_embeddings)

    # ──────────────────────────────────────────────────────────────────────
    # 3e. Embeddings SNLI (avec cache partagé)
    # ──────────────────────────────────────────────────────────────────────
    snli_embeddings = None

    if use_cache:
        key = _cache_key("snli_train", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            snli_embeddings = cached["embeddings"]

    if snli_embeddings is None:
        print(f"\nEmbedding de {n_snli_total:,} paires SNLI (train) …")
        snli_embeddings = embed_texts(
            snli_premises_all, snli_hypotheses_all,
            model_name=bert_model, max_length=max_length,
            device=device, batch_size=batch_size, pooling=pooling,
            loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=snli_embeddings)

    # ──────────────────────────────────────────────────────────────────────
    # 3f. Construction des envs d'entraînement (MNLI genre + SNLI injecté)
    # ──────────────────────────────────────────────────────────────────────
    print("\nConstruction des environnements d'entraînement :")
    train_envs = []
    for genre, ratio in zip(TRAIN_GENRES, snli_ratios):
        mask_mnli = (mnli_splits == "train") & (mnli_genres == genre)
        X_mnli = mnli_embeddings[mask_mnli]
        y_mnli = mnli_labels[mask_mnli]

        idx_s  = snli_injection_idx[genre]
        X_snli = snli_embeddings[idx_s]
        y_snli = snli_labels_all[idx_s]

        X = np.concatenate([X_mnli, X_snli], axis=0)
        y = np.concatenate([y_mnli, y_snli], axis=0)

        # Mélange déterministe
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        cc = np.bincount(y, minlength=3)
        print(f"  {genre:12s} ({ratio:.0%} SNLI) : {len(y):,} total  "
              f"[MNLI={len(y_mnli):,}, SNLI={len(y_snli):,}]  "
              f"(E={cc[0]}, N={cc[1]}, C={cc[2]})")

        env = Env(
            X=torch.from_numpy(X).float(),
            y=torch.from_numpy(y).long(),
            meta={
                "genre": genre, "snli_ratio": ratio, "split": "train",
                "n_mnli": int(len(y_mnli)), "n_snli": int(len(y_snli)),
            },
        )
        train_envs.append(env)

    # ──────────────────────────────────────────────────────────────────────
    # 3g. Validation ID (val_matched, par genre)
    # ──────────────────────────────────────────────────────────────────────
    print("\nConstruction des environnements val_matched (ID) :")
    val_envs = []
    for genre in TRAIN_GENRES:
        mask = (mnli_splits == "val_matched") & (mnli_genres == genre)
        X = torch.from_numpy(mnli_embeddings[mask]).float()
        y = torch.from_numpy(mnli_labels[mask]).long()
        env = Env(X=X, y=y, meta={"genre": genre, "split": "val_matched"})
        val_envs.append(env)
        cc = np.bincount(mnli_labels[mask], minlength=3)
        print(f"  {genre:12s} : {mask.sum():,}  (E={cc[0]}, N={cc[1]}, C={cc[2]})")

    # ──────────────────────────────────────────────────────────────────────
    # 3h. Val mismatched (OOD domaine)
    # ──────────────────────────────────────────────────────────────────────
    print("\nConstruction de l'env val_mismatched (OOD domaine) :")
    mask_mm = mnli_splits == "val_mismatched"
    X_mm = torch.from_numpy(mnli_embeddings[mask_mm]).float()
    y_mm = torch.from_numpy(mnli_labels[mask_mm]).long()
    mismatched_env = Env(X=X_mm, y=y_mm, meta={"split": "val_mismatched"})
    cc = np.bincount(mnli_labels[mask_mm], minlength=3)
    print(f"  val_mismatched : {mask_mm.sum():,}  (E={cc[0]}, N={cc[1]}, C={cc[2]})")

    return train_envs, val_envs, mismatched_env


# =============================================================================
# 4. Construction des environnements d'évaluation OOD
# =============================================================================

def build_anli_envs(
    anli_ds: dict,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    loaded_model=None,
    loaded_tokenizer=None,
    cache_dir: str | None = None,
) -> Dict[str, Env]:
    """Construit un Env par round ANLI (r1, r2, r3) + un Env combiné ('all')."""
    all_premises, all_hypotheses, all_labels, all_rounds = [], [], [], []

    for r in ANLI_ROUNDS:
        split = anli_ds[f"test_{r}"]
        for idx, label in enumerate(split["label"]):
            if label == -1:
                continue
            all_premises.append(split["premise"][idx])
            all_hypotheses.append(split["hypothesis"][idx])
            all_labels.append(label)
            all_rounds.append(r)

    all_labels = np.array(all_labels, dtype=np.int64)
    all_rounds = np.array(all_rounds)

    embeddings = None
    use_cache = cache_dir is not None and loaded_model is None

    if use_cache:
        key = _cache_key("anli", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            embeddings = cached["embeddings"]

    if embeddings is None:
        print(f"\nEmbedding de {len(all_premises):,} paires ANLI (test R1+R2+R3) …")
        embeddings = embed_texts(
            all_premises, all_hypotheses,
            model_name=bert_model, max_length=max_length,
            device=device, batch_size=batch_size, pooling=pooling,
            loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=embeddings)

    envs: Dict[str, Env] = {}
    print("\nConstruction des Envs ANLI :")
    for r in ANLI_ROUNDS:
        mask = all_rounds == r
        X = torch.from_numpy(embeddings[mask]).float()
        y = torch.from_numpy(all_labels[mask]).long()
        envs[r] = Env(X=X, y=y, meta={"round": r, "split": "anli_test"})
        cc = np.bincount(all_labels[mask], minlength=3)
        print(f"  {r} : {mask.sum():,}  (E={cc[0]}, N={cc[1]}, C={cc[2]})")

    X_all = torch.from_numpy(embeddings).float()
    y_all = torch.from_numpy(all_labels).long()
    envs["all"] = Env(X=X_all, y=y_all, meta={"round": "all", "split": "anli_test"})
    cc = np.bincount(all_labels, minlength=3)
    print(f"  all : {len(all_labels):,}  (E={cc[0]}, N={cc[1]}, C={cc[2]})")

    return envs


def build_hans_env(
    hans: dict,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    loaded_model=None,
    loaded_tokenizer=None,
    cache_dir: str | None = None,
) -> Tuple[Env, Dict[str, Env]]:
    """
    Construit l'env HANS global + un Env par heuristique.

    HANS est un benchmark **binaire** (entailment vs non-entailment).
    Le modèle prédit 3 classes ; l'évaluation projette les classes 1 et 2
    (neutral, contradiction) vers non-entailment via evaluate_hans().

    Returns
    -------
    hans_env  : Env global
    heur_envs : Dict{"lexical_overlap", "subsequence", "constituent"} → Env
    """
    premises   = hans["premise"]
    hypotheses = hans["hypothesis"]
    labels     = np.array(hans["label"], dtype=np.int64)
    heuristics = np.array(hans["heuristic"])

    embeddings = None
    use_cache = cache_dir is not None and loaded_model is None

    if use_cache:
        key = _cache_key("hans", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            embeddings = cached["embeddings"]

    if embeddings is None:
        print(f"\nEmbedding de {len(premises):,} paires HANS …")
        embeddings = embed_texts(
            premises, hypotheses,
            model_name=bert_model, max_length=max_length,
            device=device, batch_size=batch_size, pooling=pooling,
            loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=embeddings)

    X_all = torch.from_numpy(embeddings).float()
    y_all = torch.from_numpy(labels).long()
    hans_env = Env(X=X_all, y=y_all, meta={"split": "hans", "binary": True})
    cc = np.bincount(labels, minlength=2)
    print(f"\nHANS global : {len(labels):,}  "
          f"(entailment={cc[0]}, non-entailment={cc[1]})")

    heur_envs: Dict[str, Env] = {}
    for h in HANS_HEURISTICS:
        mask = heuristics == h
        X_h = torch.from_numpy(embeddings[mask]).float()
        y_h = torch.from_numpy(labels[mask]).long()
        heur_envs[h] = Env(X=X_h, y=y_h,
                           meta={"split": "hans", "heuristic": h, "binary": True})
        cc_h = np.bincount(labels[mask], minlength=2)
        print(f"  {h:22s} : {mask.sum():,}  "
              f"(entailment={cc_h[0]}, non-entailment={cc_h[1]})")

    return hans_env, heur_envs


def build_snli_hard_env(
    snli_hard: dict,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    loaded_model=None,
    loaded_tokenizer=None,
    cache_dir: str | None = None,
) -> Env:
    """Construit l'env SNLI-Hard (test robuste aux artefacts d'annotation)."""
    premises   = snli_hard["premise"]
    hypotheses = snli_hard["hypothesis"]
    labels     = np.array(snli_hard["label"], dtype=np.int64)

    embeddings = None
    use_cache = cache_dir is not None and loaded_model is None

    if use_cache:
        key = _cache_key("snli_hard", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            embeddings = cached["embeddings"]

    if embeddings is None:
        print(f"\nEmbedding de {len(premises):,} paires SNLI-Hard …")
        embeddings = embed_texts(
            premises, hypotheses,
            model_name=bert_model, max_length=max_length,
            device=device, batch_size=batch_size, pooling=pooling,
            loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=embeddings)

    X = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).long()
    cc = np.bincount(labels, minlength=3)
    print(f"\nSNLI-Hard : {len(labels):,}  (E={cc[0]}, N={cc[1]}, C={cc[2]})")
    return Env(X=X, y=y, meta={"split": "snli_hard"})


# =============================================================================
# 5. Évaluation
# =============================================================================

def _eval_full(model: nn.Module, env: Env, device: str) -> float:
    """Accuracy sur l'env complet (sans sous-échantillonnage)."""
    logits = _predict_logits(model, env.X, device=device)
    if logits.dim() == 2 and logits.shape[1] > 1:
        pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
    else:
        pred = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(np.int64).reshape(-1)
    y_true = env.y.cpu().numpy().reshape(-1)
    return float((pred == y_true).mean())


def evaluate_hans(model: nn.Module, env: Env, device: str = "cpu") -> float:
    """
    Accuracy sur HANS avec projection 3 classes → 2 classes.

    Le modèle produit des logits 3-classes (entailment=0, neutral=1,
    contradiction=2). Pour HANS (binaire) : classe 0 reste entailment (0),
    classes 1 et 2 → non-entailment (1).
    """
    logits = _predict_logits(model, env.X, device=device)
    if logits.dim() == 2 and logits.shape[1] == 3:
        pred_3 = logits.argmax(dim=-1).cpu().numpy()
        pred   = (pred_3 != 0).astype(np.int64)        # 0=entail, 1=non-entail
    elif logits.dim() == 2 and logits.shape[1] == 2:
        pred = logits.argmax(dim=-1).cpu().numpy()
    else:
        pred = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(np.int64).reshape(-1)
    y_true = env.y.cpu().numpy().reshape(-1)
    return float((pred == y_true).mean())


# =============================================================================
# 6. Visualisation
# =============================================================================

def _ema(values: list, alpha: float = 0.05) -> list:
    if not values:
        return values
    s = [values[0]]
    for v in values[1:]:
        s.append(alpha * v + (1 - alpha) * s[-1])
    return s


def plot_training_curves(hist_erm: dict, hist_irm: dict, out_dir: str):
    """Courbes d'entraînement ERM vs IRM (loss, train acc, val ID acc, ANLI acc)."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for hist, color, name in [(hist_erm, "C0", "ERM"), (hist_irm, "C1", "IRM")]:
        steps = hist["step"]
        axes[0].plot(steps, hist["loss"], color=color, alpha=0.15, lw=0.8)
        axes[0].plot(steps, _ema(hist["loss"]), color=color, lw=1.8, label=name)
        axes[1].plot(steps, hist["train_acc"], lw=1.5, label=f"{name} train", color=color)
        axes[2].plot(steps, hist["val_acc"],   lw=1.5, label=f"{name} val",   color=color)
        axes[3].plot(steps, hist["test_acc"],  lw=1.5, label=f"{name} ANLI",  color=color)

    for ax, title, ylabel in zip(
        axes,
        ["Training Loss (EMA)", "Train Accuracy", "Val Matched (ID)", "Test ANLI (OOD)"],
        ["Loss", "Accuracy", "Accuracy", "Accuracy"],
    ):
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Courbes d'entraînement → {path}")


def plot_accuracy_comparison(
    results: dict,
    snli_ratios: List[float],
    out_dir: str,
):
    """Bar chart ERM vs IRM sur tous les benchmarks d'évaluation."""
    benchmark_keys = [
        "val_matched", "val_mismatched",
        "anli_r1", "anli_r2", "anli_r3", "anli_all",
        "hans", "hans_lexical_overlap", "hans_subsequence", "hans_constituent",
        "snli_hard",
    ]
    benchmark_labels = [
        "Val\nMatched", "Val\nMismatch",
        "ANLI R1", "ANLI R2", "ANLI R3", "ANLI All",
        "HANS", "HANS\nLex.", "HANS\nSubseq.", "HANS\nConst.",
        "SNLI\nHard",
    ]

    x = np.arange(len(benchmark_keys))
    width = 0.30

    erm_accs = [results["erm"].get(k, 0.0) for k in benchmark_keys]
    irm_accs = [results["irm"].get(k, 0.0) for k in benchmark_keys]

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - width / 2, erm_accs, width, label="ERM",
           color="#e74c3c", edgecolor="black", alpha=0.85)
    ax.bar(x + width / 2, irm_accs, width, label="IRM",
           color="#2ecc71", edgecolor="black", alpha=0.85)

    for i, (e, r) in enumerate(zip(erm_accs, irm_accs)):
        ax.text(i - width / 2, e + 0.004, f"{e:.3f}",
                ha="center", fontsize=7.5, rotation=45)
        ax.text(i + width / 2, r + 0.004, f"{r:.3f}",
                ha="center", fontsize=7.5, rotation=45)

    # Séparateurs entre groupes
    for xv, ls in [(1.5, "--"), (5.5, ":"), (9.5, "-.")]:
        ax.axvline(x=xv, color="gray", ls=ls, alpha=0.5)

    ax.text(0.75, 1.02, "ID", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=9, color="gray")
    ax.text(3.5,  1.02, "ANLI (adv.)", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=9, color="gray")
    ax.text(7.5,  1.02, "HANS (heur.)", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=9, color="gray")
    ax.text(10.0, 1.02, "SNLI-Hard", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=9, color="gray")

    ratios_str = " / ".join([f"{int(r * 100)}%" for r in snli_ratios])
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_labels)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"ERM vs IRM — MNLI genres + injection SNLI [{ratios_str}]\n"
        "Évaluation : Val Matched/Mismatched · ANLI R1-R3 · HANS · SNLI-Hard"
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Bar chart benchmarks → {path}")


def plot_train_env_sizes(
    train_envs: List[Env],
    snli_ratios: List[float],
    out_dir: str,
):
    """Stacked bar : composition MNLI / SNLI de chaque environnement d'entraînement."""
    genres = [e.meta["genre"] for e in train_envs]
    n_mnli = [e.meta["n_mnli"] for e in train_envs]
    n_snli = [e.meta["n_snli"] for e in train_envs]

    x = np.arange(len(genres))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x, n_mnli, label="MNLI", color="#3498db", edgecolor="black", alpha=0.85)
    ax.bar(x, n_snli, bottom=n_mnli, label="SNLI injecté",
           color="#e67e22", edgecolor="black", alpha=0.85)

    for i, (nm, ns, ratio) in enumerate(zip(n_mnli, n_snli, snli_ratios)):
        ax.text(i, nm + ns + 200, f"{ratio:.0%}",
                ha="center", fontsize=11, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"MNLI\n{g}" for g in genres])
    ax.set_ylabel("Nombre d'exemples")
    ax.set_title(
        "Composition des environnements d'entraînement\n"
        "(genre MNLI + fraction SNLI croissante)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "train_env_sizes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Composition des envs → {path}")


def plot_per_genre_val(
    model_erm: nn.Module,
    model_irm: nn.Module,
    val_envs: List[Env],
    device: str,
    out_dir: str,
):
    """Bar chart : accuracy par genre sur val_matched (ID)."""
    genres   = [e.meta["genre"] for e in val_envs]
    erm_accs = [_eval_full(model_erm, e, device) for e in val_envs]
    irm_accs = [_eval_full(model_irm, e, device) for e in val_envs]

    x = np.arange(len(genres))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_e = ax.bar(x - width / 2, erm_accs, width, label="ERM",
                    color="#e74c3c", edgecolor="black", alpha=0.85)
    bars_i = ax.bar(x + width / 2, irm_accs, width, label="IRM",
                    color="#2ecc71", edgecolor="black", alpha=0.85)

    for bar, v in zip(bars_e, erm_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=9)
    for bar, v in zip(bars_i, irm_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy par genre MNLI — val_matched (ID)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_genre_val_matched.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Accuracy par genre → {path}")


def plot_hans_breakdown(
    model_erm: nn.Module,
    model_irm: nn.Module,
    hans_heur: Dict[str, Env],
    device: str,
    out_dir: str,
):
    """Bar chart : accuracy HANS par heuristique."""
    heur_labels = {
        "lexical_overlap": "Lexical\nOverlap",
        "subsequence":     "Subsequence",
        "constituent":     "Constituent",
    }
    ks    = list(HANS_HEURISTICS)
    erm_a = [evaluate_hans(model_erm, hans_heur[h], device) for h in ks]
    irm_a = [evaluate_hans(model_irm, hans_heur[h], device) for h in ks]

    x = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_e = ax.bar(x - width / 2, erm_a, width, label="ERM",
                    color="#e74c3c", edgecolor="black", alpha=0.85)
    bars_i = ax.bar(x + width / 2, irm_a, width, label="IRM",
                    color="#2ecc71", edgecolor="black", alpha=0.85)

    for bar, v in zip(bars_e, erm_a):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10)
    for bar, v in zip(bars_i, irm_a):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([heur_labels[h] for h in ks])
    ax.set_ylabel("Accuracy")
    ax.set_title("HANS — ERM vs IRM par heuristique")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance")
    plt.tight_layout()
    path = os.path.join(out_dir, "hans_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  HANS breakdown → {path}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NLI ERM vs IRM — MNLI genres + injection SNLI par environnement"
    )
    # Modèle / embeddings
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--bert_model",  type=str,   default="bert-base-uncased")
    parser.add_argument("--max_length",  type=int,   default=256,
                        help="Longueur max tokens. 256 recommandé pour MultiNLI.")
    parser.add_argument("--embed_batch", type=int,   default=64,
                        help="Batch size pour l'embedding BERT.")
    parser.add_argument("--pooling",     type=str,   default="cls",
                        choices=["cls", "mean"],
                        help="cls (recommandé pour sentence-pair) ou mean.")

    # Injection SNLI
    parser.add_argument(
        "--snli_ratios", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
        metavar="RATIO",
        help=(
            "Fraction de SNLI dans chaque env d'entraînement "
            "(1 valeur par genre MNLI, dans l'ordre : "
            "fiction, government, slate, telephone, travel). "
            "Exemple : --snli_ratios 0.05 0.10 0.15 0.20 0.25"
        ),
    )

    # Entraînement ERM
    parser.add_argument("--erm_steps",  type=int,   default=25_000)
    parser.add_argument("--erm_lr",     type=float, default=1e-3)

    # Entraînement IRM
    parser.add_argument("--irm_steps",  type=int,   default=25_000)
    parser.add_argument("--irm_lr",     type=float, default=1e-3)
    parser.add_argument("--irm_lambda", type=float, default=500.0,
                        help="Coefficient de la pénalité IRM.")

    # Entraînement commun
    parser.add_argument("--batch",      type=int,   default=512)
    parser.add_argument("--eval_every", type=int,   default=100,
                        help="Fréquence d'évaluation intermédiaire (en steps).")
    parser.add_argument("--use_mlp",    action="store_true", default=True,
                        help="Utiliser SmallMLP (défaut) plutôt que LogisticReg.")
    parser.add_argument("--no_mlp",     dest="use_mlp", action="store_false",
                        help="Forcer LogisticReg.")
    parser.add_argument("--mlp_hidden", type=int,   default=512)

    # Output / cache
    parser.add_argument(
        "--out_dir",   type=str,
        default=str(_Path(__file__).parent / "plots_snli_injection"),
    )
    parser.add_argument(
        "--cache_dir", type=str,
        default=_DEFAULT_CACHE_DIR,
        help=(
            "Répertoire de cache pour les embeddings. Partagé avec "
            "run_multinli_erm_vs_irm.py (MNLI + SNLI réutilisés). "
            "Utiliser --cache_dir '' pour désactiver."
        ),
    )

    args = parser.parse_args()
    device    = resolve_device(args.device)
    cache_dir = args.cache_dir if args.cache_dir else None
    os.makedirs(args.out_dir, exist_ok=True)
    n_classes = 3

    if len(args.snli_ratios) != len(TRAIN_GENRES):
        parser.error(
            f"--snli_ratios requiert exactement {len(TRAIN_GENRES)} valeurs "
            f"(une par genre MNLI : {', '.join(TRAIN_GENRES)}), "
            f"reçu {len(args.snli_ratios)}"
        )

    print("\n" + "=" * 70)
    print("EXPÉRIENCE : NLI ERM vs IRM avec injection SNLI par environnement")
    print("=" * 70)
    print(f"  Genres MNLI   : {', '.join(TRAIN_GENRES)}")
    print(f"  Ratios SNLI   : {[f'{r:.0%}' for r in args.snli_ratios]}")
    print(f"  Device        : {device}")
    print(f"  Seed          : {args.seed}")
    print(f"  Backbone      : {args.bert_model}  pooling={args.pooling}")
    print(f"  ERM steps={args.erm_steps}  lr={args.erm_lr}")
    print(f"  IRM steps={args.irm_steps}  lr={args.irm_lr}  λ={args.irm_lambda}")
    print(f"  Classifieur   : {'SmallMLP' if args.use_mlp else 'LogisticReg'}"
          f"  (hidden={args.mlp_hidden})" if args.use_mlp else "")
    print(f"  Cache         : {cache_dir or '(désactivé)'}")
    print(f"  Output        : {args.out_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 1 : Chargement des datasets
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement de MultiNLI, SNLI, ANLI, HANS et SNLI-Hard")
    print("=" * 70)

    mnli_ds       = load_multinli()
    snli_ds       = load_snli()
    anli_ds       = load_anli()
    print("Chargement de HANS …")
    hans_raw      = load_hans()
    print("Chargement de SNLI-Hard …")
    snli_hard_raw = load_snli_hard(cache_dir=cache_dir, snli_ds=snli_ds)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2 : Injection SNLI et construction des environnements
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Injection SNLI — construction des environnements d'entraînement")
    print("=" * 70)

    train_envs, val_envs, mismatched_env = build_envs_with_snli_injection(
        mnli_ds=mnli_ds,
        snli_ds=snli_ds,
        snli_ratios=args.snli_ratios,
        bert_model=args.bert_model,
        max_length=args.max_length,
        device=device,
        batch_size=args.embed_batch,
        pooling=args.pooling,
        seed=args.seed,
        cache_dir=cache_dir,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Embeddings des benchmarks d'évaluation
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Embeddings des benchmarks d'évaluation OOD")
    print("=" * 70)

    embed_kw = dict(
        bert_model=args.bert_model,
        max_length=args.max_length,
        device=device,
        batch_size=args.embed_batch,
        pooling=args.pooling,
        cache_dir=cache_dir,
    )

    anli_envs           = build_anli_envs(anli_ds, **embed_kw)
    hans_env, hans_heur = build_hans_env(hans_raw, **embed_kw)
    snli_hard_env       = build_snli_hard_env(snli_hard_raw, **embed_kw)

    # test_env pour les courbes de training = ANLI combiné (référence OOD rapide)
    test_env_curves = anli_envs["all"]

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4 : Entraînement ERM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Entraînement ERM")
    print("=" * 70)

    erm_model, erm_hist = train_erm(
        envs=train_envs,
        val_envs=val_envs,
        test_env=test_env_curves,
        steps=args.erm_steps,
        lr=args.erm_lr,
        batch=args.batch,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        dataset_name="multinli_snli_injection",
        n_classes=n_classes,
        use_mlp=args.use_mlp,
        mlp_hidden=args.mlp_hidden,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5 : Entraînement IRM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Entraînement IRM")
    print("=" * 70)

    irm_model, irm_hist = train_irm(
        envs=train_envs,
        val_envs=val_envs,
        test_env=test_env_curves,
        steps=args.irm_steps,
        lr=args.irm_lr,
        batch=args.batch,
        irm_lambda=args.irm_lambda,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        dataset_name="multinli_snli_injection",
        n_classes=n_classes,
        use_mlp=args.use_mlp,
        mlp_hidden=args.mlp_hidden,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 6 : Évaluation finale (toujours sur l'intégralité des sets)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 6 : Évaluation finale")
    print("=" * 70)

    results: Dict[str, Dict[str, float]] = {"erm": {}, "irm": {}}

    for method, model in [("erm", erm_model), ("irm", irm_model)]:

        # Val matched (ID) — moyenne sur les genres
        genre_accs = [_eval_full(model, e, device) for e in val_envs]
        results[method]["val_matched"] = float(np.mean(genre_accs))

        # Val mismatched (OOD domaine)
        results[method]["val_mismatched"] = _eval_full(model, mismatched_env, device)

        # ANLI par round + combiné
        for r in ANLI_ROUNDS:
            results[method][f"anli_{r}"] = _eval_full(model, anli_envs[r], device)
        results[method]["anli_all"] = _eval_full(model, anli_envs["all"], device)

        # HANS global + par heuristique (projection 3→2 classes)
        results[method]["hans"] = evaluate_hans(model, hans_env, device)
        for h in HANS_HEURISTICS:
            results[method][f"hans_{h}"] = evaluate_hans(model, hans_heur[h], device)

        # SNLI-Hard
        results[method]["snli_hard"] = _eval_full(model, snli_hard_env, device)

    # ── Affichage récapitulatif ──────────────────────────────────────────
    header = (
        f"\n  {'':6s}  {'ValID':>6s}  {'Mism.':>6s}  "
        f"{'ANLI R1':>7s}  {'ANLI R2':>7s}  {'ANLI R3':>7s}  {'ANLI All':>8s}  "
        f"{'HANS':>6s}  {'SNLI-H':>6s}"
    )
    print(header)
    print("  " + "─" * (len(header) - 3))
    for method in ["erm", "irm"]:
        r = results[method]
        print(
            f"  {method.upper():6s}  "
            f"{r['val_matched']:6.4f}  {r['val_mismatched']:6.4f}  "
            f"{r['anli_r1']:7.4f}  {r['anli_r2']:7.4f}  {r['anli_r3']:7.4f}  "
            f"{r['anli_all']:8.4f}  {r['hans']:6.4f}  {r['snli_hard']:6.4f}"
        )

    print("\n  Détail HANS par heuristique :")
    for h in HANS_HEURISTICS:
        k = f"hans_{h}"
        print(f"    {h:22s}  ERM={results['erm'][k]:.4f}  IRM={results['irm'][k]:.4f}")

    print("\n  Détail val_matched par genre :")
    for e in val_envs:
        g    = e.meta["genre"]
        a_e  = _eval_full(erm_model, e, device)
        a_i  = _eval_full(irm_model, e, device)
        print(f"    {g:12s}  ERM={a_e:.4f}  IRM={a_i:.4f}")

    # ── Sauvegarde JSON ───────────────────────────────────────────────────
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats JSON → {results_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 7 : Visualisation
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 7 : Visualisation")
    print("=" * 70)

    plot_training_curves(erm_hist, irm_hist, args.out_dir)
    plot_train_env_sizes(train_envs, args.snli_ratios, args.out_dir)
    plot_accuracy_comparison(results, args.snli_ratios, args.out_dir)
    plot_per_genre_val(erm_model, irm_model, val_envs, device, args.out_dir)
    plot_hans_breakdown(erm_model, irm_model, hans_heur, device, args.out_dir)

    print("\n✓ Expérience terminée.")


if __name__ == "__main__":
    main()
