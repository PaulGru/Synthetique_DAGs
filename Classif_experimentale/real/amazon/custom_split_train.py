#!/usr/bin/env python
import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

"""
custom_split_train.py — Adversarial Label-Shift Split on Amazon-WILDS
======================================================================

Setup expérimental
------------------
Nous adaptons Amazon-WILDS (Koh et al., 2021) pour étudier la robustesse
face à un **shift de label adversarial** induit par la sévérité des
évaluateurs, plutôt qu'un simple shift d'identité aléatoire.

**Données.**  Nous utilisons l'intégralité du corpus Amazon Reviews
(~4 M exemples, sous-ensemble 30-core déjà appliqué par WILDS), puis
nous reproduisons le data processing de Koh et al. (2021) :
  (i)   suppression des reviews sans texte,
  (ii)  suppression des reviews contenant du HTML,
  (iii) suppression des doublons exacts (même user, produit, note, année),
  (iv)  suppression des textes identiques intra-utilisateur,
  (v)   suppression des textes identiques entre utilisateurs (reviews génériques),
  (vi)  conservation des utilisateurs avec ≥ 150 reviews après nettoyage.

**Partitionnement.**  Contrairement au papier original qui assigne les
reviewers aléatoirement aux splits, nous les partitionnons selon leur
**note moyenne historique**, reflétant leur tendance naturelle à noter
sévèrement ou généreusement.  On calcule μᵤ = moyenne des notes données
par le reviewer u sur l'ensemble du corpus, puis on définit :

  - Train env 0  :  μᵤ ∈ [1.0, 2.5)  – reviewers « durs en affaire »
  - Train env 1  :  μᵤ ∈ [2.5, 3.0)  – reviewers indifférents
  - Train env 2  :  μᵤ ∈ [3.0, 4.0)  – reviewers assez satisfaits
  - Val OOD      :  μᵤ ∈ [4.0, 5.0]  – reviewers très satisfaits (moitié)
  - Test OOD     :  μᵤ ∈ [4.0, 5.0]  – reviewers très satisfaits (autre moitié)

Les splits Val et Test OOD sont constitués par tirage aléatoire 50/50
parmi les utilisateurs à μᵤ ≥ 4.0.  Les reviewers d'entraînement et OOD
sont parfaitement disjoints.

**Shift modélisé.**  En entraînant sur des reviewers ≤ 4★ et en testant
sur des reviewers ≥ 4★, on introduit un **prior shift** sur le label :
la distribution p(y) du test est fortement décalée vers les hautes notes
par rapport au train.  Ce shift est représentatif d'un déploiement en
production où le modèle rencontre des populations de clients
intrinsèquement plus satisfaites.

**Environnements IRM.**  Pour IRM, les 3 buckets d'entraînement définissent
les environnements.  Ce choix garantit que les environnements coïncident
exactement avec le train set, condition nécessaire à la théorie IRM
(Arjovsky et al., 2019).

**Modèle et hyperparamètres.**  Nous fine-tunons DistilBERT-base-uncased
(Sanh et al., 2019) avec un AdamW optimizer, lr = 2×10⁻⁶, weight decay
= 0.01, 3 époques, max_tokens = 512, AMP (fp16) sur GPU.  Ces valeurs
sont dérivées de la grille de recherche de Koh et al. (2021), Table 14.

**Évaluation.**  Accuracy moyenne sur le test OOD (reviewers ≥ 4★ non
vus à l'entraînement), comparée entre ERM (baseline) et IRM.

Usage :
  # ERM (baseline)
  python custom_split_train.py --mode erm --root_dir ./data --download

  # IRM (pénalité λ = 1.0, environnements = severity buckets)
  python custom_split_train.py --mode irm --root_dir ./data
"""

import argparse
import json
import os
import random
import re

import matplotlib
matplotlib.use("Agg")  # pas de display requis (serveur sans X11)
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper

# ═════════════════════════════════════════════════════════════════════
# Hyperparamètres (identiques au papier WILDS, Table 14)
# ═════════════════════════════════════════════════════════════════════
BATCH_SIZE = 8
LR = 1e-5
WEIGHT_DECAY = 0.01
N_EPOCHS = 3
MAX_SEQ_LEN = 512
IRM_LAMBDA = 1.0
SEED = 0
N_CLASSES = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adversarial Label-Shift Split — Amazon-WILDS"
    )
    p.add_argument("--mode", choices=["erm", "irm"], default="erm")
    p.add_argument("--root_dir", type=str, default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--irm_lambda", type=float, default=IRM_LAMBDA)
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--log_dir", type=str, default="./logs_adversarial")
    # ── Accélération ──────────────────────────────────────────────────
    p.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Fraction du split à utiliser (0 < frac ≤ 1.0). "
             "Exemple : --frac 0.1 pour un run rapide de dev.",
    )
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Nombre de micro-batchs avant un pas d'optimiseur. "
             "Batch effectif = batch_size × grad_accum_steps.",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Désactive le mixed-precision (AMP). Utile si pas de GPU.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Nombre de workers pour les DataLoaders.",
    )
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Nombre de batches pré-chargés par worker (DataLoader). "
             "Augmenter si le GPU attend les données (utilisation <80%%).",
    )
    p.add_argument(
        "--eval_frac",
        type=float,
        default=0.1,
        help="Fraction du val set utilisée pour l'évaluation inter-époques "
             "(indépendant de --frac). Exemple : --eval_frac 0.05.",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.02,
        help="Fraction du test OOD utilisée pour l'évaluation finale. "
             "1.0 = test complet (référence). "
             "Exemple : --test_frac 0.1 pour un test rapide en dev.",
    )
    p.add_argument(
        "--val_metric",
        choices=["acc_avg"],
        default="acc_avg",
        help="Métrique utilisée pour sélectionner le meilleur checkpoint.",
    )
    p.add_argument(
        "--no_preprocessing",
        action="store_true",
        help="Désactive le data processing (HTML/dedup/≥150 reviews). "
             "À utiliser uniquement pour des tests rapides.",
    )
    p.add_argument(
        "--min_reviews_per_user",
        type=int,
        default=150,
        help="Nombre minimal de reviews par utilisateur après nettoyage "
             "(défaut : 150, identique à Koh et al., 2021).",
    )
    p.add_argument(
        "--id_val_frac",
        type=float,
        default=0.05,
        help="Fraction des utilisateurs d'entraînement réservée en validation "
             "in-distribution (sélection de checkpoint). Défaut : 0.05 (5%%).",
    )
    p.add_argument(
        "--irm_grouper",
        choices=["user", "severity_bucket"],
        default="severity_bucket",
        help="Définition des environnements pour la pénalité IRM. "
             "'user' : un environnement par reviewer (~25k envs, signal IRM faible). "
             "'severity_bucket' : 5 environnements groupés par note moyenne "
             "arrondie (1★–5★), bien plus d'exemples par env et par batch.",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# 0. Data processing (reproduit Koh et al., 2021 §E.9.4)
# ═════════════════════════════════════════════════════════════════════


def apply_data_processing(dataset, min_reviews_per_user: int = 150):
    """
    Reproduit le data processing de Koh et al. (2021), §E.9.4, sur le
    sous-ensemble 30-core (~4 M exemples) déjà fourni par WILDS.

    Étapes appliquées dans l'ordre :
    1. Suppression des reviews sans texte.
    2. Suppression des reviews contenant du HTML.
    3. Suppression des doublons exacts (même user, produit, note, année).
    4. Suppression des textes identiques intra-utilisateur.
    5. Suppression des textes identiques entre plusieurs utilisateurs
       (« reviews génériques » copiées-collées).
    6. Conservation des seuls utilisateurs ayant ≥ ``min_reviews_per_user``
       reviews après les filtres précédents.

    Note : le 30-core global (chaque reviewer et produit a ≥ 30 reviews
    dans le corpus entier) est déjà appliqué par la librairie WILDS lors
    du chargement du dataset.

    Le masque booléen des exemples conservés est stocké dans
    ``dataset._preprocessing_mask`` pour être appliqué après la création
    du split adversarial.
    """
    texts = dataset._input_array
    n = len(texts)

    user_col    = dataset.metadata_fields.index("user")
    product_col = dataset.metadata_fields.index("product")
    year_col    = dataset.metadata_fields.index("year")
    meta = dataset._metadata_array.numpy()
    users    = meta[:, user_col].astype(np.int64)
    products = meta[:, product_col].astype(np.int64)
    years    = meta[:, year_col].astype(np.int64)
    y_arr    = dataset._y_array.numpy().astype(np.int64)

    html_re = re.compile(r"<[^>]+>")

    valid        = np.ones(n, dtype=bool)
    intra_dup    = np.zeros(n, dtype=bool)
    text_hashes  = np.zeros(n, dtype=np.int64)
    seen_exact   = set()          # (user, product, y, year) tuples
    user_hashes  = {}             # uid → set of text hashes
    hash_to_uids = {}             # text_hash → set of uids

    print("[Preprocessing] Analyse des textes (passe 1/2)...")
    for i in tqdm(range(n), desc="  filtre", mininterval=2.0):
        text = texts[i]
        uid  = int(users[i])

        # 1. Texte vide
        if not text or not text.strip():
            valid[i] = False
            continue

        # 2. HTML
        if html_re.search(text):
            valid[i] = False
            continue

        # 3. Doublon exact (user, product, note, année)
        exact_key = (uid, int(products[i]), int(y_arr[i]), int(years[i]))
        if exact_key in seen_exact:
            valid[i] = False
            continue
        seen_exact.add(exact_key)

        # 4. Doublon intra-utilisateur (même texte pour le même user)
        h = hash(text)
        text_hashes[i] = h
        if uid not in user_hashes:
            user_hashes[uid] = set()
        if h in user_hashes[uid]:
            intra_dup[i] = True
            valid[i] = False
            continue
        user_hashes[uid].add(h)

        # Suivi pour le filtre cross-user (étape 5)
        if h not in hash_to_uids:
            hash_to_uids[h] = set()
        hash_to_uids[h].add(uid)

    # 5. Textes identiques entre utilisateurs (reviews génériques)
    print("[Preprocessing] Détection des textes génériques (passe 2/2)...")
    generic_hashes = {h for h, uids in hash_to_uids.items() if len(uids) > 1}
    for i in tqdm(range(n), desc="  générique", mininterval=2.0):
        if valid[i] and text_hashes[i] in generic_hashes:
            valid[i] = False

    # 6. Utilisateurs avec ≥ min_reviews_per_user reviews
    max_uid = int(users.max())
    user_counts = np.bincount(users[valid], minlength=max_uid + 1)
    valid &= user_counts[users] >= min_reviews_per_user

    n_kept    = int(valid.sum())
    n_removed = n - n_kept
    print(
        f"[Preprocessing] {n:,} → {n_kept:,} exemples conservés "
        f"({n_removed:,} supprimés, {100.0 * n_removed / n:.1f}%)"
    )
    unique_users_kept = int((user_counts[user_counts >= min_reviews_per_user] > 0).sum())
    print(f"[Preprocessing] Utilisateurs conservés (≥{min_reviews_per_user} reviews) : "
          f"{unique_users_kept:,}")

    dataset._preprocessing_mask = valid
    return dataset


# ═════════════════════════════════════════════════════════════════════
# 1. Analyse des métadonnées : note moyenne par utilisateur
# ═════════════════════════════════════════════════════════════════════


def compute_user_mean_ratings(dataset):
    """
    Calcule la note moyenne (échelle 1–5) attribuée par chaque utilisateur
    à travers l'ensemble du dataset.

    Internals WILDS :
      - metadata_fields = ['user', 'product', 'category', 'year', 'y', ...]
      - _y_array contient des entiers 0–4 (mappés depuis les étoiles 1–5).
      - On reconvertit en échelle 1–5 via  rating = y + 1.

    Returns
    -------
    unique_users : np.ndarray, shape (n_users,)
        Identifiants (entiers mappés) des utilisateurs.
    mean_ratings : np.ndarray, shape (n_users,)
        Note moyenne de chaque utilisateur (échelle 1–5).
    """
    user_col = dataset.metadata_fields.index("user")
    y_col = dataset.metadata_fields.index("y")

    users = dataset.metadata_array[:, user_col].numpy()
    # Reconversion 0–4 → 1–5
    ratings = dataset.metadata_array[:, y_col].numpy().astype(np.float64) + 1.0

    # Calcul vectorisé via np.bincount (O(n_examples), rapide sur ~2.5 M lignes)
    max_uid = int(users.max())
    rating_sums = np.bincount(users, weights=ratings, minlength=max_uid + 1)
    user_counts = np.bincount(users, minlength=max_uid + 1)

    valid = user_counts > 0
    mean_all = np.zeros(max_uid + 1, dtype=np.float64)
    mean_all[valid] = rating_sums[valid] / user_counts[valid]

    unique_users = np.where(valid)[0]
    mean_ratings = mean_all[unique_users]

    return unique_users, mean_ratings


def add_severity_bucket_metadata(dataset):
    """
    Ajoute une colonne ``severity_bucket`` à ``_metadata_array``.

    Regroupe les utilisateurs en 5 environnements selon leur note moyenne
    arrondie à l'entier le plus proche (1★ à 5★) :

        bucket 0 : reviewers dont la moyenne est  ≈ 1★  (mean ∈ [1.0, 1.5[
        bucket 1 : reviewers dont la moyenne est  ≈ 2★  (mean ∈ [1.5, 2.5[
        bucket 2 : reviewers dont la moyenne est  ≈ 3★  (mean ∈ [2.5, 3.5[
        bucket 3 : reviewers dont la moyenne est  ≈ 4★  (mean ∈ [3.5, 4.5[
        bucket 4 : reviewers dont la moyenne est  ≈ 5★  (mean ∈ [4.5, 5.0]

    Utilise le ``_severity_bucket_lookup`` précalculé par
    ``create_adversarial_split`` pour éviter de recalculer les moyennes.

    Buckets (seuils continus sur la note moyenne) :

        bucket 0 : mean ∈ [1.0, 2.5)  – reviewers durs en affaire
        bucket 1 : mean ∈ [2.5, 3.0)  – reviewers indifférents
        bucket 2 : mean ∈ [3.0, 4.0)  – reviewers assez satisfaits
        bucket 3 : mean ∈ [4.0, 5.0]  – reviewers très satisfaits (OOD)
    """
    # Réutiliser le lookup déjà calculé par create_adversarial_split
    bucket_lookup = dataset._severity_bucket_lookup

    user_col = dataset.metadata_fields.index("user")
    users = dataset.metadata_array[:, user_col].numpy()
    example_buckets = torch.from_numpy(bucket_lookup[users]).unsqueeze(1)

    # Append de la nouvelle colonne
    dataset._metadata_array = torch.cat(
        [dataset._metadata_array, example_buckets], dim=1
    )
    dataset._metadata_fields.append("severity_bucket")
    if dataset.metadata_map is not None:
        dataset._metadata_map["severity_bucket"] = ["1★–2.5★", "2.5★–3★", "3★–4★", "≥4★"]

    # Statistiques par split et par bucket
    print("[Severity Buckets] Exemples par bucket et par split :")
    bucket_labels = ["1★–2.5★", "2.5★–3★", "3★–4★", "≥4★"]
    for split_name, split_id in [("train", 0), ("id_val", 2), ("val", 1), ("test", 3)]:
        split_mask = dataset._split_array == split_id
        total = int(split_mask.sum())
        if total == 0:
            continue
        buckets_in_split = example_buckets[split_mask, 0]
        parts = []
        for b, lbl in enumerate(bucket_labels):
            n = int((buckets_in_split == b).sum())
            if n > 0:
                parts.append(f"{lbl}:{n:>7d}")
        print(f"  {split_name:>5s} ({total:>8d}) | " + "  ".join(parts))

    return dataset


# ═════════════════════════════════════════════════════════════════════
# 2. Création de l'Adversarial Split basé sur les buckets de sévérité
# ═════════════════════════════════════════════════════════════════════


def create_adversarial_split(dataset, seed: int = 0, id_val_frac: float = 0.05):
    """
    Partitionne les utilisateurs en 3 environnements d'entraînement et
    un split OOD, sur la base de seuils continus sur la note moyenne :

    * **Train env 0** : mean ∈ [1.0, 2.5)  – reviewers durs en affaire
    * **Train env 1** : mean ∈ [2.5, 3.0)  – reviewers indifférents
    * **Train env 2** : mean ∈ [3.0, 4.0)  – reviewers assez satisfaits
    * **Val ID**      : ``id_val_frac`` des users train (sélection checkpoint)
    * **Val OOD**     : moitié des utilisateurs à mean ≥ 4.0
    * **Test OOD**    : autre moitié des utilisateurs à mean ≥ 4.0

    La validation ID (split_id=2, clé 'id_val' dans WILDS) est tirée
    aléatoirement parmi les utilisateurs d'entraînement (< 4★).  Elle
    partage la même distribution que le train et sert uniquement à
    sélectionner le meilleur checkpoint sans accès à la distribution OOD.

    Pourquoi des seuils continus plutôt que l'arrondi
    -------------------------------------------------
    L'arrondi regroupe presque toute la population dans bucket 2 (3★)
    car la distribution des moyennes est concentrée autour de 3–4.
    Les seuils [2.5, 3.0) et [3.0, 4.0) coupent cette masse en deux
    environnements plus proches en taille, ce qui donne à l'IRM un
    signal de pénalité plus équilibré.

    Stockage interne
    ----------------
    Le ``bucket_lookup`` (array user_id → bucket) est attaché au dataset
    sous ``dataset._severity_bucket_lookup`` pour être réutilisé par
    ``add_severity_bucket_metadata`` sans recalcul.

    Le ``split_dict`` Amazon-WILDS (schéma 'user') est :
        {'train': 0, 'val': 1, 'id_val': 2, 'test': 3, 'id_test': 4}
    Les exemples n'appartenant à aucun split reçoivent -1.
    """
    rng = np.random.RandomState(seed)

    unique_users, mean_ratings = compute_user_mean_ratings(dataset)

    # ── Calcul des buckets par seuils continus sur la note moyenne ────
    # bucket 0 : mean ∈ [1.0, 2.5)  – reviewers durs en affaire
    # bucket 1 : mean ∈ [2.5, 3.0)  – reviewers indifférents
    # bucket 2 : mean ∈ [3.0, 4.0)  – reviewers assez satisfaits
    # bucket 3 : mean ∈ [4.0, 5.0]  – reviewers très satisfaits (OOD)
    #
    # Avantage vs. l'arrondi : les utilisateurs à moyenne 1★–2.5★ ne
    # représentent que ~60 personnes mais forment un environnement IRM
    # sémantiquement pur ; le découpage [2.5, 3.0) / [3.0, 4.0) coupe
    # la masse gaussienne centrale en deux parts plus équilibrées.
    user_buckets = np.zeros(len(mean_ratings), dtype=np.int64)
    user_buckets[mean_ratings >= 2.5] = 1
    user_buckets[mean_ratings >= 3.0] = 2
    user_buckets[mean_ratings >= 4.0] = 3

    # Table de lookup  user_id → bucket_id  (réutilisée par add_severity_bucket_metadata)
    max_uid = int(unique_users.max())
    bucket_lookup = np.zeros(max_uid + 1, dtype=np.int64)
    bucket_lookup[unique_users] = user_buckets
    dataset._severity_bucket_lookup = bucket_lookup

    # ── Partition des utilisateurs ────────────────────────────────────
    all_train_users = unique_users[user_buckets <= 2].copy()  # mean < 4.0
    ood_users       = unique_users[user_buckets == 3].copy()  # mean ≥ 4.0

    # Réserver id_val_frac des users train comme validation in-distribution
    rng.shuffle(all_train_users)
    n_id_val    = max(1, int(round(len(all_train_users) * id_val_frac)))
    id_val_users = all_train_users[:n_id_val]
    train_users  = all_train_users[n_id_val:]

    # Mélange aléatoire avant le découpage val OOD / test OOD
    rng.shuffle(ood_users)
    n_val = len(ood_users) // 2
    val_users  = ood_users[:n_val]
    test_users = ood_users[n_val:]

    bucket_def = ["1★–2.5★", "2.5★–3★", "3★–4★"]
    print(f"[Split] Train    : {len(train_users):>6d} utilisateurs (mean < 4★, hors val ID)")
    print(f"[Split] Val ID   : {len(id_val_users):>6d} utilisateurs (mean < 4★, {id_val_frac*100:.0f}% du pool train)")
    print(f"[Split] Val OOD  : {len(val_users):>6d} utilisateurs (mean ≥ 4★)")
    print(f"[Split] Test OOD : {len(test_users):>6d} utilisateurs (mean ≥ 4★)")
    for b, lbl in enumerate(bucket_def):
        n = int((user_buckets == b).sum())
        print(f"  train bucket {b} ({lbl}) : {n:>6d} utilisateurs")

    # ── Construction du split_array ───────────────────────────────────
    user_col = dataset.metadata_fields.index("user")
    users = dataset.metadata_array[:, user_col].numpy()

    split_lookup = np.full(max_uid + 1, -1, dtype=np.int64)
    split_lookup[train_users]  = 0   # train
    split_lookup[id_val_users] = 2   # val ID  (clé 'id_val' dans WILDS)
    split_lookup[val_users]    = 1   # val OOD
    split_lookup[test_users]   = 3   # test OOD

    new_split = split_lookup[users]  # O(n_examples)
    dataset._split_array = new_split

    # ── Recalculer from_source_domain (vectorisé) ─────────────────────
    fd_idx = dataset.metadata_fields.index("from_source_domain")
    source_splits_arr = np.array(list(dataset.source_domain_splits))
    dataset._metadata_array[:, fd_idx] = torch.from_numpy(
        np.isin(new_split, source_splits_arr).astype(np.int64)
    )

    # ── Statistiques des exemples par split ───────────────────────────
    for split_name, split_id in [("train", 0), ("id_val", 2), ("val", 1), ("test", 3)]:
        mask = new_split == split_id
        n = int(mask.sum())
        if n > 0:
            y_vals = dataset._y_array[mask].numpy()
            dist = np.bincount(y_vals, minlength=N_CLASSES)
            print(
                f"  {split_name:>5s} : {n:>8d} exemples | "
                f"mean_label(0-4)={y_vals.mean():.2f}  "
                f"distrib={dist.tolist()}"
            )

    return dataset


# ═════════════════════════════════════════════════════════════════════
# 3. Modèle et tokenisation
# ═════════════════════════════════════════════════════════════════════


def build_model(n_classes: int = N_CLASSES):
    """DistilBERT-base-uncased + tête de classification à ``n_classes`` sorties."""
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=n_classes
    )


def make_collate_fn(tokenizer, max_length: int = MAX_SEQ_LEN):
    """
    Renvoie une fonction collate compatible avec les DataLoaders WILDS.

    Le dataset Amazon-WILDS renvoie des triplets ``(text: str, y, metadata)``.
    La collate tokenise le batch de textes en tenseurs DistilBERT
    (input_ids + attention_mask) et empile les labels et métadonnées.
    """

    def _collate(batch):
        texts, labels, metadata = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded, torch.stack(labels), torch.stack(metadata)

    return _collate


# ═════════════════════════════════════════════════════════════════════
# 4. Pénalité IRM
# ═════════════════════════════════════════════════════════════════════


def _irm_penalty_single(logits, y):
    """
    Pénalité IRM (Arjovsky et al., 2019) pour un seul environnement.

    On introduit un scalaire factice  w = 1  (requires_grad=True) devant
    les logits et on calcule  ‖∇_w L(w · Φ(θ))‖²  évalué en w = 1.

    Cette quantité est nulle si et seulement si les logits sont déjà
    optimaux en w = 1 (c'est-à-dire si le classifieur linéaire est un
    point fixe de la loss de l'environnement).
    """
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, y)
    (grad,) = torch.autograd.grad(loss, scale, create_graph=True)
    return (grad**2).sum()


def compute_irm_loss(logits, y, groups, irm_lambda: float):
    """
    Loss combinée  ERM + λ · IRM_penalty  avec décomposition par groupe.

    Pour chaque sous-groupe (utilisateur) présent dans le mini-batch, on
    calcule la pénalité IRM séparément, puis on prend la moyenne.

    Cas particulier : avec batch_size=8 et des milliers d'utilisateurs,
    chaque groupe contient souvent un seul exemple. On applique alors un
    *fallback* au niveau batch (toute la batch = un seul environnement),
    ce qui revient à la formulation « IRM-v1 » (Arjovsky et al., 2019).

    Returns :  (loss_totale, erm_loss_val, penalty_val)
    """
    erm_loss = F.cross_entropy(logits, y)

    # Tenter la décomposition par groupe (environnement)
    unique_groups = groups.unique()
    penalties = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 2:
            continue
        penalties.append(_irm_penalty_single(logits[mask], y[mask]))

    if len(penalties) > 0:
        penalty = torch.stack(penalties).mean()
    else:
        # Fallback batch-level
        penalty = _irm_penalty_single(logits, y)

    total_loss = erm_loss + irm_lambda * penalty
    return total_loss, erm_loss.item(), penalty.item()


# ═════════════════════════════════════════════════════════════════════
# 5. Boucle d'entraînement
# ═════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    mode: str = "erm",
    irm_lambda: float = 1.0,
    grouper=None,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    scaler: GradScaler = None,
):
    """
    Entraîne un epoch complet (ERM ou IRM).

    Optimisations activées :
    - **Mixed Precision (AMP)** : ``autocast`` réduit la mémoire de moitié
      et double le débit sur GPU compatible (Ampere/Volta/Turing).
    - **Gradient Accumulation** : accumule ``grad_accum_steps`` micro-batchs
      avant chaque pas d'optimiseur, ce qui émule un batch plus grand sans
      augmenter la VRAM requise.
    """
    model.train()
    total_loss = 0.0
    total_penalty = 0.0
    n_batches = 0
    amp_device = device.type if isinstance(device, torch.device) else device
    use_amp = use_amp and (amp_device == "cuda")

    optimizer.zero_grad()

    for step, (encoded, labels, metadata) in enumerate(
        tqdm(loader, desc=f"  [{mode.upper()}]")
    ):
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = labels.to(device)

        with autocast(device_type=amp_device, enabled=use_amp):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

            if mode == "erm":
                loss = F.cross_entropy(logits, labels)
                penalty_val = 0.0
            else:
                groups = grouper.metadata_to_group(metadata).to(device)
                loss, _, penalty_val = compute_irm_loss(
                    logits, labels, groups, irm_lambda
                )

        # Normaliser la loss pour la gradient accumulation
        loss = loss / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum_steps  # logguer la loss réelle
        total_penalty += penalty_val
        n_batches += 1

        # Pas d'optimiseur tous les `grad_accum_steps` micro-batchs
        if (step + 1) % grad_accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / max(n_batches, 1)
    avg_pen = total_penalty / max(n_batches, 1)
    return avg_loss, avg_pen


def evaluate(model, loader, grouper, device, use_amp=True):
    """
    Évaluation des métriques WILDS sans dépendance à ``torch_scatter``.

    Remplace l'appel ``dataset.eval()`` qui requiert torch_scatter via
    ``wilds.common.utils.avg_over_groups``. On recalcule l'accuracy moyenne
    avec ``np.bincount``.

    Accélérations :
    - ``autocast`` (AMP) pour le forward pass d'inférence.
    - ``torch.no_grad()`` sur l'ensemble de la boucle.
    """
    model.eval()
    all_preds, all_labels, all_meta = [], [], []
    amp_device = device.type if isinstance(device, torch.device) else str(device)
    _use_amp = use_amp and (amp_device == "cuda")

    with torch.no_grad():
        for encoded, labels, metadata in tqdm(loader, desc="  [eval]", leave=False):
            with autocast(device_type=amp_device, enabled=_use_amp):
                logits = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                ).logits
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_meta.append(metadata)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    metadata = torch.cat(all_meta)

    # ── Métriques par groupe (utilisateur) via np.bincount ────────────
    # np.bincount remplace torch_scatter et est O(n_examples).
    # group_accs[g] = accuracy de l'utilisateur g sur ses exemples.
    groups = grouper.metadata_to_group(metadata).numpy()
    n_groups = grouper.n_groups

    correct = (y_pred == y_true).astype(np.float64)
    group_correct = np.bincount(groups, weights=correct, minlength=n_groups)
    group_counts = np.bincount(groups, minlength=n_groups)

    active = group_counts > 0
    group_accs = np.where(
        active, group_correct / np.maximum(group_counts, 1.0), np.nan
    )
    active_accs = group_accs[active]

    avg_acc = float(correct.mean())

    results = {"acc_avg": avg_acc}
    results_str = f"  Average acc : {avg_acc:.3f}\n"
    return results, results_str


def plot_training_curves(history: dict, log_dir: str, mode: str) -> None:
    """
    Trace et sauvegarde les courbes d'évaluation époque par époque.

    ``history`` doit contenir les clés :
        'epoch', 'train_loss', 'val_id_acc', 'val_ood_acc'
    Le graphe est sauvegardé dans ``log_dir/curves.png`` et les données
    brutes dans ``log_dir/history.json``.
    """
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training curves — {mode.upper()}", fontsize=13)

    # -- Loss --
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -- Accuracy val ID vs val OOD --
    axes[1].plot(epochs, history["val_id_acc"],  marker="o", label="Val ID acc (checkpoint)")
    axes[1].plot(epochs, history["val_ood_acc"], marker="s", linestyle="--", label="Val OOD acc (monitoring)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Val ID vs Val OOD accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    curves_path = os.path.join(log_dir, "curves.png")
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"  [Courbes] Graphe sauvegardé : {curves_path}")

    history_path = os.path.join(log_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [Courbes] Historique sauvegardé : {history_path}")


# ═════════════════════════════════════════════════════════════════════
# 6. Main
# ═════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # ── 1) Charger le dataset ─────────────────────────────────────────
    print("=" * 65)
    print(" 1. Chargement du dataset Amazon-WILDS")
    print("=" * 65)
    dataset = get_dataset(
        dataset="amazon", root_dir=args.root_dir, download=args.download
    )
    print(f"  Nombre total d'exemples : {len(dataset)}")
    print(f"  Nombre de classes       : {dataset.n_classes}")
    print(f"  Metadata fields         : {dataset.metadata_fields}")

    # ── 1b) Data processing (reproduit Koh et al., 2021 §E.9.4) ──────
    if not args.no_preprocessing:
        print("\n" + "=" * 65)
        print(" 1b. Data Processing (HTML / dedup / ≥150 reviews)")
        print("=" * 65)
        dataset = apply_data_processing(
            dataset, min_reviews_per_user=args.min_reviews_per_user
        )
    else:
        print("  [Preprocessing désactivé via --no_preprocessing]")

    # ── 2) Créer l'adversarial split ──────────────────────────────────
    print("\n" + "=" * 65)
    print(" 2. Création de l'Adversarial Split (Label / Prior Shift)")
    print("=" * 65)
    dataset = create_adversarial_split(dataset, seed=args.seed, id_val_frac=args.id_val_frac)

    # Appliquer le masque de preprocessing APRÈS l'assignation des splits :
    # create_adversarial_split écrit _split_array pour tous les exemples ;
    # on remet à -1 les exemples exclus par le data processing.
    if hasattr(dataset, "_preprocessing_mask"):
        dataset._split_array[~dataset._preprocessing_mask] = -1

    # ── 2b) Ajouter les severity buckets (colonnes de métadonnées) ──
    # Fait ici, après le split, pour que les stats s'affichent
    # sur le train set définitif.
    print()
    dataset = add_severity_bucket_metadata(dataset)

    # ── 3) Modèle et tokenizer ────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" 3. Configuration du modèle DistilBERT")
    print("=" * 65)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = build_model(n_classes=N_CLASSES).to(device)
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_seq_len)
    print(f"  Modèle        : distilbert-base-uncased ({N_CLASSES} classes)")
    print(f"  Max seq length : {args.max_seq_len}")

    # ── 4) Subsets et DataLoaders (API WILDS native) ──────────────────
    #    get_subset() utilise  split_array == split_dict[split]
    #    et renvoie un WILDSSubset. Notre modification de _split_array
    #    est transparente pour cette méthode.
    #    --frac < 1.0 : sous-échantillonnage aléatoire natif WILDS,
    #    utile pour les runs de développement rapides.
    train_subset  = dataset.get_subset("train",  frac=args.frac,      transform=None)
    id_val_subset = dataset.get_subset("id_val", frac=1.0,            transform=None)
    val_eval_subset = dataset.get_subset("val",  frac=args.eval_frac, transform=None)
    test_subset   = dataset.get_subset("test",   frac=args.test_frac, transform=None)

    print(f"\n  Train subset     : {len(train_subset):>9d} exemples  (frac={args.frac})")
    print(f"  Val ID subset    : {len(id_val_subset):>9d} exemples  (sélection checkpoint)")
    print(f"  Val OOD subset   : {len(val_eval_subset):>9d} exemples  (eval_frac={args.eval_frac}, monitoring)")
    print(f"  Test  subset     : {len(test_subset):>9d} exemples  (test_frac={args.test_frac})")

    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        # prefetch_factor : chaque worker maintient ce nombre de batches
        # en mémoire en avance. Permet au GPU de ne jamais attendre
        # la tokenisation CPU. Ignoré si num_workers=0.
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    id_val_loader = DataLoader(
        id_val_subset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_eval_subset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        **loader_kwargs,
    )

    # ── 5) Groupers ───────────────────────────────────────────────────
    eval_grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=["user"])

    # irm_grouper : contrôlé par --irm_grouper.
    # 'severity_bucket' : 5 environnements sémantiquement cohérents,
    #   ~13 exemples/env/batch avec batch=64 → pénalité IRM effective.
    # 'user' : granularité maximale, mais dégénère en ERM en pratique.
    irm_grouper = CombinatorialGrouper(
        dataset=dataset, groupby_fields=[args.irm_grouper]
    )
    print(f"  Eval grouper : user ({eval_grouper.n_groups} groupes)")
    print(f"  IRM  grouper : {args.irm_grouper} ({irm_grouper.n_groups} groupes)")

    # ── 6) Optimiseur, scheduler et AMP ──────────────────────────────
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Le nombre de pas d'optimiseur est réduit par grad_accum_steps
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    use_amp = not args.no_amp
    scaler = GradScaler(device="cuda") if (use_amp and device.type == "cuda") else None
    if use_amp and device.type == "cuda":
        print("  Mixed precision (AMP) : activé  ✓")
    else:
        print("  Mixed precision (AMP) : désactivé")
    if args.grad_accum_steps > 1:
        print(
            f"  Gradient accumulation : {args.grad_accum_steps} micro-batchs "
            f"(batch effectif = {args.batch_size * args.grad_accum_steps})"
        )

    # ── 7) Boucle d'entraînement ──────────────────────────────────────
    print("\n" + "=" * 65)
    print(f" 4. Entraînement  [{args.mode.upper()}]  —  {args.n_epochs} époque(s)")
    if args.mode == "irm":
        print(f"    irm_lambda = {args.irm_lambda}")
    print("=" * 65)

    os.makedirs(args.log_dir, exist_ok=True)
    best_val_metric = -1.0
    history = {"epoch": [], "train_loss": [], "val_id_acc": [], "val_ood_acc": []}

    for epoch in range(1, args.n_epochs + 1):
        print(f"\n── Époque {epoch}/{args.n_epochs} ──")

        avg_loss, avg_pen = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            mode=args.mode,
            irm_lambda=args.irm_lambda,
            grouper=irm_grouper,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=use_amp,
            scaler=scaler,
        )
        pen_str = f"  IRM_penalty={avg_pen:.6f}" if args.mode == "irm" else ""
        print(f"  Loss={avg_loss:.4f}{pen_str}")

        # Évaluation sur val ID (sélection du checkpoint)
        id_val_results, id_val_str = evaluate(
            model, id_val_loader, eval_grouper, device, use_amp=use_amp
        )
        print(f"  Val ID   :\n{id_val_str}")

        # Évaluation sur val OOD (à titre de monitoring uniquement)
        val_results, val_str = evaluate(
            model, val_loader, eval_grouper, device, use_amp=use_amp
        )
        print(f"  Val OOD  (eval_frac={args.eval_frac}, monitoring) :\n{val_str}")

        # Enregistrement dans l'historique
        history["epoch"].append(epoch)
        history["train_loss"].append(float(avg_loss))
        history["val_id_acc"].append(float(id_val_results["acc_avg"]))
        history["val_ood_acc"].append(float(val_results["acc_avg"]))

        # Sauvegarde du meilleur modèle — critère : Val ID acc_avg
        ckpt_path = os.path.join(args.log_dir, "best_model.pt")
        val_metric = id_val_results["acc_avg"]
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"  -> Checkpoint sauvegardé à l'époque {epoch} "
                f"(val_ID_acc={val_metric:.4f})"
            )
        else:
            print(
                f"  -> Pas d'amélioration "
                f"(val_ID_acc={val_metric:.4f} ≤ best={best_val_metric:.4f})"
            )

        # Sauvegarde systématique du modèle de la dernière époque
        # (utile si aucune époque ne bat le score initial, et pour comparaison).
        last_ckpt = os.path.join(args.log_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), last_ckpt)

        # Créer best_model.pt dès la première époque si pas encore créé
        if not os.path.exists(ckpt_path):
            torch.save(model.state_dict(), ckpt_path)

    # Tracer et sauvegarder les courbes d'entraînement
    plot_training_curves(history, args.log_dir, args.mode)

    # ── 8) Évaluation finale sur test OOD ─────────────────────────────
    print("\n" + "=" * 65)
    print(" 5. Évaluation finale — Test OOD")
    print("=" * 65)

    model.load_state_dict(
        torch.load(
            os.path.join(args.log_dir, "best_model.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    test_results, test_str = evaluate(
        model, test_loader, eval_grouper, device, use_amp=use_amp
    )
    print(test_str)

    # Sauvegarder les résultats en JSON
    results_path = os.path.join(args.log_dir, "test_results.json")
    serializable = {}
    for k, v in test_results.items():
        if isinstance(v, (np.floating, float)):
            serializable[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            serializable[k] = int(v)
        elif isinstance(v, torch.Tensor):
            serializable[k] = v.item()
        else:
            serializable[k] = v
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRésultats sauvegardés dans {results_path}")


if __name__ == "__main__":
    main()
