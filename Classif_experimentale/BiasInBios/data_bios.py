"""
data_bios.py — Chargement et construction des environnements pour Bias in Bios.

Dataset : LabHC/bias_in_bios (HuggingFace) — ~400k biographies, 28 professions, genre binaire.

Structure causale :
    G (genre)      ──→ X_texte  (pronoms, tournures genrées)
    Y (profession) ──→ X_texte  (contenu professionnel)
    G ←── corrélation sociétale P(G,Y) ──→ Y

Mécanisme des environnements :
    On contrôle la force de corrélation genre-profession via `rho_e ∈ [0, 1]`.
    rho_e ≈ 1.0  → corrélation amplifiée dans le sens naturel  (train)
    rho_e ≈ 0.0  → corrélation inversée                        (test OOD)

Le genre n'est JAMAIS fourni comme feature au modèle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Dataclass Env (analogue à data_synth.Env)
# =============================================================================

@dataclass
class Env:
    X: torch.Tensor            # (N, d_bert) embeddings BERT — features du modèle
    y: torch.Tensor            # (N,) labels profession int64 (0..27)
    meta: Dict = field(default_factory=dict)
    # meta contient notamment : rho, kind, n_samples, gender_array (np.ndarray)


# =============================================================================
# Mapping professions → indices
# =============================================================================

PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker", "interior_designer",
    "journalist", "landscape_architect", "magician", "model", "nurse",
    "painter", "paralegal", "pastor", "personal_trainer", "photographer",
    "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon",
]
PROF2IDX: Dict[str, int] = {p: i for i, p in enumerate(PROFESSIONS)}
IDX2PROF: Dict[int, str] = {i: p for p, i in PROF2IDX.items()}
N_CLASSES = len(PROFESSIONS)  # 28


# =============================================================================
# Chargement du dataset
# =============================================================================

def load_bias_in_bios(
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Charge le dataset Bias in Bios depuis HuggingFace.

    Champs réels du dataset LabHC/bias_in_bios :
        - hard_text  : str  — biographie brute
        - profession : int64 (0..27) — indice de profession
        - gender     : int64 (0=male, 1=female)

    Returns
    -------
    texts   : List[str]   — biographies brutes
    labels  : List[int]   — indice profession (0..27)
    genders : List[int]   — 0=male, 1=female
    """
    print("Chargement du dataset Bias in Bios (LabHC/bias_in_bios)...")
    ds = load_dataset("LabHC/bias_in_bios")

    texts, labels, genders = [], [], []
    for split_name in ["train", "dev", "test"]:
        if split_name not in ds:
            continue
        for example in ds[split_name]:
            bio = example.get("hard_text", "").strip()
            prof = example.get("profession")
            gender = example.get("gender")
            # Filtrer les exemples incomplets
            if not bio or prof is None or gender is None:
                continue
            # Filtrer les professions hors range
            if not (0 <= int(prof) < N_CLASSES):
                continue
            texts.append(bio)
            labels.append(int(prof))
            genders.append(int(gender))

    # Shuffle reproductible
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    genders = [genders[i] for i in idx]

    if max_samples is not None:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        genders = genders[:max_samples]

    print(f"Dataset chargé : {len(texts)} exemples, {N_CLASSES} classes")
    _print_gender_stats(labels, genders)
    return texts, labels, genders


def _print_gender_stats(labels: List[int], genders: List[int]):
    """Affiche P(F | Y=k) pour les 5 professions les plus déséquilibrées."""
    labels_np = np.array(labels)
    genders_np = np.array(genders)
    stats = []
    for k in range(N_CLASSES):
        mask = labels_np == k
        if mask.sum() == 0:
            continue
        p_f = genders_np[mask].mean()
        stats.append((IDX2PROF[k], mask.sum(), p_f))
    stats.sort(key=lambda x: x[2])
    print("  Professions les plus masculines :")
    for prof, n, pf in stats[:3]:
        print(f"    {prof}: n={n}, P(F)={pf:.2f}")
    print("  Professions les plus féminines :")
    for prof, n, pf in stats[-3:]:
        print(f"    {prof}: n={n}, P(F)={pf:.2f}")


# =============================================================================
# Embeddings BERT
# =============================================================================

def tokenize_and_embed_with_bert(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Calcule les embeddings BERT (CLS ou mean pooling) pour une liste de textes.

    Returns
    -------
    np.ndarray of shape (N, 768)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    all_embeddings = []
    n = len(texts)
    print(f"  Encodage BERT de {n} textes (batch={batch_size})...")

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = texts[start: start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state  # (B, L, 768)
            if pooling == "cls":
                emb = hidden[:, 0, :]
            else:  # mean
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (hidden * mask).sum(1) / mask.sum(1)
            all_embeddings.append(emb.cpu().numpy())
            if (start // batch_size) % 10 == 0:
                print(f"    {start + len(batch)}/{n}")

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


# =============================================================================
# Sous-échantillonnage contrôlé par rho
# =============================================================================

def _subsample_by_rho(
    texts: List[str],
    labels: List[int],
    genders: List[int],
    rho: float,
    rng: np.random.Generator,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Sous-échantillonne le dataset pour contrôler la force de corrélation genre-profession.

    rho = 1.0 : corrélation naturelle amplifiée au maximum
                 (surreprésentation des paires "typiques" : homme dans métiers masculins,
                  femme dans métiers féminins)
    rho = 0.0 : corrélation entièrement inversée
                 (surreprésentation des paires "atypiques")
    rho = 0.5 : distribution équilibrée (genre indépendant du label)

    Implémentation par pondération softmax inversée :
    Pour chaque exemple (g, k), calculer P_naturelle(G=g | Y=k) depuis les données,
    puis calculer un poids w = rho * P_nat + (1 - rho) * (1 - P_nat).
    On tire des exemples proportionnellement à ces poids.
    """
    labels_np = np.array(labels)
    genders_np = np.array(genders)
    n_total = len(texts)

    # 1. Calculer P_naturelle(G=F | Y=k) pour chaque profession k
    p_female_given_k = np.zeros(N_CLASSES)
    for k in range(N_CLASSES):
        mask = labels_np == k
        if mask.sum() == 0:
            p_female_given_k[k] = 0.5
        else:
            p_female_given_k[k] = genders_np[mask].mean()

    # 2. Calculer le poids de chaque exemple
    weights = np.zeros(n_total)
    for i in range(n_total):
        k = labels[i]
        g = genders[i]
        p_nat = p_female_given_k[k] if g == 1 else (1 - p_female_given_k[k])
        # Interpolation entre corrélation naturelle (rho=1) et inversée (rho=0)
        weights[i] = rho * p_nat + (1 - rho) * (1 - p_nat)
        weights[i] = max(weights[i], 1e-6)  # éviter les poids nuls

    # 3. Normaliser et sous-échantillonner
    # Double normalisation pour éviter les erreurs de précision flottante
    weights = weights / weights.sum()
    weights = weights / weights.sum()  # 2ème normalisation : garantit sum=1 exact
    # On garde au plus 50% du pool pour préserver la diversité
    n_sample = min(n_total // 2, n_total)
    chosen = rng.choice(n_total, size=n_sample, replace=False, p=weights)

    texts_out = [texts[i] for i in chosen]
    labels_out = [labels[i] for i in chosen]
    genders_out = [genders[i] for i in chosen]

    # Stats
    labels_arr = np.array(labels_out)
    genders_arr = np.array(genders_out)
    print(f"  Sous-échantillonné : {n_sample} exemples (rho={rho:.2f})")
    # Afficher quelques stats pour vérifier
    for k in [PROF2IDX.get("physician", 0), PROF2IDX.get("nurse", 0),
              PROF2IDX.get("surgeon", 0)]:
        mask = labels_arr == k
        if mask.sum() > 0:
            pf = genders_arr[mask].mean()
            print(f"    P(F | {IDX2PROF[k]}) dans cet env : {pf:.2f}")

    return texts_out, labels_out, genders_out


# =============================================================================
# Construction des environnements
# =============================================================================

def build_envs_bios(
    train_rho: List[float],
    test_rho: float,
    seed: int = 42,
    val_frac: float = 0.1,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    bert_batch_size: int = 64,
    max_samples: Optional[int] = None,
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit les environnements train/val/test pour Bias in Bios.

    Parameters
    ----------
    train_rho : List[float]
        Force de corrélation genre-profession pour chaque env d'entraînement.
        rho=1.0 → corrélation naturelle amplifiée (ERM exploite le genre)
        rho=0.0 → corrélation inversée
    test_rho : float
        Force de corrélation pour le test OOD.
        Doit être distinctement différent de train_rho pour créer le shift.
    seed : int
        Graine aléatoire.
    val_frac : float
        Fraction de validation par env.
    bert_model : str
        Modèle BERT à utiliser pour les embeddings.
    max_length : int
        Longueur max de séquence BERT.
    device : str
        Device PyTorch pour BERT.
    pooling : str
        "mean" ou "cls".
    bert_batch_size : int
        Taille de batch pour BERT.
    max_samples : Optional[int]
        Limite le nombre total d'exemples chargés (utile pour les tests rapides).

    Returns
    -------
    train_envs : List[Env]
    val_envs   : List[Env]
    test_env   : Env
    """
    # 1. Charger le pool complet
    all_texts, all_labels, all_genders = load_bias_in_bios(
        seed=seed, max_samples=max_samples
    )
    n_total = len(all_texts)

    rng = np.random.default_rng(seed)

    # 2. Split global : 80% train pool / 10% val pool / 10% test pool
    idx = rng.permutation(n_total)
    n_test = int(n_total * 0.10)
    n_val = int(n_total * 0.10)

    test_pool_idx = idx[:n_test]
    val_pool_idx = idx[n_test: n_test + n_val]
    train_pool_idx = idx[n_test + n_val:]

    def _pool(indices):
        t = [all_texts[i] for i in indices]
        l = [all_labels[i] for i in indices]
        g = [all_genders[i] for i in indices]
        return t, l, g

    train_pool = _pool(train_pool_idx)
    val_pool = _pool(val_pool_idx)
    test_pool = _pool(test_pool_idx)

    print(f"\nSplit global : Train={len(train_pool[0])} | Val={len(val_pool[0])} | Test={len(test_pool[0])}")

    # 3. Helper pour construire un Env à partir d'un pool + rho
    def _make_env(pool, rho, kind, env_seed):
        texts_p, labels_p, genders_p = pool
        rng_e = np.random.default_rng(env_seed)
        texts_s, labels_s, genders_s = _subsample_by_rho(
            texts_p, labels_p, genders_p, rho=rho, rng=rng_e
        )
        print(f"  Encodage BERT pour env '{kind}' ({len(texts_s)} exemples)...")
        X = tokenize_and_embed_with_bert(
            texts_s, bert_model, max_length, device, pooling, bert_batch_size
        )
        y = np.array(labels_s, dtype=np.int64)
        g = np.array(genders_s, dtype=np.int64)
        return Env(
            X=torch.from_numpy(X),
            y=torch.from_numpy(y),
            meta={
                "rho": rho,
                "kind": kind,
                "n_samples": len(texts_s),
                "gender_array": g,  # stocké en méta, pas dans X
            },
        )

    # 4. Environnements d'entraînement
    train_envs, val_envs = [], []
    n_envs = len(train_rho)

    # Diviser le pool train en sous-pools disjoints (un par env)
    t_texts, t_labels, t_genders = train_pool
    per_env = len(t_texts) // n_envs

    for i, rho in enumerate(train_rho):
        print(f"\n=== Train Env {i} (rho={rho}) ===")
        start = i * per_env
        end = (i + 1) * per_env if i < n_envs - 1 else len(t_texts)
        sub_pool = (t_texts[start:end], t_labels[start:end], t_genders[start:end])
        env = _make_env(sub_pool, rho, kind=f"train_env{i}", env_seed=seed + i)
        train_envs.append(env)

        print(f"=== Val Env {i} (rho={rho}) ===")
        # Le val utilise le même rho mais depuis le val pool commun
        val_env = _make_env(val_pool, rho, kind=f"val_env{i}", env_seed=seed + 1000 + i)
        val_envs.append(val_env)

    # 5. Environnement de test OOD
    print(f"\n=== Test OOD (rho={test_rho}) ===")
    test_env = _make_env(test_pool, test_rho, kind="test_ood", env_seed=seed + 9999)

    print(f"\n✅ Environnements Bias in Bios créés !")
    print(f"   Train : {[e.X.shape[0] for e in train_envs]}")
    print(f"   Val   : {[e.X.shape[0] for e in val_envs]}")
    print(f"   Test  : {test_env.X.shape[0]}")

    return train_envs, val_envs, test_env
