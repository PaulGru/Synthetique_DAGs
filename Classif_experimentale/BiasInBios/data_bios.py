"""
data_bios.py — Chargement et construction des environnements pour Bias in Bios.

Dataset : LabHC/bias_in_bios (HuggingFace) — ~400k biographies, binaire.
Tâche binaire spécifique : Architecte vs Architecte d'intérieur.

Stratégie d'environnements (Inférence par Modèle Paresseux) :
    - On entraîne un modèle très simple (TF-IDF + Régression Logistique) sur le train complet.
    - Ce modèle va se baser sur les stéréotypes de genre (pronoms, etc.) pour prédire la profession.
    - On utilise ses prédictions pour diviser le train en deux :
      * Env 1 (Biais Aligné) : Les exemples où le modèle a eu la bonne réponse avec une FORTE confiance.
      * Env 2 (Biais Désaligné/Neutre) : Les exemples où le modèle s'est trompé OU a eu une FAIBLE confiance.
    - Split "dev"   → 1 environnement de validation (complet, tel quel).
    - Split "test"  → test In-Distribution (complet).
    - Split "test" filtré aux genres minoritaires → test OOD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =============================================================================
# Dataclass Env
# =============================================================================

@dataclass
class Env:
    X: torch.Tensor            # (N, d_bert) embeddings BERT
    y: torch.Tensor            # (N,) labels profession int64
    meta: Dict = field(default_factory=dict)


# =============================================================================
# Mapping professions → indices
# =============================================================================

_ALL_PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker", "interior_designer",
    "journalist", "model", "nurse", "painter", "paralegal",
    "pastor", "personal_trainer", "photographer", "physician", "poet",
    "professor", "psychologist", "rapper", "software_engineer", "surgeon",
    "teacher", "yoga_teacher"
]

# Cas d'étude binaire spécifique
PROFESSIONS: List[str] = ["architect", "interior_designer"]
PROF2IDX: Dict[str, int] = {p: i for i, p in enumerate(PROFESSIONS)}
IDX2PROF: Dict[int, str]  = {i: p for p, i in PROF2IDX.items()}
N_CLASSES: int = len(PROFESSIONS)  # 2

_DATASET_TO_LOCAL: Dict[int, int] = {
    ds_idx: PROF2IDX[name]
    for ds_idx, name in enumerate(_ALL_PROFESSIONS)
    if name in PROF2IDX
}


# =============================================================================
# BERT : tokenisation & embeddings
# =============================================================================

def tokenize_and_embed_with_bert(
    texts: List[str],
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    batch_size: int = 64,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model).to(device)
    model.eval()

    all_embeddings = []
    print(f"    Encodage BERT de {len(texts)} textes (batch={batch_size})...")
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded)
        if pooling == "cls":
            emb = out.last_hidden_state[:, 0, :]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embeddings.append(emb.cpu().float().numpy())
        if (start // batch_size) % 10 == 0:
            print(f"      {start + len(batch)}/{len(texts)}")

    return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# Chargement brut du dataset par split
# =============================================================================

def _load_split(ds, split_name: str) -> Tuple[List[str], List[int], List[int]]:
    texts, labels, genders = [], [], []
    if split_name not in ds:
        return texts, labels, genders
    for example in ds[split_name]:
        bio     = example.get("hard_text", "").strip()
        prof_ds = example.get("profession")
        gender  = example.get("gender")
        if not bio or prof_ds is None or gender is None:
            continue
        local_idx = _DATASET_TO_LOCAL.get(int(prof_ds))
        if local_idx is None:
            continue
        texts.append(bio)
        labels.append(local_idx)
        genders.append(int(gender))
    return texts, labels, genders


def _compute_majority_gender(labels: List[int], genders: List[int]) -> np.ndarray:
    labels_np  = np.array(labels)
    genders_np = np.array(genders)
    majority   = np.zeros(N_CLASSES, dtype=int)
    for k in range(N_CLASSES):
        mask = labels_np == k
        if mask.sum() == 0:
            majority[k] = 0
        else:
            majority[k] = 1 if genders_np[mask].mean() >= 0.5 else 0
    return majority


def _filter_minority_gender(
    texts: List[str],
    labels: List[int],
    genders: List[int],
    majority_gender: np.ndarray,
) -> Tuple[List[str], List[int], List[int]]:
    texts_out, labels_out, genders_out = [], [], []
    for t, l, g in zip(texts, labels, genders):
        if g != majority_gender[l]:
            texts_out.append(t)
            labels_out.append(l)
            genders_out.append(g)
    return texts_out, labels_out, genders_out


# =============================================================================
# Inférence Environnements par Modèle Paresseux
# =============================================================================

def _infer_environments_lazy_model(
    texts: List[str], labels: List[int], genders: List[int], seed: int
) -> Tuple[Tuple[List[str], List[int], List[int]], Tuple[List[str], List[int], List[int]]]:
    """
    Entraîne un modèle TF-IDF + Régression Logistique "paresseux" (peu itérations ou forte reg.)
    pour séparer le dataset en E1 (Biais Aligné) et E2 (Biais Neutre/Désaligné).
    """
    print("\n  [Lazy Model] Étape 1 : Vectorisation TF-IDF (max 5000 features)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(texts)
    
    # Modèle basique, max_iter faible pour ne pas "trop" apprendre les features complexes
    print("  [Lazy Model] Étape 1 : Entraînement de la Régression Logistique...")
    clf = LogisticRegression(C=0.1, max_iter=50, random_state=seed, solver="lbfgs", class_weight="balanced")
    clf.fit(X_tfidf, labels)
    
    print("  [Lazy Model] Étape 2 : Inférence et extraction des scores...")
    probs = clf.predict_proba(X_tfidf)
    preds = clf.predict(X_tfidf)
    
    print("  [Lazy Model] Étape 3 : Construction des environnements (seuil par classe)...")
    labels_np = np.array(labels)
    correct = (preds == labels_np)
    # probabilité conférée à la **vraie** classe
    conf = probs[np.arange(len(labels_np)), labels_np]
    
    mask_e1 = np.zeros(len(labels_np), dtype=bool)
    for k in range(N_CLASSES):
        mask_k = (labels_np == k)
        correct_k = correct & mask_k
        if correct_k.any():
            thresh_k = np.median(conf[correct_k])
            print(f"    -> Seuil de confiance (médiane corrects) pour {IDX2PROF[k]:<20} : {thresh_k:.3f}")
            mask_e1[correct_k & (conf >= thresh_k)] = True
            
    # E1 : correct ET forte confiance (Biais Aligné, stéréotypes forts)
    # E2 : faux OU faible confiance (Biais Désaligné/Neutre, contre-stéréotypes)
    mask_e2 = ~mask_e1
    
    print(f"    -> Taille E1 (Aligné)   : {mask_e1.sum()} exemples")
    print(f"    -> Taille E2 (Désaligné): {mask_e2.sum()} exemples")
    
    e1 = ([texts[i] for i in range(len(texts)) if mask_e1[i]],
          [labels[i] for i in range(len(labels)) if mask_e1[i]],
          [genders[i] for i in range(len(genders)) if mask_e1[i]])
          
    e2 = ([texts[i] for i in range(len(texts)) if mask_e2[i]],
          [labels[i] for i in range(len(labels)) if mask_e2[i]],
          [genders[i] for i in range(len(genders)) if mask_e2[i]])
          
    # Sanity check
    print("\n  [Lazy Model] Étape 4 : Validation (Sanity Check) - Distribution par genre")
    _print_env_breakdown(e1[1], e1[2], "Env 1 (Biais Aligné)")
    _print_env_breakdown(e2[1], e2[2], "Env 2 (Désaligné/Neutre)")
    
    return e1, e2


# =============================================================================
# Affichage
# =============================================================================

def _print_env_breakdown(labels: List[int], genders: List[int], title: str = ""):
    labels_np  = np.array(labels)
    genders_np = np.array(genders)
    prefix = f"    {title} - " if title else "    "
    print(f"{prefix}Répartition par profession :")
    for k in range(N_CLASSES):
        mask = labels_np == k
        n_k  = int(mask.sum())
        if n_k == 0:
            continue
        p_f = float(genders_np[mask].mean())
        print(f"      - {IDX2PROF[k]:<22}: {n_k:<5} ex. "
              f"(F: {p_f*100:5.1f}%, M: {(1-p_f)*100:5.1f}%)")


# =============================================================================
# Encodage BERT → Env
# =============================================================================

def _make_env(
    texts: List[str],
    labels: List[int],
    genders: List[int],
    kind: str,
    bert_model: str,
    max_length: int,
    device: str,
    pooling: str,
    bert_batch_size: int,
) -> Env:
    _print_env_breakdown(labels, genders)
    X = tokenize_and_embed_with_bert(texts, bert_model, max_length, device, pooling, bert_batch_size)
    y = np.array(labels, dtype=np.int64)
    g = np.array(genders, dtype=np.int64)
    return Env(
        X=torch.from_numpy(X),
        y=torch.from_numpy(y),
        meta={"kind": kind, "n_samples": len(texts), "gender_array": g},
    )


# =============================================================================
# Construction des environnements
# =============================================================================

def build_envs_bios(
    n_train_envs: int = 2,  # gardé par compatibilité Makefile mais on en produit 2
    seed: int = 42,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    bert_batch_size: int = 64,
    max_samples: Optional[int] = None,
) -> Tuple[List[Env], Env, Env, Env]:
    """
    Construit les environnements pour le sous-ensemble binaire (Architect vs Interior Designer).
    Train envs inférés dynamiquement avec le Modèle Paresseux (EIIL).
    """
    print("Chargement du dataset Bias in Bios (LabHC/bias_in_bios)...")
    ds = load_dataset("LabHC/bias_in_bios")

    tr_texts, tr_labels, tr_genders = _load_split(ds, "train")
    dv_texts, dv_labels, dv_genders = _load_split(ds, "dev")
    te_texts, te_labels, te_genders = _load_split(ds, "test")

    print(f"  Raw binaire : Train={len(tr_texts)} | Dev={len(dv_texts)} | Test={len(te_texts)}")
    print(f"  Classes binaire : {PROFESSIONS}")

    # Limitation optionnelle (tests rapides)
    if max_samples is not None:
        n_tr = int(max_samples * 0.60)
        n_dv = int(max_samples * 0.20)
        n_te = int(max_samples * 0.20)
        tr_texts, tr_labels, tr_genders = tr_texts[:n_tr], tr_labels[:n_tr], tr_genders[:n_tr]
        dv_texts, dv_labels, dv_genders = dv_texts[:n_dv], dv_labels[:n_dv], dv_genders[:n_dv]
        te_texts, te_labels, te_genders = te_texts[:n_te], te_labels[:n_te], te_genders[:n_te]

    # Shuffle du train
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(tr_texts))
    tr_texts   = [tr_texts[i]   for i in perm]
    tr_labels  = [tr_labels[i]  for i in perm]
    tr_genders = [tr_genders[i] for i in perm]

    # Calcul du genre majoritaire dans le train
    majority_gender = _compute_majority_gender(tr_labels, tr_genders)
    print("\n  Genre majoritaire par profession (depuis le split train) :")
    for k in range(N_CLASSES):
        g_maj = "Femme" if majority_gender[k] == 1 else "Homme"
        labels_np = np.array(tr_labels)
        genders_np = np.array(tr_genders)
        mask = labels_np == k
        n_k  = int(mask.sum())
        p_f  = float(genders_np[mask].mean()) if n_k > 0 else 0.0
        print(f"    {IDX2PROF[k]:<22}: {g_maj} (P(F)={p_f:.2f}, n={n_k})")

    # ── Test OOD : filtrage aux genres minoritaires ───────────────────────────
    ood_texts, ood_labels, ood_genders = _filter_minority_gender(
        te_texts, te_labels, te_genders, majority_gender
    )
    print(f"\n  Test OOD (test, genres minoritaires) : {len(ood_texts)} / {len(te_texts)} exemples conservés")

    # ── Inférence d'environnements d'entraînement avec le Modèle Paresseux ────
    (e1_t, e1_l, e1_g), (e2_t, e2_l, e2_g) = _infer_environments_lazy_model(
        tr_texts, tr_labels, tr_genders, seed=seed
    )

    print(f"\n=== Train Env 1 (Biais Aligné, {len(e1_t)} exemples) ===")
    env1 = _make_env(e1_t, e1_l, e1_g, kind="train_env1_aligned",
                     bert_model=bert_model, max_length=max_length,
                     device=device, pooling=pooling, bert_batch_size=bert_batch_size)
                     
    print(f"\n=== Train Env 2 (Désaligné/Neutre, {len(e2_t)} exemples) ===")
    env2 = _make_env(e2_t, e2_l, e2_g, kind="train_env2_misaligned",
                     bert_model=bert_model, max_length=max_length,
                     device=device, pooling=pooling, bert_batch_size=bert_batch_size)

    # ── Validation : split dev complet ────────────────────────────────────────
    print(f"\n=== Validation ({len(dv_texts)} exemples — split dev) ===")
    val_env = _make_env(
        dv_texts, dv_labels, dv_genders,
        kind="val", bert_model=bert_model, max_length=max_length,
        device=device, pooling=pooling, bert_batch_size=bert_batch_size,
    )

    # ── Test ID : split test complet ──────────────────────────────────────────
    print(f"\n=== Test ID ({len(te_texts)} exemples — split test) ===")
    test_id_env = _make_env(
        te_texts, te_labels, te_genders,
        kind="test_id", bert_model=bert_model, max_length=max_length,
        device=device, pooling=pooling, bert_batch_size=bert_batch_size,
    )

    # ── Test OOD : split test filtré ──────────────────────────────────────────
    print(f"\n=== Test OOD ({len(ood_texts)}/{len(te_texts)} ex. — split test, genres minoritaires) ===")
    test_ood_env = _make_env(
        ood_texts, ood_labels, ood_genders,
        kind="test_ood", bert_model=bert_model, max_length=max_length,
        device=device, pooling=pooling, bert_batch_size=bert_batch_size,
    )

    print(f"\n✅ Environnements créés !")
    print(f"   Train E1(Aligné)   : {env1.X.shape[0]}")
    print(f"   Train E2(Désaligné): {env2.X.shape[0]}")
    print(f"   Val     : {val_env.X.shape[0]}")
    print(f"   Test ID : {test_id_env.X.shape[0]}")
    print(f"   Test OOD: {test_ood_env.X.shape[0]} ({len(ood_texts)*100//len(te_texts)}% du test)")

    return [env1, env2], val_env, test_id_env, test_ood_env
