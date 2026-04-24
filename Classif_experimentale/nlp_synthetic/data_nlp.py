from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

# Générateurs d'environnements NLP (SMS Spam) pour expériences IRM
# Adaptation du DAG semi anti-causal : Text → Y → Z → X_y (spurious token embedding)

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import re
import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from data_synth import Env  # Réutiliser la même structure Env


# =============================================================================
# Configuration des tokens trompeurs
# =============================================================================

def define_spurious_tokens() -> Dict[str, str]:
    """
    Définit les tokens textuels trompeurs pour SMS Spam.

    Choix : "sky" (ham) et "fire" (spam).
    - Tokens BERT uniques (un seul token WordPiece chacun).
    - Pas de connotation directe spam/ham : "sky" n'est pas associé aux
      conversations normales, "fire" n'apparaît pas dans les vocabulaires
      typiques de spam promotionnel (FREE, WIN, PRIZE...).
    - Injectés en préfixe : le texte original reste intact, BERT peut
      toujours encoder le contenu SMS normalement.
    """
    return {
        "spam_correlated": "fire",
        "ham_correlated":  "sky",
    }


# =============================================================================
# =============================================================================
# Chargement du dataset SMS Spam
# =============================================================================

def load_sms_spam_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge le dataset SMS Spam depuis Hugging Face.
    
    Dataset: mshenoda/spam-messages (59k messages)
    
    Parameters
    ----------
    seed : int
        Graine pour le shuffle du dataset.
    
    Returns
    -------
    texts : List[str]
        Liste des SMS (textes bruts).
    labels : List[int]
        Labels binaires (0=ham, 1=spam).
    """
    from datasets import concatenate_datasets
    
    # Charger le nouveau dataset (59k messages)
    dataset = load_dataset("mshenoda/spam-messages")
    
    # Combiner les 3 splits (train + validation + test)
    all_data = concatenate_datasets([
        dataset['train'],
        dataset['validation'],
        dataset['test']
    ])
    
    # Shuffle avec seed fixe
    all_data = all_data.shuffle(seed=seed)
    
    # Extraire textes et convertir labels string → int
    texts = all_data['text']
    labels = [1 if label == 'spam' else 0 for label in all_data['label']]
    
    return texts, labels


# =============================================================================
# Injection de tokens trompeurs (semi anti-causal)
# =============================================================================

# Mots de haute fréquence, sémantiquement vides, présents dans les trois datasets.
# Le token spurieux REMPLACE chacune de leurs occurrences dans le texte.
# Résultat : le signal trompeur est distribué à travers tout le texte (plusieurs
# positions) plutôt que concentré en un seul préfixe, ce qui le rend beaucoup
# plus difficile à ignorer pour ERM.
NEUTRAL_WORDS: List[str] = [
    "the", "a", "an", "of", "in", "to", "is", "it",
    "and", "or", "at", "on", "for", "with", "as", "by",
]


def _prepend_token_to_neutral_words(text: str, neutral_words: List[str], token: str) -> Optional[str]:
    """
    Insère `token` devant chaque occurrence de mot neutre dans `text`.

    Ex : token="fire", text="I went to the store and it was great"
         → "I went to fire the store fire and fire it was great"

    Les mots neutres sont conservés (non remplacés). Le signal spurieux est
    ainsi distribué à travers tout le texte, ce qui le rend beaucoup plus
    difficile à ignorer pour ERM que le simple préfixe de phrase.
    Utilise des word boundaries (\\b) pour ne pas toucher aux sous-chaînes
    (ex. : "for" dans "format" n'est pas affecté).
    Retourne None si aucun mot neutre n'est trouvé (fallback préfixe).
    """
    result = text
    found_any = False
    for word in neutral_words:
        pattern = re.compile(r'\b(' + re.escape(word) + r')\b', re.IGNORECASE)
        new_result, count = pattern.subn(token + r' \1', result)
        if count > 0:
            found_any = True
            result = new_result
    return result if found_any else None


def inject_spurious_token(
    text: str,
    label: int,
    p_correct: float,
    spurious_tokens: Dict[str, str],
    rng: np.random.Generator,
    position: str = "prefix",
    neutral_words: Optional[List[str]] = NEUTRAL_WORDS,
) -> str:
    """
    Injecte un token textuel trompeur dans le texte avec corrélation contrôlée.

    Deux modes selon `neutral_words` :

    Mode 1 — préfixe devant chaque mot neutre (défaut, recommandé) :
      Insère le token spurieux devant chaque occurrence de mot fréquent
      sémantiquement vide ("the", "a", "of", ...) en conservant le mot.
      Ex : "fire the cat" (conserve "the"). Le signal est distribué à travers
      l'ensemble du texte → beaucoup plus difficile à ignorer pour ERM.
      Utilisé quand `neutral_words` est fourni ET que le texte contient des occurrences.

    Mode 2 — préfixe de phrase (fallback) :
      Quand aucun mot neutre n'est présent dans le texte (textes très courts),
      le token est ajouté en préfixe ou suffixe.

    Parameters
    ----------
    text : str            Texte SMS original.
    label : int           Label vrai (0=ham, 1=spam).
    p_correct : float     Probabilité d'associer le "bon" token au label.
    spurious_tokens : dict  {"spam_correlated": tok, "ham_correlated": tok}.
    rng : np.random.Generator
    position : str        "prefix" ou "suffix" (fallback uniquement).
    neutral_words : List[str] ou None
        Si None, mode préfixe de phrase seul.
    """
    spam_token = spurious_tokens["spam_correlated"]
    ham_token  = spurious_tokens["ham_correlated"]

    if rng.uniform() < p_correct:
        token = spam_token if label == 1 else ham_token
    else:
        token = ham_token  if label == 1 else spam_token

    if neutral_words is not None:
        prepended = _prepend_token_to_neutral_words(text, neutral_words, token)
        if prepended is not None:
            return prepended

    # Fallback préfixe/suffixe
    return f"{token} {text}" if position == "prefix" else f"{text} {token}"


# =============================================================================
# Tokenisation et extraction d'embeddings BERT
# =============================================================================

# =============================================================================
# Singleton BERT — chargé une seule fois par processus
# =============================================================================

_BERT_CACHE: Dict[str, Any] = {}   # clé : model_name → {"tokenizer": ..., "model": ...}


def _get_bert(model_name: str, device: str):
    """
    Retourne (tokenizer, model) en les chargeant une seule fois par model_name.
    Les appels suivants réutilisent l'instance en mémoire, évitant de recharger
    BERT depuis le disque (~500 Mo, ~2-5 s) à chaque build_envs_*.
    """
    if model_name not in _BERT_CACHE:
        print(f"  [BERT] Chargement de {model_name} (une seule fois)…")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _BERT_CACHE[model_name] = {"tokenizer": tokenizer, "model": model}

    entry = _BERT_CACHE[model_name]
    # Déplacer sur le bon device si nécessaire
    current_device = next(entry["model"].parameters()).device
    if str(current_device) != str(device):
        entry["model"] = entry["model"].to(device)
    return entry["tokenizer"], entry["model"]


# =============================================================================
# Cache d'embeddings sur disque
# =============================================================================

def _embed_cache_path(texts: List[str], model_name: str,
                      max_length: int, pooling: str) -> str:
    """
    Calcule un chemin de cache unique basé sur un hash du contenu +
    des hyperparamètres d'embedding.  Stocké dans nlp_synthetic/.embed_cache/.
    """
    import hashlib, os
    digest = hashlib.md5(
        (repr(texts) + model_name + str(max_length) + pooling).encode()
    ).hexdigest()
    cache_dir = os.path.join(os.path.dirname(__file__), ".embed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{digest}.npy")


def tokenize_and_embed_with_bert(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    use_cache: bool = True,
) -> np.ndarray:
    """
    Tokenise les textes et extrait les embeddings BERT.

    Optimisations par rapport à la version initiale :
    - **Singleton BERT** : le modèle est chargé une seule fois par processus
      (pas de rechargement à chaque build_envs_*).
    - **Cache disque** : les embeddings sont sauvegardés dans
      nlp_synthetic/.embed_cache/<hash>.npy.  Un second lancement avec les mêmes
      textes + hyperparamètres est quasi-instantané.
    - **Batch size adaptatif** : 64 sur GPU/MPS, 32 sur CPU.

    Parameters
    ----------
    texts : List[str]
    model_name : str
    max_length : int
    device : str
    pooling : str      "mean" | "cls" | "max"
    use_cache : bool   Mettre False pour forcer le re-calcul.
    """
    import os

    # ── Cache disque ──────────────────────────────────────────────────────
    if use_cache:
        cache_path = _embed_cache_path(texts, model_name, max_length, pooling)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    # ── Modèle (singleton) ────────────────────────────────────────────────
    tokenizer, model = _get_bert(model_name, device)
    model_device = next(model.parameters()).device

    if str(model_device) in ("cuda", "mps"):
        batch_size = 256 if max_length <= 128 else 128
    else:
        batch_size = 64

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids      = encoded["input_ids"].to(model_device)
            attention_mask = encoded["attention_mask"].to(model_device)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, seq_len, 768)

            if pooling == "mean":
                mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_emb  = torch.sum(last_hidden * mask_exp, dim=1)
                sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
                batch_emb = sum_emb / sum_mask
            elif pooling == "cls":
                batch_emb = last_hidden[:, 0, :]
            elif pooling == "max":
                batch_emb = torch.max(last_hidden, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            embeddings.append(batch_emb.cpu().numpy())

    result = np.concatenate(embeddings, axis=0).astype(np.float32)

    # ── Sauvegarde cache ─────────────────────────────────────────────────
    if use_cache:
        np.save(cache_path, result)
        print(f"  [cache] Embeddings sauvegardés → {os.path.basename(cache_path)}")

    return result


# =============================================================================
# Génération d'environnements NLP semi anti-causal
# =============================================================================

def make_env_nlp_semi_anti_causal(
    all_texts: List[str],
    all_labels: List[int],
    n: int,
    p_correct: float,
    seed: int,
    label_flip: float = 0.25,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Génère un environnement NLP avec corrélation emoji-label contrôlée.
    
    Parameters
    ----------
    all_texts : List[str]
        Pool de tous les SMS disponibles.
    all_labels : List[int]
        Labels correspondants.
    n : int
        Nombre d'exemples à échantillonner.
    p_correct : float
        Probabilité d'associer le "bon" emoji au label (0.0=inversé, 1.0=parfait).
    seed : int
        Graine aléatoire.
    bert_model : str
        Modèle BERT à utiliser.
    max_length : int
        Longueur max de séquence.
    device : str
        Device PyTorch.
    pooling : str
        Type de pooling pour les embeddings.
    
    Returns
    -------
    X : np.ndarray (n, 768)
        Embeddings BERT.
    Y : np.ndarray (n, 1)
        Labels.
    texts_original : List[str]
        Textes originaux (pour debug).
    texts_modified : List[str]
        Textes avec tokens trompeurs injectés.
    """
    rng = np.random.default_rng(seed)
    
    # 1) Échantillonner n exemples
    n_available = len(all_texts)
    if n > n_available:
        # Si on demande plus d'exemples que disponible, échantillonner avec remplacement
        indices = rng.choice(n_available, size=n, replace=True)
    else:
        indices = rng.choice(n_available, size=n, replace=False)
    
    sampled_texts = [all_texts[i] for i in indices]
    sampled_labels = np.array([all_labels[i] for i in indices])
    
    # 2) Appliquer le flip de label (AVANT l'injection d'emoji)
    #    Cela affaiblit le signal causal texte→label
    if label_flip > 0.0:
        flip_mask = rng.uniform(size=len(sampled_labels)) < label_flip
        sampled_labels_flipped = sampled_labels.copy()
        sampled_labels_flipped[flip_mask] = 1 - sampled_labels_flipped[flip_mask]
    else:
        sampled_labels_flipped = sampled_labels.copy()
    
    # 3) Injecter tokens trompeurs basés sur les labels BRUITÉS
    #    (La corrélation emoji-label sera donc avec le label bruité, pas le vrai)
    spurious_tokens = define_spurious_tokens()
    texts_modified = []
    
    for text, label in zip(sampled_texts, sampled_labels_flipped):
        modified_text = inject_spurious_token(
            text=text,
            label=int(label),
            p_correct=p_correct,
            spurious_tokens=spurious_tokens,
            rng=rng,
            position="prefix"  # Ajouter au début
        )
        texts_modified.append(modified_text)
    
    # 4) Tokeniser et extraire embeddings BERT
    X = tokenize_and_embed_with_bert(
        texts=texts_modified,
        model_name=bert_model,
        max_length=max_length,
        device=device,
        pooling=pooling
    )
    
    # 5) Préparer labels (on retourne les labels BRUITÉS)
    Y = sampled_labels_flipped.reshape(-1, 1).astype(np.float32)
    
    return X, Y, sampled_texts, texts_modified


# =============================================================================
# Construction de plusieurs environnements (train/val/test)
# =============================================================================

def _split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (train_idx, val_idx) comme dans data_synth.py"""

# =============================================================================
# Construction complète avec tout le dataset (80/10/10 split)
# =============================================================================

def build_envs_nlp_semi_anti_causal(
    n: int,  # IGNORÉ
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    val_frac: float = 0.2,  # IGNORÉ  
    n_test: Optional[int] = None,  # IGNORÉ
    label_flip: float = 0.25,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """Utilise TOUT le dataset avec split 80/10/10 et pas de chevauchement entre envs."""
    print("Chargement du dataset SMS Spam...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} SMS")
    
    # Split global
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split = int(n_total * 0.1)
    
    test_indices = indices[:n_test_split]
    val_indices = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]
    
    print(f"\nSplit: Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")
    
    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs
    
    train_envs, val_envs = [], []
    spurious_tokens = define_spurious_tokens()
    
    # TRAIN ENVS
    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start = i * samples_per_env
        end = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]
        
        texts = [all_texts[j] for j in env_idx]
        labels = np.array([all_labels[j] for j in env_idx])
        
        if label_flip > 0:
            mask = rng.uniform(size=len(labels)) < label_flip
            labels[mask] = 1 - labels[mask]
        
        texts_mod = [inject_spurious_token(t, int(l), p_correct, spurious_tokens, rng)
                     for t, l in zip(texts, labels)]
        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling)
        Y = labels.reshape(-1, 1).astype(np.float32)
        
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                             meta={"p_correct": p_correct, "label_flip": label_flip, "n_samples": len(X)}))
        
        # VAL ENV
        print(f"=== Val Env {i} ===")
        val_texts = [all_texts[j] for j in val_indices]
        val_labels = np.array([all_labels[j] for j in val_indices])
        # Même label_flip que le train pour que val ait la même distribution
        if label_flip > 0:
            val_rng = np.random.default_rng(seed + 5000 + i)
            mask_val = val_rng.uniform(size=len(val_labels)) < label_flip
            val_labels[mask_val] = 1 - val_labels[mask_val]
        val_texts_mod = [inject_spurious_token(t, int(l), p_correct, spurious_tokens,
                                              np.random.default_rng(seed+5000+i))
                        for t, l in zip(val_texts, val_labels)]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling)
        Y_val = val_labels.reshape(-1, 1).astype(np.float32)
        
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                           meta={"p_correct": p_correct, "n_samples": len(X_val)}))
    
    # TEST ENV
    print(f"\n=== Test OOD (p_correct={test_p_correct:.0%}) ===")
    test_texts = [all_texts[j] for j in test_indices]
    test_labels = np.array([all_labels[j] for j in test_indices])
    test_rng = np.random.default_rng(seed+777)
    test_texts_mod = [inject_spurious_token(t, int(l), test_p_correct, spurious_tokens, test_rng)
                     for t, l in zip(test_texts, test_labels)]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling)
    Y_test = test_labels.reshape(-1, 1).astype(np.float32)
    
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                  meta={"p_correct": test_p_correct, "n_samples": len(X_test)})
    
    print(f"\n✅ Done! Train: {sum(e.X.shape[0] for e in train_envs)} | Val: {val_envs[0].X.shape[0]} | Test: {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Confounding helpers (shared by the 3 variants)
# =============================================================================
#
# DAG commun à tous les cas de confounding :
#
#   C   (confondeur latent binaire)
#   ├── C → Z → token injecté dans le texte   (chemin spurieux)
#   └── C → Y   (bruitage direct du label)
#   text → Y                                   (chemin causal invariant)
#
# Trois variants selon ce qui varie entre les environnements :
#   1. varying_proxy  : a_e = bruit sur le lien C → Z   (a_e varie)
#   2. varying_gamma  : gamma_e = force de C → Y         (gamma_e varie)
#   3. varying_pc     : p_e = prévalence de C             (p_e varie)
# =============================================================================

def _apply_conf_label_flip(
    labels: np.ndarray,
    C: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Applique le bruitage de label dû au confondeur C.
    Quand C=1, flip le label avec probabilité gamma.
    """
    out = labels.copy()
    if gamma > 0.0:
        mask = (C == 1) & (rng.uniform(size=len(out)) < gamma)
        out[mask] = 1 - out[mask]
    return out


def _conf_make_env(
    texts: List[str],
    labels: np.ndarray,     # labels ORIGINAUX (avant bruitage)
    C: np.ndarray,          # confondeur (n,) valeurs 0/1
    Z: np.ndarray,          # proxy spurieux (n,) valeurs 0/1
    gamma: float,           # prob que C bruite le label
    rng: np.random.Generator,
    bert_model: str,
    max_length: int,
    device: str,
    pooling: str,
    apply_gamma: bool = True,   # False pour val/test dans varying_gamma
    conf_tokens: Optional[Dict[str, str]] = None,  # Si None, utilise define_spurious_tokens()
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un environment confounding :
      - Bruite les labels via C (si apply_gamma)
      - Injecte un token basé sur Z (pas sur Y) dans chaque texte
      - Encode avec BERT
    Retourne (X, Y).
    """
    spurious_tokens = conf_tokens if conf_tokens is not None else define_spurious_tokens()
    labels_obs = _apply_conf_label_flip(labels, C, gamma, rng) if apply_gamma else labels.copy()

    # Z=1 → token "spam_correlated" (fire), Z=0 → token "ham_correlated" (sky)
    # On passe z comme "label" avec p_correct=1.0 → token = class_tokens[z] toujours
    rng_inj = np.random.default_rng(int(rng.integers(0, 2**31)))
    texts_mod = [
        inject_spurious_token(text, int(z), 1.0, spurious_tokens, rng_inj)
        for text, z in zip(texts, Z)
    ]
    X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling)
    Y = labels_obs.reshape(-1, 1).astype(np.float32)
    return X, Y


def _conf_global_split(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Charge SMS Spam et retourne (all_texts, all_labels, indices_permuted, n_total)."""
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    return all_texts, np.array(all_labels), indices, n_total


# =============================================================================
# Confounding variant 1 — varying_proxy
# =============================================================================
# Le paramètre a_e (bruit dans Z = C XOR Ber(a_e)) varie entre les envs.
# a ≈ 0  → Z ≈ C  → token fortement corrélé avec Y (via C)
# a = 0.5 → Z aléatoire → pas de corrélation token–Y
# a ≈ 1  → Z ≈ NOT C → token anti-corrélé avec Y → ERM piégé en OOD
# Flip de label : C ~ Ber(p_c_flip=0.25), si C=1 → flip déterministe.
# =============================================================================

def build_envs_nlp_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    SMS Spam — confounding avec variation du proxy Z = C XOR Ber(a_e).

    DAG :  C ~ Ber(p_c_flip) → Z(a_e) → token ; C → Y (flip déterministe si C=1) ; text → Y
    Variation d'env : a_e (bruit sur C→Z).
    OOD : a_test ≈ 1 → token anti-corrélé avec Y.

    Parameters
    ----------
    a_train  : List[float]  Bruit proxy par env train (ex : [0.01, 0.11]).
    a_test   : float        Bruit proxy OOD (ex : 0.99).
    p_c_flip : float        P(C=1) = fraction des labels flippés (défaut 0.25).
                            Le flip est déterministe : C=1 ⟹ label toujours inversé.
    """
    print("Chargement du dataset SMS Spam (confounding – varying proxy)...")
    all_texts, all_labels, indices, n_total = _conf_global_split(seed)
    n_test  = int(n_total * 0.1)
    n_val   = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} SMS | Split 80/10/10 : Train {len(train_idx)} | Val {len(n_val if False else val_idx)} | Test {len(test_idx)}")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    train_envs, val_envs = [], []
    rng_master = np.random.default_rng(seed + 1)

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[j]  for j in env_idx]
        labels = all_labels[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c_flip, size=len(labels))
        N = rng_e.binomial(1, a_e,  size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(texts, labels, C, Z, 1.0, rng_e, bert_model, max_length, device, pooling)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "nlp_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (a={a_e}) ===")
        val_texts  = [all_texts[j]  for j in val_idx]
        val_labels = all_labels[val_idx]
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c_flip, size=len(val_labels))
        Nv = rng_v.binomial(1, a_e,  size=len(val_labels))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        X_val, Y_val = _conf_make_env(val_texts, val_labels, Cv, Zv, 1.0, rng_v,
                                      bert_model, max_length, device, pooling)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "nlp_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[j]  for j in test_idx]
    test_labels = all_labels[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, p_c_flip, size=len(test_labels))
    Nt = rng_t.binomial(1, a_test, size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(test_texts, test_labels, Ct, Zt, 1.0, rng_t,
                                    bert_model, max_length, device, pooling,
                                    apply_gamma=False)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "nlp_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Confounding variant 2 — varying_gamma
# =============================================================================
# gamma_e (force de C sur Y) varie entre les envs.
# gamma ≈ 1 → C détermine quasiment Y → token très corrélé avec Y
# gamma = 0 → C n'affecte pas Y → token non corrélé avec Y → ERM piégé en OOD
# a = 0 (Z = C, proxy parfait)
# =============================================================================

def build_envs_nlp_conf_varying_gamma(
    gamma_train: List[float],
    gamma_test: float,
    seed: int,
    a: float = 0.0,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    SMS Spam — confounding avec variation de l'influence C → Y (gamma_e).

    DAG :  C → Z=C → token ; C → Y (flip prob gamma_e) ; text → Y
    Variation d'env : gamma_e (force du confondeur sur Y).
    OOD : gamma_test = 0 → token sans corrélation avec Y → ERM piégé.

    Parameters
    ----------
    gamma_train : List[float]  Force de C→Y par env train (ex : [0.8, 0.5]).
    gamma_test  : float        Force en OOD (ex : 0.0).
    a           : float        Bruit proxy C→Z (0.0 = proxy parfait).
    """
    print("Chargement du dataset SMS Spam (confounding – varying gamma)...")
    all_texts, all_labels, indices, n_total = _conf_global_split(seed)
    n_test  = int(n_total * 0.1)
    n_val   = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} SMS | Split 80/10/10 : Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(gamma_train)
    spe = len(train_idx) // n_envs
    train_envs, val_envs = [], []

    for i, g in enumerate(gamma_train):
        print(f"\n=== Train Env {i} (gamma={g}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[j]  for j in env_idx]
        labels = all_labels[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, 0.5, size=len(labels))
        N = rng_e.binomial(1, a,   size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(texts, labels, C, Z, g, rng_e, bert_model, max_length, device, pooling)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "nlp_conf_varying_gamma", "a": a, "gamma": g,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (gamma={g}) ===")
        val_texts  = [all_texts[j]  for j in val_idx]
        val_labels = all_labels[val_idx]
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, 0.5, size=len(val_labels))
        Nv = rng_v.binomial(1, a,   size=len(val_labels))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        # Val : pas de flip gamma (labels propres)
        X_val, Y_val = _conf_make_env(val_texts, val_labels, Cv, Zv, g, rng_v,
                                      bert_model, max_length, device, pooling, apply_gamma=False)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "nlp_conf_varying_gamma", "a": a, "gamma": g,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (gamma={gamma_test}) ===")
    if gamma_test == 0.0:
        print("  gamma=0 → C n'affecte pas Y → token non corrélé avec Y → ERM piégé.")
    test_texts  = [all_texts[j]  for j in test_idx]
    test_labels = all_labels[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, 0.5, size=len(test_labels))
    Nt = rng_t.binomial(1, a,   size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(test_texts, test_labels, Ct, Zt, gamma_test, rng_t,
                                    bert_model, max_length, device, pooling, apply_gamma=False)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "nlp_conf_varying_gamma", "a": a, "gamma": gamma_test,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ Confounding varying gamma — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Confounding variant 3 — varying_pc
# =============================================================================
# p_e = prévalence de C (C ~ Ber(p_e)) varie entre les envs.
# p_e élevé → C plus fréquent → token plus souvent corrélé avec Y
# p_e faible → C rare → token peu corrélé avec Y en test OOD
# a et gamma sont fixés.
# =============================================================================

def build_envs_nlp_conf_varying_pc(
    pc_train: List[float],
    pc_test: float,
    seed: int,
    a: float = 0.0,
    gamma: float = 0.5,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    SMS Spam — confounding avec variation de la prévalence de C (p_e).

    DAG :  C~Ber(p_e) → Z=C⊕Ber(a) → token ; C → Y (flip prob gamma) ; text → Y
    Variation d'env : p_e (prévalence du confondeur).
    OOD : pc_test non vu en train → corrélation token–Y change de régime.

    Parameters
    ----------
    pc_train : List[float]  Prévalence de C par env train (ex : [0.8, 0.5]).
    pc_test  : float        Prévalence OOD (ex : 0.1).
    a        : float        Bruit proxy (0.0 = Z=C).
    gamma    : float        Force de C→Y (fixe).
    """
    print("Chargement du dataset SMS Spam (confounding – varying pc)...")
    all_texts, all_labels, indices, n_total = _conf_global_split(seed)
    n_test  = int(n_total * 0.1)
    n_val   = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} SMS | Split 80/10/10 : Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(pc_train)
    spe = len(train_idx) // n_envs
    train_envs, val_envs = [], []

    for i, p_c in enumerate(pc_train):
        print(f"\n=== Train Env {i} (p_c={p_c}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[j]  for j in env_idx]
        labels = all_labels[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c, size=len(labels))
        N = rng_e.binomial(1, a,   size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(texts, labels, C, Z, gamma, rng_e, bert_model, max_length, device, pooling)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "nlp_conf_varying_pc", "p_c": p_c, "a": a, "gamma": gamma,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (p_c={p_c}) ===")
        val_texts  = [all_texts[j]  for j in val_idx]
        val_labels = all_labels[val_idx]
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c, size=len(val_labels))
        Nv = rng_v.binomial(1, a,   size=len(val_labels))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        X_val, Y_val = _conf_make_env(val_texts, val_labels, Cv, Zv, gamma, rng_v,
                                      bert_model, max_length, device, pooling)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "nlp_conf_varying_pc", "p_c": p_c, "a": a, "gamma": gamma,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (p_c={pc_test}) ===")
    test_texts  = [all_texts[j]  for j in test_idx]
    test_labels = all_labels[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, pc_test, size=len(test_labels))
    Nt = rng_t.binomial(1, a,       size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(test_texts, test_labels, Ct, Zt, gamma, rng_t,
                                    bert_model, max_length, device, pooling)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "nlp_conf_varying_pc", "p_c": pc_test, "a": a, "gamma": gamma,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ Confounding varying pc — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Size-based selection helpers
# =============================================================================


# =============================================================================
# Size-based selection helpers
# =============================================================================

def compute_size_thresholds(
    texts: List[str],
    labels: List[int],
    method: str = "quartile"
) -> Tuple[float, float]:
    """
    Calcule les seuils de taille pour la sélection basée sur la longueur.
    
    Parameters
    ----------
    texts : List[str]
        Liste des textes.
    labels : List[int]
        Labels (0=ham, 1=spam).
    method : str
        Méthode de calcul ("quartile", "median", ou "auto").
    
    Returns
    -------
    t1 : float
        Seuil pour HAM (HAM sélectionné si len < t1).
    t2 : float
        Seuil pour SPAM (SPAM sélectionné si len > t2).
    
    Notes
    -----
    - method="quartile": Q1(HAM) et Q3(SPAM) → forte séparation (25% vs 75%)
    - method="median": médianes des distributions (50% vs 50%)
    - method="auto": choisit des seuils pour ~40% d'exemples typiques
    """
    lengths = [len(text) for text in texts]
    spam_lengths = [lengths[i] for i in range(len(labels)) if labels[i] == 1]
    ham_lengths = [lengths[i] for i in range(len(labels)) if labels[i] == 0]
    
    if method == "quartile":
        # Q1 pour HAM (25% plus courts) et Q3 pour SPAM (75% plus longs)
        # → Signal spurious FORT : HAM très courts, SPAM très longs
        t1 = np.percentile(ham_lengths, 25)   # Q1
        t2 = np.percentile(spam_lengths, 75)   # Q3
    elif method == "median":
        t1 = np.median(ham_lengths)
        t2 = np.median(spam_lengths)
    elif method == "auto":
        # Choisir des seuils qui créent ~40% d'exemples typiques
        t1 = np.percentile(ham_lengths, 60)
        t2 = np.percentile(spam_lengths, 40)
    elif method == "soft":
        # "Soft" selection: HAM < 40th, SPAM > 60th
        # Less extreme than quartile (25/75), keeps more examples and reduces size gap
        t1 = np.percentile(ham_lengths, 35)
        t2 = np.percentile(spam_lengths, 65)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  Seuils calculés ({method}): HAM < {t1:.0f} chars, SPAM > {t2:.0f} chars")
    return t1, t2


def is_typical_by_size(
    text: str,
    label: int,
    t1: float,
    t2: float
) -> bool:
    """
    Vérifie si un message est "typique" selon sa taille.
    
    Parameters
    ----------
    text : str
        Texte du message.
    label : int
        Label (0=ham, 1=spam).
    t1 : float
        Seuil pour HAM.
    t2 : float
        Seuil pour SPAM.
    
    Returns
    -------
    bool
        True si typique (HAM court ou SPAM long), False sinon.
    
    Examples
    --------
    >>> is_typical_by_size("Hi", 0, 333, 457)  # HAM court
    True
    >>> is_typical_by_size("Very long message...", 1, 333, 457)  # SPAM long
    True
    >>> is_typical_by_size("Very long message...", 0, 333, 457)  # HAM long (atypique)
    False
    """
    text_len = len(text)
    
    if label == 0:  # HAM
        return text_len < t1
    else:  # SPAM (1)
        return text_len > t2


# =============================================================================
# Construction d'environnements avec sélection basée sur la taille
# =============================================================================

def build_envs_nlp_size_selection(
    train_p_select: List[float],
    seed: int,
    threshold_method: str = "quartile",
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit des environnements NLP avec sélection basée sur la TAILLE des messages.
    
    **DAG**: Y → Z (taille) → S (sélection)
    
    **Amélioration** : Signal spurious FORT via quartiles + OOD extrêmes opposés
    
    **Mécanisme**:
    1. Diviser le dataset en 2 moitiés (une par environnement)
    2. Calculer seuils t1 (HAM) et t2 (SPAM) sur chaque moitié
    3. Sélectionner avec p_select si "typique" (HAM court ou SPAM long)
    4. Rejeter les "atypiques" (HAM longs, SPAM courts) → test OOD
    
    Parameters
    ----------
    train_p_select : List[float]
        Probabilités de sélection par env (ex: [0.9, 0.8]).
    seed : int
        Graine aléatoire.
    threshold_method : str
        Méthode de calcul des seuils ("median" ou "auto").
    val_frac : float
        Fraction pour validation.
    label_flip : float
        Taux de bruit symetrique sur les labels train/val. Applique apres la
        selection pour affaiblir le signal causal sans changer le mecanisme de
        selection lui-meme.
    bert_model : str
        Modèle BERT.
    max_length : int
        Longueur max séquence.
    device : str
        Device PyTorch.
    pooling : str
        Type de pooling.
    
    Returns
    -------
    train_envs : List[Env]
        Environnements d'entraînement (messages typiques).
    val_envs : List[Env]
        Environnements de validation.
    test_env : Env
        Environnement de test OOD (messages atypiques).
    """
    print("Chargement du dataset SMS Spam (mshenoda/spam-messages)...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} SMS")
    
    rng = np.random.default_rng(seed)
    
    # 1) SPLIT EN 2 MOITIÉS (une par environnement)
    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    samples_per_env = n_total // n_envs
    
    print(f"\n📊 Split en {n_envs} environnements:")
    
    # Pool for EXTREME OPPOSITE examples only (not all rejected)
    extreme_opposite_texts = []
    extreme_opposite_labels = []
    
    train_envs, val_envs = [], []
    spurious_tokens = define_spurious_tokens()
    
    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        
        # Partition spécifique à cet environnement
        env_start = i * samples_per_env
        env_end = (i + 1) * samples_per_env if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]
        
        env_texts = [all_texts[j] for j in env_indices]
        env_labels = [all_labels[j] for j in env_indices]
        
        print(f"  {len(env_texts)} SMS dans cette partition")
        
        # 2) CALCULER SEUILS pour cet environnement
        t1, t2 = compute_size_thresholds(env_texts, env_labels, threshold_method)
        
        # 3) SÉLECTIONNER selon la taille + collecter OPPOSÉS EXTRÊMES
        selected_texts = []
        selected_labels = []
        
        for text, label in zip(env_texts, env_labels):
            text_len = len(text)
            
            if is_typical_by_size(text, label, t1, t2):
                # Typique → sélectionner avec p_select
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            else:
                # Atypique → vérifier si EXTRÊME OPPOSÉ pour OOD
                # HAM long (> t2) OU SPAM court (< t1) = opposé extrême
                if label == 0 and text_len > t2:  # HAM très long (> seuil SPAM)
                    extreme_opposite_texts.append(text)
                    extreme_opposite_labels.append(label)
                elif label == 1 and text_len < t1:  # SPAM très court (< seuil HAM)
                    extreme_opposite_texts.append(text)
                    extreme_opposite_labels.append(label)
        
        print(f"  Sélectionné: {len(selected_texts)} SMS typiques")
        
        # 4) Split train/val
        n_selected = len(selected_texts)
        n_val = int(n_selected * val_frac)
        n_train = n_selected - n_val
        
        indices_shuffled = rng.permutation(n_selected)
        train_idx = indices_shuffled[:n_train]
        val_idx = indices_shuffled[n_train:]
        
        # Train env
        train_texts = [selected_texts[j] for j in train_idx]
        train_labels = np.array([selected_labels[j] for j in train_idx])
        if label_flip > 0.0:
            rng_train_flip = np.random.default_rng(seed + 7000 + i)
            flip_mask = rng_train_flip.uniform(size=len(train_labels)) < label_flip
            train_labels[flip_mask] = 1 - train_labels[flip_mask]
        
        X_train = tokenize_and_embed_with_bert(train_texts, bert_model, max_length, device, pooling)
        Y_train = train_labels.reshape(-1, 1).astype(np.float32)
        
        train_envs.append(Env(
            torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            meta={"p_select": p_select, "kind": "nlp_size_train", "env_id": i,
                  "t1": t1, "t2": t2, "label_flip": label_flip,
                  "n_samples": len(X_train)}
        ))
        
        # Val env
        val_texts = [selected_texts[j] for j in val_idx]
        val_labels = np.array([selected_labels[j] for j in val_idx])
        if label_flip > 0.0:
            rng_val_flip = np.random.default_rng(seed + 8000 + i)
            flip_mask = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            val_labels[flip_mask] = 1 - val_labels[flip_mask]
        
        X_val = tokenize_and_embed_with_bert(val_texts, bert_model, max_length, device, pooling)
        Y_val = val_labels.reshape(-1, 1).astype(np.float32)
        
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val),
            meta={"p_select": p_select, "kind": "nlp_size_val", "env_id": i,
                  "label_flip": label_flip,
                  "n_samples": len(X_val)}
        ))
    
    # 5) TEST OOD (exemples EXTRÊMES OPPOSÉS uniquement)
    print(f"\n=== Test OOD (opposés extrêmes) ===")
    print(f"  {len(extreme_opposite_texts)} SMS extrêmes:")
    print(f"    - HAM très longs (> seuil SPAM)")
    print(f"    - SPAM très courts (< seuil HAM)")
    print(f"  → Signal spurious INVERSÉ au maximum !")
    
    extreme_labels = np.array(extreme_opposite_labels)
    
    X_test = tokenize_and_embed_with_bert(extreme_opposite_texts, bert_model, max_length, device, pooling)
    Y_test = extreme_labels.reshape(-1, 1).astype(np.float32)
    
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(Y_test),
        meta={"kind": "nlp_size_test_ood", "n_samples": len(X_test),
              "description": "extreme_opposite_by_size"}
    )
    
    print(f"\n✅ Environnements créés avec sélection par taille (forte séparation) !")
    print(f"   - {len(train_envs)} envs d'entraînement ({sum(e.X.shape[0] for e in train_envs)} SMS typiques)")
    print(f"   - {len(val_envs)} envs de validation ({sum(e.X.shape[0] for e in val_envs)} SMS)")
    print(f"   - 1 env de test OOD ({test_env.X.shape[0]} SMS extrêmes opposés)")
    
    return train_envs, val_envs, test_env


# =============================================================================
# SST-2 — Dataset ANTI-CAUSAL (Y → X : sentiment → texte)
# =============================================================================
#
# SST-2 (Stanford Sentiment Treebank, Socher et al. 2013) est constitué de
# critiques de films tirées de Rotten Tomatoes.  C'est un dataset ANTI-CAUSAL :
# le SENTIMENT du critique (Y) *cause* ce qu'il écrit (X_z = le texte).
#
# Contrairement à AG News (topique → article) ou SMS Spam (contenu → label),
# dans SST-2 le mécanisme stable est P(X|Y) et non P(Y|X).
#
# Deux expériences (analogues à Rosenfeld et al. 2021 "An Empirical Study of
# Invariant Risk Minimization" et Ahmad & Bhatt "Environment Agnostic IRM
# for Classification of Sequential Datasets") :
#
#   1) sst2_semi_anti_causal : injection d'un token spurieux binaire
#      DAG : Y → X_z   ET   Y → Z = token(Y, p_correct) → X total
#      Variation env : force de corrélation Z–Y (p_correct)
#
#   2) sst2_selection : sélection par lexique de sentiment fort
#      DAG : Y → Z (présence de lexique) → S (sélection)
#      Typique = critique avec mots de sentiment évidents  (train)
#      Atypique = critique subtile, sans marqueur lexical (test OOD)
# =============================================================================

SST2_CLASS_NAMES: Dict[int, str] = {0: "negative", 1: "positive"}

# Tokens spurieux pour SST-2 — directions cardinales, sémantiquement neutres
# vis-à-vis du sentiment. "north"/"south" n'ont aucune association positive/négative
# dans les corpus de pré-entraînement BERT (Wikipedia/BooksCorpus).
SST2_TOKENS: Dict[int, str] = {
    0: "north",   # négatif
    1: "south",   # positif
}

# Mots de sentiment pour le biais de sélection SST-2.
# Listes larges de mots courants dans les critiques de films —
# les critiques CONTENANT ces mots sont les "cas typiques" (train/val),
# les autres constituent le pool de test OOD.
SST2_POSITIVE_WORDS: List[str] = [
    # Jugements généraux positifs
    "good", "great", "best", "love", "loved", "enjoy", "enjoyed",
    "like", "liked", "nice", "fine", "solid", "strong", "smart",
    # Qualité narrative / artistique
    "funny", "fun", "sweet", "clever", "witty", "sharp", "rich",
    "moving", "touching", "emotional", "heartfelt", "warm", "tender",
    "beautiful", "gorgeous", "stunning", "striking", "vivid",
    "compelling", "engaging", "absorbing", "gripping", "riveting",
    "entertaining", "enjoyable", "satisfying", "rewarding", "pleasing",
    # Intensifs
    "wonderful", "excellent", "brilliant", "outstanding", "perfect",
    "masterpiece", "superb", "terrific", "magnificent", "exceptional",
    "charming", "delightful", "remarkable", "unforgettable", "inspired",
    "stunning", "breathtaking", "glorious", "spectacular", "phenomenal",
    "extraordinary", "exquisite", "thrilling", "uplifting", "hilarious",
    "powerful", "refreshing", "impressive", "interesting", "intriguing",
    # Expressions
    "must see", "must-see", "highly recommended", "worth watching",
    "well done", "well-done", "well made", "well-made",
]
SST2_NEGATIVE_WORDS: List[str] = [
    # Jugements généraux négatifs
    "bad", "worst", "poor", "weak", "flat", "dull", "slow", "bland",
    "boring", "tired", "stale", "thin", "cheap", "hollow", "empty",
    "mess", "failure", "fail", "fails", "failed",
    # Qualité narrative / artistique
    "stupid", "silly", "lame", "weak", "clumsy", "lazy", "sloppy",
    "predictable", "formulaic", "clichéd", "cliched", "derivative",
    "contrived", "forced", "unconvincing", "uninteresting", "tedious",
    "forgettable", "pointless", "aimless", "incoherent", "confusing",
    "annoying", "irritating", "painful", "unwatchable", "unbearable",
    # Intensifs
    "terrible", "awful", "horrible", "dreadful", "pathetic",
    "disappointing", "atrocious", "abysmal", "ridiculous", "laughable",
    "pretentious", "insufferable", "excruciating", "soulless", "joyless",
    "vapid", "shallow", "lifeless", "numbing",
    # Expressions
    "complete waste", "waste of time", "don't bother", "avoid",
    "fell flat", "falls flat", "doesn't work", "doesn't work",
]


# =============================================================================
# Chargement de SST-2
# =============================================================================

def load_sst2_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge SST-2 depuis Hugging Face (glue / sst2).

    On combine le split « train » (67 349 exemples) et « validation »
    (872 exemples labelisés).  Le split « test » officiel n'a pas de labels
    (GLUE benchmark) et est ignoré.

    Returns
    -------
    texts  : List[str]   – phrases de critiques de films
    labels : List[int]   – 0 = négatif, 1 = positif
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("glue", "sst2")
    labeled = concatenate_datasets([dataset["train"], dataset["validation"]])
    labeled = labeled.shuffle(seed=seed)

    texts  = list(labeled["sentence"])
    labels = [int(l) for l in labeled["label"]]
    return texts, labels


# =============================================================================
# 1) SST-2 semi anti-causal — injection de token spurieux
# =============================================================================

def build_envs_sst2_semi_anti_causal(
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    label_flip: float = 0.0,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — expérience semi anti-causale par injection de token spurieux.

    Chaque exemple (texte, label y) est modifié :
      - avec proba p_correct  : injecter SST2_TOKENS[y]
      - avec proba 1-p_correct : injecter SST2_TOKENS[1-y]

    La corrélation Z–Y est donc exactement p_correct.
    En test OOD (p_correct=0.0), la corrélation est totalement inversée :
    un ERM qui utilise Z obtient 0% de bonnes prédictions sur cette dimension.

    Paramètres
    ----------
    train_p_correct : List[float]
        Corrélation Z–Y par env d'entraînement (ex : [0.9, 0.8]).
    test_p_correct : float
        Corrélation en test OOD (souvent 0.0).
    seed : int
        Graine aléatoire globale.
    label_flip : float
        Fraction de labels bruités en train (affaiblit le signal causal texte→label).
    bert_model / max_length / device / pooling : voir tokenize_and_embed_with_bert.

    Retourne
    --------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset SST-2...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    neg = int((all_labels_arr == 0).sum())
    pos = int((all_labels_arr == 1).sum())
    print(f"Dataset chargé : {n_total} exemples  (négatif={neg}, positif={pos})")

    # ── Split global 80/10/10 ──────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.10)
    n_val_split  = int(n_total * 0.10)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"Split : Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")

    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    # ── Envs train ─────────────────────────────────────────────────────────
    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start   = i * samples_per_env
        end     = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        texts  = [all_texts[int(j)]  for j in env_idx]
        labels = np.array([all_labels[int(j)] for j in env_idx], dtype=np.int64)

        # Label flip binaire (bruit sur le signal causal)
        if label_flip > 0.0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels)) < label_flip
            labels[flip_mask] = 1 - labels[flip_mask]

        # Injection de token spurieux
        rng_inject = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, SST2_TOKENS, rng_inject)
            for t, l in zip(texts, labels)
        ]
        n_correct = sum(
            SST2_TOKENS[int(l)] in tm.lower().split()
            for tm, l in zip(texts_mod, labels)
        )
        print(f"  Token correct : {n_correct}/{len(labels)} ({n_correct/len(labels):.1%})")

        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling)
        Y = labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(
            torch.from_numpy(X), torch.from_numpy(Y),
            meta={
                "p_correct": p_correct, "label_flip": label_flip,
                "kind": "sst2_semi_anti_causal_train",
                "n_samples": len(X), "dataset": "sst2",
            },
        ))

        # ── Val env (même p_correct, même label_flip que le train) ──────────────────
        print(f"=== Val Env {i} (p_correct={p_correct:.0%}) ===")
        val_texts  = [all_texts[int(j)]  for j in val_indices]
        val_labels = np.array([all_labels[int(j)] for j in val_indices], dtype=np.int64)

        # Même label_flip que le train pour que val ait la même distribution
        if label_flip > 0.0:
            rng_val_flip = np.random.default_rng(seed + 5000 + i + 1)
            flip_mask_val = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            val_labels[flip_mask_val] = 1 - val_labels[flip_mask_val]

        rng_val = np.random.default_rng(seed + 5000 + i)
        val_texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, SST2_TOKENS, rng_val)
            for t, l in zip(val_texts, val_labels)
        ]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling)
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(val_labels.reshape(-1, 1).astype(np.float32)),
            meta={
                "p_correct": p_correct, "kind": "sst2_semi_anti_causal_val",
                "n_samples": len(X_val), "dataset": "sst2",
            },
        ))

    # ── Test OOD ──────────────────────────────────────────────────────────
    print(f"\n=== Test OOD SST-2 (p_correct={test_p_correct:.0%}) ===")
    if test_p_correct == 0.0:
        print("  Token toujours erroné → ERM piégé, IRM attendu robuste.")

    test_texts  = [all_texts[int(j)]  for j in test_indices]
    test_labels = np.array([all_labels[int(j)] for j in test_indices], dtype=np.int64)

    rng_test = np.random.default_rng(seed + 777)
    test_texts_mod = [
        inject_spurious_token_multiclass(t, int(l), test_p_correct, SST2_TOKENS, rng_test)
        for t, l in zip(test_texts, test_labels)
    ]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling)

    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(test_labels.reshape(-1, 1).astype(np.float32)),
        meta={
            "p_correct": test_p_correct,
            "kind": "sst2_semi_anti_causal_test_ood",
            "n_samples": len(X_test), "dataset": "sst2",
        },
    )

    print(f"\n✅ SST-2 Semi Anti-Causal — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} exemples)")
    print(f"   - {n_envs} envs val    ({val_envs[0].X.shape[0]} exemples / env)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} exemples, p_correct={test_p_correct:.0%})")

    return train_envs, val_envs, test_env


# =============================================================================
# 2) SST-2 selection — biais de sélection par lexique de sentiment fort
# =============================================================================

def is_typical_sst2(
    text: str,
    label: int,
    positive_words: List[str] = SST2_POSITIVE_WORDS,
    negative_words: List[str] = SST2_NEGATIVE_WORDS,
) -> bool:
    """
    Retourne True si la critique contient des marqueurs lexicaux forts
    correspondant à son label.

    - Positif (1) avec ≥ 1 mot de ``SST2_POSITIVE_WORDS``
    - Négatif (0) avec ≥ 1 mot de ``SST2_NEGATIVE_WORDS``

    Les critiques sans marqueur lexical fort sont des « cas subtils »
    utilisés comme test OOD.
    """
    text_lower = text.lower()
    if label == 1:
        return any(w in text_lower for w in positive_words)
    else:
        return any(w in text_lower for w in negative_words)


def is_cross_label_sst2(
    text: str,
    label: int,
    positive_words: List[str] = SST2_POSITIVE_WORDS,
    negative_words: List[str] = SST2_NEGATIVE_WORDS,
) -> bool:
    """
    Retourne True si la critique contient des marqueurs lexicaux forts du
    label OPPOSÉ au sien.

    - Positif (1) contenant ≥ 1 mot négatif fort
      (ex: « terrible acting but overall wonderful », label=1)
    - Négatif (0) contenant ≥ 1 mot positif fort
      (ex: « wonderful premise, terrible execution », label=0)

    Ces exemples sont utilisés comme test OOD adversarial : les mots-clés
    pointent dans la mauvaise direction pour ERM.
    """
    text_lower = text.lower()
    if label == 1:  # critique positive contenant des mots négatifs
        return any(w in text_lower for w in negative_words)
    else:           # critique négative contenant des mots positifs
        return any(w in text_lower for w in positive_words)


# =============================================================================
# SST-2 — Genre-type selection helpers
# =============================================================================
# Corrélation naturelle dans Rotten Tomatoes :
#   - Documentaires / films indépendants → notés positivement par les critiques
#   - Suites / remakes / reboots → notés négativement par les critiques
# Cette corrélation est SPURIEUSE : le type de film ne cause pas le sentiment
# de la critique individuelle (une bonne suite et un mauvais documentaire existent).
# =============================================================================

SST2_POSITIVE_GENRE_WORDS: List[str] = [
    "documentary", "documentaries",
    "indie", "independent",
    "foreign", "foreign-language",
    "arthouse", "art-house", "art house",
    "debut", "first film", "first feature",
    "biopic",
]

SST2_NEGATIVE_GENRE_WORDS: List[str] = [
    "sequel", "sequels",
    "remake", "remakes",
    "reboot", "reboots",
    "franchise", "franchises",
    "prequel", "prequels",
    "spin-off", "spinoff",
    "part 2", "part ii", "part iii", "part iv",
    "chapter 2", "chapter ii",
    "episode ii", "episode iii",
]


def is_typical_genre_sst2(text: str, label: int) -> bool:
    """
    Retourne True si la critique suit le pattern genre–note typique :
      - Positive (Y=1) + vocabulaire documentaire/indie → typique
      - Négative (Y=0) + vocabulaire suite/remake → typique
    """
    text_lower = text.lower()
    if label == 1:
        return any(w in text_lower for w in SST2_POSITIVE_GENRE_WORDS)
    else:
        return any(w in text_lower for w in SST2_NEGATIVE_GENRE_WORDS)


def is_cross_genre_sst2(text: str, label: int) -> bool:
    """
    Retourne True si le vocabulaire de genre CONTREDIT le label :
      - Positive (Y=1) + vocabulaire suite/remake → bonne suite (OOD)
      - Négative (Y=0) + vocabulaire documentaire/indie → mauvais documentaire (OOD)
    """
    text_lower = text.lower()
    if label == 1:
        return any(w in text_lower for w in SST2_NEGATIVE_GENRE_WORDS)
    else:
        return any(w in text_lower for w in SST2_POSITIVE_GENRE_WORDS)


def build_envs_sst2_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    ood_strategy: str = "cross_label",
) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — expérience de sélection par lexique de sentiment fort.

    DAG : Y → Z (présence d'un marqueur lexical) → S (sélection d'entraînement)

    Mécanisme :
      - Typique (Z=1) : critique contenant ≥ 1 mot fort (ex : "wonderful")
        → sélectionné avec proba p_select dans les envs de train.
      - Test OOD selon ood_strategy :
          "atypical"    : critiques sans aucun mot fort (signal Z absent)
          "cross_label" : critiques où les mots forts *contredisent* le label
                          → ERM prédit à l'inverse du vrai label (défaut).

    Paramètres
    ----------
    train_p_select : List[float]
        Proba de garder un exemple typique par env (ex : [0.99, 0.5]).
    label_flip : float
        Taux de bruit symetrique sur les labels train/val. Applique apres la
        selection pour diminuer le signal causal sans changer la selection.
    ood_strategy : str
        Stratégie OOD : "cross_label" (défaut) ou "atypical".
    seed, val_frac, bert_model, max_length, device, pooling : cf. hab.

    Retourne
    --------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset SST-2...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} critiques  |  OOD strategy : {ood_strategy}")

    rng = np.random.default_rng(seed)

    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    samples_per_env = n_total // n_envs

    ood_texts:  List[str] = []
    ood_labels: List[int] = []

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")

        env_start  = i * samples_per_env
        env_end    = (i + 1) * samples_per_env if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]

        env_texts  = [all_texts[int(j)]  for j in env_indices]
        env_labels = [all_labels[int(j)] for j in env_indices]
        print(f"  Partition : {len(env_texts)} exemples")

        selected_texts:  List[str] = []
        selected_labels: List[int] = []

        for text, label in zip(env_texts, env_labels):
            if is_typical_sst2(text, label):
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            else:
                # Collecte OOD uniquement dans env 0
                if i == 0:
                    if ood_strategy == 'cross_label':
                        # Exemple où le lexique contredit le label → adversarial pour ERM
                        if is_cross_label_sst2(text, label):
                            ood_texts.append(text)
                            ood_labels.append(label)
                    else:  # 'atypical'
                        # Exemple sans aucun mot fort (ni du bon ni du mauvais label)
                        if not is_cross_label_sst2(text, label):
                            ood_texts.append(text)
                            ood_labels.append(label)

        print(f"  Sélectionné : {len(selected_texts)} critiques typiques")

        # ── Split train/val ──────────────────────────────────────────────
        n_sel = len(selected_texts)
        n_val = int(n_sel * val_frac)
        idx_sh = rng.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [selected_texts[j]  for j in tr_idx]
        tr_labels = np.array([selected_labels[j] for j in tr_idx])
        if label_flip > 0.0:
            rng_train_flip = np.random.default_rng(seed + 9000 + i)
            flip_mask = rng_train_flip.uniform(size=len(tr_labels)) < label_flip
            tr_labels[flip_mask] = 1 - tr_labels[flip_mask]
        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling)
        Y_tr = tr_labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(
            torch.from_numpy(X_tr), torch.from_numpy(Y_tr),
            meta={"p_select": p_select, "kind": "sst2_selection_train",
                  "env_id": i, "n_samples": len(X_tr), "dataset": "sst2",
                  "label_flip": label_flip},
        ))

        va_texts  = [selected_texts[j]  for j in va_idx]
        va_labels = np.array([selected_labels[j] for j in va_idx])
        if label_flip > 0.0:
            rng_val_flip = np.random.default_rng(seed + 10000 + i)
            flip_mask = rng_val_flip.uniform(size=len(va_labels)) < label_flip
            va_labels[flip_mask] = 1 - va_labels[flip_mask]
        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling)
        Y_va = va_labels.reshape(-1, 1).astype(np.float32)
        val_envs.append(Env(
            torch.from_numpy(X_va), torch.from_numpy(Y_va),
            meta={"p_select": p_select, "kind": "sst2_selection_val",
                  "env_id": i, "n_samples": len(X_va), "dataset": "sst2",
                  "label_flip": label_flip},
        ))

    # ── Test OOD ──────────────────────────────────────────────────────────
    if ood_strategy == 'cross_label':
        ood_desc = "cross_label : mots forts contredisent le label (adversarial ERM)"
    else:
        ood_desc = "atypical : aucun mot fort (signal spurieux absent)"
    print(f"\n=== Test OOD SST-2 ({ood_strategy}) ===")
    print(f"  {len(ood_texts)} critiques — {ood_desc}")

    ood_labels_arr = np.array(ood_labels)
    X_test = tokenize_and_embed_with_bert(ood_texts, bert_model, max_length, device, pooling)
    Y_test = ood_labels_arr.reshape(-1, 1).astype(np.float32)

    test_env = Env(
        torch.from_numpy(X_test), torch.from_numpy(Y_test),
        meta={"kind": "sst2_selection_test_ood", "ood_strategy": ood_strategy,
              "n_samples": len(X_test), "dataset": "sst2", "description": ood_desc},
    )

    print(f"\n✅ SST-2 Selection — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} critiques typiques)")
    print(f"   - {n_envs} envs val    ({sum(e.X.shape[0] for e in val_envs)} critiques)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} critiques)")

    return train_envs, val_envs, test_env


# =============================================================================
# SST-2 — Sélection par genre de film (corrélation naturelle genre ↔ note)
# =============================================================================
#
# DAG : Y → Z (genre du film) → S (sélection)
#
# Corrélation naturelle dans Rotten Tomatoes :
#   documentaire / indie (Z=pos) → critique positive (Y=1)
#   suite / remake       (Z=neg) → critique négative (Y=0)
#
# Mécanisme de sélection :
#   Typique  : critique positive + vocabulaire doc/indie
#              critique négative + vocabulaire suite/remake
#   OOD      : critique positive + vocabulaire suite/remake (bonne suite)
#              critique négative + vocabulaire doc/indie (mauvais documentaire)
#
# Avantage vs. sélection par lexique de sentiment :
#   - Les mots de genre ne sont pas directement porteurs de sentiment dans BERT
#   - La corrélation est causalement vide (le type ne CAUSE pas le sentiment)
#   - L'OOD est bien défini et plus fréquent que le cas cross_label
# =============================================================================

def build_envs_sst2_genre_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — expérience de sélection par genre de film.

    DAG : Y → Z (genre) → S (sélection d'entraînement)

    Mécanisme :
      - Typique (Z cohérent) : critique positive sur doc/indie ou négative sur suite/remake
        → sélectionné avec proba p_select dans les envs de train.
      - Test OOD (cross_genre) : critique positive sur suite/remake ou négative sur doc/indie
        → exemples où le genre contredit la note → adversarial pour ERM.

    Parameters
    ----------
    train_p_select : List[float]
        Proba de garder un exemple typique par env (ex : [0.9, 0.7]).
    label_flip : float
        Taux de bruit symétrique sur les labels train/val.
    seed, val_frac, bert_model, max_length, device, pooling : cf. hab.
    """
    print("Chargement du dataset SST-2 (genre selection)...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} critiques")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    samples_per_env = n_total // n_envs

    ood_texts:  List[str] = []
    ood_labels: List[int] = []
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")

        env_start   = i * samples_per_env
        env_end     = (i + 1) * samples_per_env if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]

        env_texts  = [all_texts[int(j)]  for j in env_indices]
        env_labels = [all_labels[int(j)] for j in env_indices]
        print(f"  Partition : {len(env_texts)} exemples")

        selected_texts:  List[str] = []
        selected_labels: List[int] = []

        n_typical = 0
        for text, label in zip(env_texts, env_labels):
            if is_typical_genre_sst2(text, label):
                n_typical += 1
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            else:
                # Collecte OOD uniquement depuis env 0
                if i == 0 and is_cross_genre_sst2(text, label):
                    ood_texts.append(text)
                    ood_labels.append(label)

        print(f"  Typiques (genre cohérent) : {n_typical} | Sélectionnés : {len(selected_texts)}")

        # ── Split train/val ──────────────────────────────────────────────
        n_sel = len(selected_texts)
        n_val = int(n_sel * val_frac)
        idx_sh = rng.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [selected_texts[j]  for j in tr_idx]
        tr_labels = np.array([selected_labels[j] for j in tr_idx])
        if label_flip > 0.0:
            rng_tr = np.random.default_rng(seed + 12000 + i)
            mask = rng_tr.uniform(size=len(tr_labels)) < label_flip
            tr_labels[mask] = 1 - tr_labels[mask]
        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling)
        Y_tr = tr_labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(
            torch.from_numpy(X_tr), torch.from_numpy(Y_tr),
            meta={"p_select": p_select, "kind": "sst2_genre_selection_train",
                  "env_id": i, "n_samples": len(X_tr), "label_flip": label_flip},
        ))

        va_texts  = [selected_texts[j]  for j in va_idx]
        va_labels = np.array([selected_labels[j] for j in va_idx])
        if label_flip > 0.0:
            rng_va = np.random.default_rng(seed + 13000 + i)
            mask = rng_va.uniform(size=len(va_labels)) < label_flip
            va_labels[mask] = 1 - va_labels[mask]
        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling)
        Y_va = va_labels.reshape(-1, 1).astype(np.float32)
        val_envs.append(Env(
            torch.from_numpy(X_va), torch.from_numpy(Y_va),
            meta={"p_select": p_select, "kind": "sst2_genre_selection_val",
                  "env_id": i, "n_samples": len(X_va), "label_flip": label_flip},
        ))

    # ── Test OOD ──────────────────────────────────────────────────────────
    print(f"\n=== Test OOD SST-2 genre (cross_genre) ===")
    n_pos_ood = sum(1 for l in ood_labels if l == 1)
    n_neg_ood = sum(1 for l in ood_labels if l == 0)
    print(f"  {len(ood_texts)} critiques : {n_pos_ood} positives-sur-suite, {n_neg_ood} négatives-sur-doc")

    ood_labels_arr = np.array(ood_labels)
    X_test = tokenize_and_embed_with_bert(ood_texts, bert_model, max_length, device, pooling)
    Y_test = ood_labels_arr.reshape(-1, 1).astype(np.float32)
    test_env = Env(
        torch.from_numpy(X_test), torch.from_numpy(Y_test),
        meta={"kind": "sst2_genre_selection_test_ood", "n_samples": len(X_test),
              "description": "cross_genre : genre contredit le label"},
    )

    print(f"\n✅ SST-2 Genre Selection — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} critiques typiques)")
    print(f"   - {n_envs} envs val    ({sum(e.X.shape[0] for e in val_envs)} critiques)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} critiques cross-genre)")
    return train_envs, val_envs, test_env


# =============================================================================
# AG News — Sélection basée sur les sources journalistiques (multiclasse)
# =============================================================================
#
# DAG : Y → Z (source) → S (sélection)
#
# Classes AG News :  0=World, 1=Sports, 2=Business, 3=Sci/Tech
#
# Mapping source ↔ label (choisi selon la corrélation naturelle dans AG News) :
#   World   (0) ↔ "(AFP)"       AFP = agence internationale → couvre le monde
#   Business(2) ↔ "(Reuters)"   Reuters = agence finance/éco → couvre business
#   Sci/Tech(3) ↔ "(AP)"        AP = agence US généraliste → couvre tech/sci
#   Sports  (1) ↔  rien         pas de source spurieuse → toujours typique
#
# - Typique   : l'article contient la source assignée à son label (ou Sports).
#             → sélectionné en train avec probabilité p_select.
#             → Sports plafonné pour équilibrage inter-classes.
# - Atypique  : l'article n'a pas la source de son label
#             → pool de test OOD, plafonné à n_ood_per_class exemples/classe.
# =============================================================================

# Mapping label → source spurieuse (None = pas de source assignée)
AG_NEWS_LABEL_TO_SOURCE: Dict[int, Optional[str]] = {
    0: "AFP",       # World   — AFP couvre les événements mondiaux
    1: None,        # Sports  — toujours typique, aucune source spurieuse
    2: "Reuters",   # Business — Reuters est très présent en finance/éco
    3: "AP",        # Sci/Tech — AP couvre la tech et la science
}

# Noms lisibles des classes AG News (pour logs et diagnostics)
AG_NEWS_CLASS_NAMES: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


# =============================================================================
# Chargement du dataset AG News (4 classes)
# =============================================================================

def load_ag_news_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge le dataset AG News depuis Hugging Face (4 classes originales).

    Dataset : fancyzhx/ag_news — 127 600 train + 7 600 test.
    Classes : 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("fancyzhx/ag_news")
    all_data = concatenate_datasets([dataset["train"], dataset["test"]])
    all_data = all_data.shuffle(seed=seed)

    texts = list(all_data["text"])
    labels = list(all_data["label"])  # déjà 0-3

    return texts, labels


# =============================================================================
# Diagnostic cross-table source × classe
# =============================================================================

def diagnose_ag_news_sources(
    texts: List[str],
    labels: List[int],
    sources: Optional[List[str]] = None,
) -> None:
    """
    Affiche une cross-table (source × classe) sur un échantillon du dataset.
    Utile pour choisir le mapping optimal source ↔ label.
    """
    if sources is None:
        sources = ["AFP", "Reuters", "AP", "cnn", "Sports Network"]

    class_names = [AG_NEWS_CLASS_NAMES[c] for c in range(4)]

    # Comptage
    counts: Dict[str, Dict[int, int]] = {s: {c: 0 for c in range(4)} for s in sources}
    total_per_class: Dict[int, int] = {c: 0 for c in range(4)}

    for text, label in zip(texts, labels):
        total_per_class[label] += 1
        for src in sources:
            if re.search(r"\(" + re.escape(src) + r"\)", text, re.IGNORECASE):
                counts[src][label] += 1

    # Affichage
    col_w = 10
    header = f"{'Source':15s}" + "".join(f"{n:>{col_w}s}" for n in class_names)
    print("\n  Cross-table : articles contenant (Source) par classe")
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for src in sources:
        row = f"  ({src}):".ljust(15)
        for c in range(4):
            n = counts[src][c]
            pct = 100 * n / max(total_per_class[c], 1)
            row += f"{n:>{col_w-4}d} ({pct:4.1f}%)"[: col_w]
        print(row)
    print("  " + "-" * len(header))
    totals_row = f"  {'Total':15s}" + "".join(
        f"{total_per_class[c]:>{col_w}d}" for c in range(4)
    )
    print(totals_row)
    print()


# =============================================================================
# Détection de typicité pour AG News (multiclasse)
# =============================================================================

def is_typical_ag_news(
    text: str,
    label: int,
    label_to_source: Optional[Dict[int, Optional[str]]] = None,
) -> bool:
    """
    Indique si un article est "typique" selon la corrélation source↔label.

    - Label avec source assignée (World/Business/Sci/Tech) :
      typique ssi le texte contient ``(Source assignée)``.
    - Sports (label sans source) :
      typique ssi le texte ne contient AUCUNE source assignée aux autres classes.
      → un article Sports avec (AFP), (Reuters) ou (AP) est atypique
        (signal spurieux contradictoire → pool OOD).
    """
    if label_to_source is None:
        label_to_source = AG_NEWS_LABEL_TO_SOURCE

    source = label_to_source.get(label)

    if source is None:
        # Sports : atypique si l'article contient une source d'une autre classe
        other_sources = [s for s in label_to_source.values() if s is not None]
        for s in other_sources:
            if re.search(r"\(" + re.escape(s) + r"\)", text, re.IGNORECASE):
                return False  # source spurieuse présente → OOD
        return True

    return bool(re.search(r"\(" + re.escape(source) + r"\)", text, re.IGNORECASE))


# =============================================================================
# Construction d'environnements AG News avec selection bias (multiclasse)
# =============================================================================

def build_envs_ag_news_source_selection(
    train_p_select: List[float],
    seed: int,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    n_ood_per_class: int = 250,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit des environnements AG News multiclasse avec selection bias.

    **DAG** : Y → Z (source) → S (sélection)

    **Mapping source ↔ label** (corrélation naturelle dans AG News) :
      - World   (0) ↔ "(AFP)"       AFP = agence internationale
      - Business(2) ↔ "(Reuters)"   Reuters = agence finance/éco
      - Sci/Tech(3) ↔ "(AP)"        AP = agence US
      - Sports  (1) ↔ rien          toujours typique

    **Équilibrage** :
      - Sports est plafonné dans chaque env à la somme des autres classes
        sélectionnées, pour éviter qu'il domine l'entraînement.
      - L'OOD est plafonné à `n_ood_per_class` exemples par classe.

    Parameters
    ----------
    train_p_select : List[float]
        Ex : [0.9, 0.7] → env 0 garde 90% des typiques, env 1 garde 70%.
    seed : int
        Graine aléatoire.
    val_frac : float
        Fraction de validation.
    label_flip : float
        Taux de bruit multiclasse sur les labels train/val. Applique apres la
        selection, de sorte que la structure de selection reste intacte.
    n_ood_per_class : int
        Nombre maximal d'exemples OOD par classe (défaut 200).
    bert_model, max_length, device, pooling : str / int
        Config BERT.

    Returns
    -------
    train_envs, val_envs, test_env
        Labels en **torch.long** (LongTensor, valeurs 0–3) pour CrossEntropyLoss.
    """
    print("Chargement du dataset AG News (4 classes)...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} articles (0=World, 1=Sports, 2=Business, 3=Sci/Tech)")

    # ── Diagnostic cross-table ──
    diagnose_ag_news_sources(all_texts, all_labels)

    # ── Comptage typiques ──
    counts = {k: 0 for k in range(4)}
    n_typical_per_class = {k: 0 for k in range(4)}
    for text, label in zip(all_texts, all_labels):
        counts[label] += 1
        if is_typical_ag_news(text, label):
            n_typical_per_class[label] += 1

    print("  Répartition typiques / atypiques par classe (mapping actuel) :")
    for cls, name in AG_NEWS_CLASS_NAMES.items():
        src = AG_NEWS_LABEL_TO_SOURCE[cls] or "—"
        n_typ = n_typical_per_class[cls]
        n_all = counts[cls]
        print(f"    {name:10s} (→ {src:8s}) : {n_typ:6d}/{n_all:6d} typiques "
              f"({100*n_typ/max(n_all,1):.1f}%)")

    rng = np.random.default_rng(seed)

    # Pools OOD séparés par classe (pour le plafonnement)
    ood_by_class: Dict[int, List[str]] = {c: [] for c in range(4)}

    n_envs = len(train_p_select)
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    # ───────────────── SÉLECTION PAR ENVIRONNEMENT ─────────────────────────
    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")

        # Collecter les typiques par classe séparément
        sel_by_class: Dict[int, List[str]] = {c: [] for c in range(4)}

        for text, label in zip(all_texts, all_labels):
            if is_typical_ag_news(text, label):
                if rng.uniform() < p_select:
                    sel_by_class[label].append(text)
            else:
                if i == 0:
                    ood_by_class[label].append(text)

        # ── Équilibrage : Sports capé au max des autres classes individuelles ──
        # max(World, Business, Sci/Tech) pour éviter que Sports domine
        n_per_nonsports = {c: len(sel_by_class[c]) for c in [0, 2, 3]}
        cap_sports = max(n_per_nonsports.values())
        if len(sel_by_class[1]) > cap_sports:
            sports_idx = rng.choice(len(sel_by_class[1]), size=cap_sports, replace=False)
            sel_by_class[1] = [sel_by_class[1][j] for j in sports_idx]

        sel_texts:  List[str] = []
        sel_labels: List[int] = []
        for c in range(4):
            sel_texts  += sel_by_class[c]
            sel_labels += [c] * len(sel_by_class[c])

        sel_arr = np.array(sel_labels)
        dist = {AG_NEWS_CLASS_NAMES[c]: int((sel_arr == c).sum()) for c in range(4)}
        print(f"  Sélectionné (Sports ≤ max autre classe) : {len(sel_texts)} → {dist}")

        # ── Split train / val ──────────────────────────────────────────────
        n_sel = len(sel_texts)
        idx = rng.permutation(n_sel)
        n_val   = int(n_sel * val_frac)
        tr_idx  = idx[n_val:]
        val_idx = idx[:n_val]

        def _make_env(indices, kind, p_sel=p_select, env_i=i):
            texts_e  = [sel_texts[j]  for j in indices]
            labels_e = np.array([sel_labels[j] for j in indices], dtype=np.int64)
            if label_flip > 0.0:
                rng_flip = np.random.default_rng(
                    seed + 11000 + env_i * 17 + (0 if kind == "train" else 1)
                )
                flip_mask = rng_flip.uniform(size=len(labels_e)) < label_flip
                for k in np.where(flip_mask)[0]:
                    others = [c for c in range(4) if c != labels_e[k]]
                    labels_e[k] = int(rng_flip.choice(others))
            X = tokenize_and_embed_with_bert(texts_e, bert_model, max_length, device, pooling)
            return Env(
                torch.from_numpy(X),
                torch.from_numpy(labels_e),
                meta={
                    "p_select": p_sel,
                    "kind": f"ag_news_source_selection_{kind}",
                    "env_id": env_i,
                    "n_classes": 4,
                    "label_flip": label_flip,
                    "n_samples": len(X),
                },
            )

        train_envs.append(_make_env(tr_idx,  "train"))
        val_envs.append(  _make_env(val_idx, "val"))

    # ──────────────────────── TEST OOD ─────────────────────────────────────
    print(f"\n=== Test OOD (plafonné à {n_ood_per_class} exemples/classe) ===")

    ood_texts_final:  List[str] = []
    ood_labels_final: List[int] = []

    for cls in range(4):
        pool = ood_by_class[cls]
        if not pool:
            continue  # Sports n'a pas d'OOD
        n_take = min(len(pool), n_ood_per_class)
        chosen = rng.choice(len(pool), size=n_take, replace=False)
        for j in chosen:
            ood_texts_final.append(pool[j])
            ood_labels_final.append(cls)

    ood_arr  = np.array(ood_labels_final)
    ood_dist = {AG_NEWS_CLASS_NAMES[c]: int((ood_arr == c).sum()) for c in range(4)}
    print(f"  {len(ood_texts_final)} articles OOD → {ood_dist}")
    print(f"  Signal spurieux Z absent → ERM piégé, IRM attendu robuste.")

    X_ood = tokenize_and_embed_with_bert(ood_texts_final, bert_model, max_length, device, pooling)
    test_env = Env(
        torch.from_numpy(X_ood),
        torch.from_numpy(ood_arr.astype(np.int64)),
        meta={
            "kind": "ag_news_source_selection_ood",
            "n_classes": 4,
            "n_samples": len(X_ood),
            "n_ood_per_class": n_ood_per_class,
            "description": "articles_without_assigned_source",
        },
    )

    print(f"\n✅ AG News multiclasse — selection bias (source-based) !")
    print(f"   - {len(train_envs)} envs train  "
          f"({sum(e.X.shape[0] for e in train_envs)} articles, équilibrés)")
    print(f"   - {len(val_envs)} envs val    "
          f"({sum(e.X.shape[0] for e in val_envs)} articles)")
    print(f"   - 1 env test OOD ({test_env.X.shape[0]} articles, ≤{n_ood_per_class}/classe)")

    return train_envs, val_envs, test_env


# =============================================================================
# AG News — Expérience semi anti-causale (injection de tokens, 4 classes)
# =============================================================================
#
# DAG : Y (classe réelle) → Z (token injecté) → X = BERT(texte + Z)
#
# Mécanisme :
#   Pour chaque article on injecte UN token de classe en préfixe :
#     - avec proba p_correct   : le token du vrai label
#     - avec proba 1-p_correct : un token tiré uniformément parmi les 3 autres
#
#   En train  : p_correct ∈ {0.9, 0.8} → forte corrélation Z–Y
#   En test OOD : p_correct = 0.0 → token toujours erroné → ERM piégé
#
# Tokens BERT (chacun tokenisé comme UN SEUL token, rares dans les news) :
#   alpha=World(0)  beta=Sports(1)  gamma=Business(2)  delta=Sci/Tech(3)
# =============================================================================

# Mapping classe AG News → token spurieux
# Tokens spurieux pour AG News — couleurs primaires/secondaires.
# Un token par classe, aucune association avec les catégories de news :
# "red" n'est pas une catégorie journalistique, idem pour blue/green/yellow.
# Chaque couleur est un token BERT unique avec un embedding bien défini.
AG_NEWS_TOKENS: Dict[int, str] = {
    0: "red",      # World
    1: "blue",     # Sports
    2: "green",    # Business
    3: "yellow",   # Sci/Tech
}


def inject_spurious_token_multiclass(
    text: str,
    label: int,
    p_correct: float,
    class_tokens: Dict[int, str],
    rng: np.random.Generator,
    position: str = "prefix",
    neutral_words: Optional[List[str]] = NEUTRAL_WORDS,
) -> str:
    """
    Injecte un token spurieux pour une classification multiclasse (N classes).

    - Avec proba p_correct   : injecte ``class_tokens[label]``.
    - Avec proba 1-p_correct : injecte un token tiré uniformément parmi
      les (N-1) autres classes.

    Par défaut, insère le token devant chaque mot neutre du texte (conservé).
    Fallback préfixe de phrase si aucun mot neutre présent.
    Passer neutral_words=None pour forcer le mode préfixe seul.
    """
    if rng.uniform() < p_correct:
        token = class_tokens[label]
    else:
        others = [k for k in class_tokens if k != label]
        token = class_tokens[int(rng.choice(others))]

    if neutral_words is not None:
        prepended = _prepend_token_to_neutral_words(text, neutral_words, token)
        if prepended is not None:
            return prepended

    return f"{token} {text}" if position == "prefix" else f"{text} {token}"


def build_envs_ag_news_semi_anti_causal(
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    label_flip: float = 0.25,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    AG News multiclasse — expérience semi anti-causale par injection de tokens.

    DAG : Y → Z (token) → X = BERT(texte + Z)

    Un token de classe est injecté dans chaque article avant l'encodage BERT.
    La corrélation entre le token Z et le label Y est contrôlée par p_correct.
    Un modèle ERM qui exploite Z sera mis en défaut sur le test OOD (token
    toujours erroné). IRM doit apprendre à ignorer Z.

    Split global : 80 % train | 10 % val | 10 % test
    Chaque valeur de train_p_correct génère un environnement d'entraînement
    distinct (les exemples train sont répartis équitablement entre les envs).

    Parameters
    ----------
    train_p_correct : List[float]
        Corrélation Z–Y par env train (ex : [0.9, 0.8]).
    test_p_correct : float
        Corrélation en test OOD (0.0 → token toujours erroné).
    seed : int
        Graine aléatoire globale.
    label_flip : float
        Fraction de labels bruités en train (affaiblit le signal causal).
        Pour chaque label bruité, une autre classe est tirée uniformément.
    bert_model : str
        Modèle BERT Hugging Face (défaut : bert-base-uncased).
    max_length : int
        Longueur max de séquence BERT (256 recommandé pour AG News).
    device : str
        Device PyTorch ("cpu", "cuda", "mps").
    pooling : str
        Stratégie de pooling BERT ("mean", "cls", "max").

    Returns
    -------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset AG News...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    print(f"Dataset chargé : {n_total} articles (4 classes)")
    class_dist = {AG_NEWS_CLASS_NAMES[c]: int((all_labels_arr == c).sum()) for c in range(4)}
    print(f"  Distribution : {class_dist}")

    # ── Split global 80/10/10 ──────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"\nSplit : Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")

    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    # ── Envs train + val ───────────────────────────────────────────────────
    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start   = i * samples_per_env
        end     = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        texts  = [all_texts[int(j)]  for j in env_idx]
        labels = np.array([all_labels[int(j)] for j in env_idx], dtype=np.int64)

        # Label flip multiclasse (bruitage signal causal texte→label)
        if label_flip > 0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels)) < label_flip
            for k in np.where(flip_mask)[0]:
                others = [c for c in range(4) if c != labels[k]]
                labels[k] = int(rng_flip.choice(others))

        # Injection de token spurieux
        rng_inject = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, AG_NEWS_TOKENS, rng_inject)
            for t, l in zip(texts, labels)
        ]
        n_correct = sum(
            AG_NEWS_TOKENS[int(l)] in tm.lower().split()
            for tm, l in zip(texts_mod, labels)
        )
        print(f"  Token correct : {n_correct}/{len(labels)} ({n_correct/len(labels):.1%})")

        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling)
        train_envs.append(Env(
            torch.from_numpy(X),
            torch.from_numpy(labels),
            meta={
                "p_correct": p_correct,
                "label_flip": label_flip,
                "n_classes": 4,
                "kind": "ag_news_semi_anti_causal_train",
                "n_samples": len(X),
            },
        ))

        # ── Val env (même p_correct, même label_flip que le train) ──────────────────
        print(f"=== Val Env {i} (p_correct={p_correct:.0%}) ===")
        val_texts  = [all_texts[int(j)]  for j in val_indices]
        val_labels = np.array([all_labels[int(j)] for j in val_indices], dtype=np.int64)

        # Même label_flip multiclasse que le train
        if label_flip > 0:
            rng_val_flip = np.random.default_rng(seed + 5000 + i + 1)
            flip_mask_val = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            for k in np.where(flip_mask_val)[0]:
                others = [c for c in range(4) if c != val_labels[k]]
                val_labels[k] = int(rng_val_flip.choice(others))

        rng_val = np.random.default_rng(seed + 5000 + i)
        val_texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, AG_NEWS_TOKENS, rng_val)
            for t, l in zip(val_texts, val_labels)
        ]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling)
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(val_labels),
            meta={
                "p_correct": p_correct,
                "n_classes": 4,
                "kind": "ag_news_semi_anti_causal_val",
                "n_samples": len(X_val),
            },
        ))

    # ── Test OOD ──────────────────────────────────────────────────────────
    print(f"\n=== Test OOD (p_correct={test_p_correct:.0%}) ===")
    if test_p_correct == 0.0:
        print("  Token toujours erroné → ERM piégé, IRM attendu robuste.")

    test_texts  = [all_texts[int(j)]  for j in test_indices]
    test_labels = np.array([all_labels[int(j)] for j in test_indices], dtype=np.int64)

    rng_test = np.random.default_rng(seed + 777)
    test_texts_mod = [
        inject_spurious_token_multiclass(t, int(l), test_p_correct, AG_NEWS_TOKENS, rng_test)
        for t, l in zip(test_texts, test_labels)
    ]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling)

    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(test_labels),
        meta={
            "p_correct": test_p_correct,
            "n_classes": 4,
            "kind": "ag_news_semi_anti_causal_test_ood",
            "n_samples": len(X_test),
        },
    )

    print(f"\n✅ AG News Semi Anti-Causal — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} articles)")
    print(f"   - {n_envs} envs val    ({val_envs[0].X.shape[0]} articles/env)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} articles, p_correct={test_p_correct:.0%})")

    return train_envs, val_envs, test_env


# =============================================================================
# Confounding — helper SST-2 (tokens binaires)
# =============================================================================
# Tokens binaires dédiés au confounding SST-2 (réutilise SST2_TOKENS).
_SST2_CONF_TOKENS: Dict[str, str] = {
    "ham_correlated":  SST2_TOKENS[0],   # "north" (label 0 = négatif)
    "spam_correlated": SST2_TOKENS[1],   # "south" (label 1 = positif)
}


# =============================================================================
# AG News — Confounding variant 1 : varying proxy (multiclasse)
# =============================================================================
# DAG : C ∈ {0,1,2,3}\{Y}  avec prob p_c_flip  (confondeur multiclasse)
#       Y_obs = C  si C fire,  sinon Y_obs = Y
#       Z_init = Y_obs
#       Z = Z_init  avec prob (1-a_e),  sinon classe aléatoire ≠ Z_init
#       token = AG_NEWS_TOKENS[Z]   (red/blue/green/yellow — 4 tokens distincts)
#
# Garantie : token corrélé à Y_obs en train (a_e≈0), anti-corrélé en OOD (a_e≈1).
# Évaluation OOD : labels vrais (Y, pas Y_obs) pour mesure propre.
# Variation d'env : a_e.
# =============================================================================

def _conf_ag_news_make_env(
    texts: List[str],
    labels: np.ndarray,        # int64 vrais labels (avant bruitage)
    a_e: float,                # bruit sur Z (0=clean, 1=entièrement bruité)
    rng: np.random.Generator,
    bert_model: str,
    max_length: int,
    device: str,
    pooling: str,
    p_c_flip: float,
    apply_label_flip: bool = True,  # False pour test OOD → retourne vrais labels
    n_classes: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un environnement AG News confounding (multiclasse).

    Mécanisme :
      1. Pour chaque sample, avec prob p_c_flip : C ~ Uniform({0..3}\\{Y})
         → Y_obs = C ;  sinon Y_obs = Y.
      2. Z_init = Y_obs.
      3. Bruit a_e : avec prob a_e, Z = classe aléatoire ≠ Z_init.
      4. token injecté = AG_NEWS_TOKENS[Z].
      5. Si apply_label_flip=False (test OOD), retourne les vrais labels.
    """
    n = len(labels)

    # ── Étape 1 : bruitage des labels (C multiclasse) ──────────────────────
    Y_obs = labels.copy()
    flip_mask = rng.uniform(size=n) < p_c_flip
    for k in np.where(flip_mask)[0]:
        others = [c for c in range(n_classes) if c != labels[k]]
        Y_obs[k] = int(rng.choice(others))

    # ── Étape 2-3 : calcul de Z avec bruit a_e ─────────────────────────────
    Z = Y_obs.copy()
    noise_mask = rng.uniform(size=n) < a_e
    for k in np.where(noise_mask)[0]:
        others = [c for c in range(n_classes) if c != Z[k]]
        Z[k] = int(rng.choice(others))

    # ── Étape 4 : injection du token spurieux ──────────────────────────────
    rng_inj = np.random.default_rng(int(rng.integers(0, 2**31)))
    texts_mod = [
        inject_spurious_token_multiclass(text, int(z), 1.0, AG_NEWS_TOKENS, rng_inj)
        for text, z in zip(texts, Z)
    ]

    X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling)
    Y_out = Y_obs if apply_label_flip else labels
    return X, Y_out.astype(np.int64)


def build_envs_ag_news_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    AG News — confounding multiclasse avec variation du proxy.

    DAG : C ∈ {0..3}\\{Y} avec prob p_c_flip → Y_obs = C, Z_init = C
          Z = Z_init XOR bruit(a_e) → token = AG_NEWS_TOKENS[Z]

    Le token est directement indexé sur la classe (red=World, blue=Sports, …)
    ce qui garantit une corrélation spurieuse exploitable par ERM en train
    (a_e≈0) et une rupture nette en OOD (a_e≈1).

    Parameters
    ----------
    a_train  : List[float]  Bruit proxy par env train (ex : [0.01, 0.1]).
    a_test   : float        Bruit proxy OOD (ex : 0.99).
    p_c_flip : float        Probabilité de flip de label (défaut 0.25).
    """
    print("Chargement du dataset AG News (confounding – varying proxy)...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    rng_split = np.random.default_rng(seed)
    indices = rng_split.permutation(n_total)
    n_test = int(n_total * 0.1)
    n_val  = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} articles | Split 80/10/10 : "
          f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    val_texts  = [all_texts[int(j)] for j in val_idx]
    val_labels = all_labels_arr[val_idx]

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[int(j)] for j in env_idx]
        labels = all_labels_arr[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        X, Y = _conf_ag_news_make_env(
            texts, labels, a_e, rng_e,
            bert_model, max_length, device, pooling, p_c_flip,
            apply_label_flip=True,
        )
        train_envs.append(Env(
            torch.from_numpy(X), torch.from_numpy(Y),
            meta={"kind": "ag_news_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                  "split": "train", "env_id": i, "n_samples": len(X), "n_classes": 4},
        ))

        print(f"=== Val Env {i} (a={a_e}) ===")
        rng_v = np.random.default_rng(seed + 5000 + i)
        X_val, Y_val = _conf_ag_news_make_env(
            val_texts, val_labels, a_e, rng_v,
            bert_model, max_length, device, pooling, p_c_flip,
            apply_label_flip=True,
        )
        val_envs.append(Env(
            torch.from_numpy(X_val), torch.from_numpy(Y_val),
            meta={"kind": "ag_news_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                  "split": "val", "env_id": i, "n_samples": len(X_val), "n_classes": 4},
        ))

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[int(j)] for j in test_idx]
    test_labels = all_labels_arr[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    X_test, Y_test = _conf_ag_news_make_env(
        test_texts, test_labels, a_test, rng_t,
        bert_model, max_length, device, pooling, p_c_flip,
        apply_label_flip=False,  # évaluation sur vrais labels
    )
    test_env = Env(
        torch.from_numpy(X_test), torch.from_numpy(Y_test),
        meta={"kind": "ag_news_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
              "split": "test_ood", "n_samples": len(X_test), "n_classes": 4},
    )

    print(f"\n✅ AG News Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# SST-2 — Confounding variant 1 : varying proxy
# =============================================================================
# Même DAG que SMS Spam confounding mais sur SST-2 (binaire, anti-causal Y→X).
# DAG : C ~ Ber(p_c_flip) → Z(a_e) = C XOR Ber(a_e) → token (north/south)
#       C → Y (flip déterministe si C=1)
#       texte → Y
# =============================================================================

def build_envs_sst2_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — confounding avec variation du proxy Z = C XOR Ber(a_e).

    Même DAG que nlp_sms_spam_conf_varying_proxy mais sur SST-2 (anti-causal).
    Les tokens spurieux binaires sont "north" (label 0) et "south" (label 1),
    cohérents avec les tokens SAC de SST-2.

    Parameters
    ----------
    a_train  : List[float]  Bruit proxy par env train (ex : [0.01, 0.1]).
    a_test   : float        Bruit proxy OOD (ex : 0.99).
    p_c_flip : float        P(C=1) = fraction des labels flippés (défaut 0.25).
                            Le flip est déterministe : C=1 ⟹ label toujours inversé.
    """
    print("Chargement du dataset SST-2 (confounding – varying proxy)...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    rng_split = np.random.default_rng(seed)
    indices = rng_split.permutation(n_total)
    n_test = int(n_total * 0.1)
    n_val  = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} reviews | Split 80/10/10 : "
          f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[int(j)]  for j in env_idx]
        labels = all_labels_arr[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c_flip, size=len(labels))
        Y_obs = np.where(C == 1, 1 - labels, labels).astype(int)
        N = rng_e.binomial(1, a_e,  size=len(labels))
        Z = np.logical_xor(Y_obs, N).astype(int)

        X, Y = _conf_make_env(
            texts, labels.astype(np.float32), C, Z, 1.0, rng_e,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_SST2_CONF_TOKENS,
        )
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "sst2_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (a={a_e}) ===")
        val_texts  = [all_texts[int(j)]  for j in val_idx]
        val_labels = all_labels_arr[val_idx]
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c_flip, size=len(val_labels))
        Y_obs_v = np.where(Cv == 1, 1 - val_labels, val_labels).astype(int)
        Nv = rng_v.binomial(1, a_e,  size=len(val_labels))
        Zv = np.logical_xor(Y_obs_v, Nv).astype(int)
        X_val, Y_val = _conf_make_env(
            val_texts, val_labels.astype(np.float32), Cv, Zv, 1.0, rng_v,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_SST2_CONF_TOKENS,
        )
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "sst2_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[int(j)]  for j in test_idx]
    test_labels = all_labels_arr[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, p_c_flip, size=len(test_labels))
    Y_obs_t = np.where(Ct == 1, 1 - test_labels, test_labels).astype(int)
    Nt = rng_t.binomial(1, a_test, size=len(test_labels))
    Zt = np.logical_xor(Y_obs_t, Nt).astype(int)
    X_test, Y_test = _conf_make_env(
        test_texts, test_labels.astype(np.float32), Ct, Zt, 1.0, rng_t,
        bert_model, max_length, device, pooling,
        apply_gamma=False, conf_tokens=_SST2_CONF_TOKENS,
    )
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "sst2_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ SST-2 Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env
